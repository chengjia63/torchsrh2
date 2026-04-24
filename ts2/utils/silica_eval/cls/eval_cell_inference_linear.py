import json
import logging
import os
import shutil
from itertools import product
from os.path import join as opj
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from ts2.utils.silica_sc_cls.eval_cell_inference_knn import (
    aggregate_mosaic_votes,
    build_balanced_binary_predictions,
    build_instance_dataframe,
    build_runs_from_sets,
    encode_prediction_labels,
    get_all_metrics,
    get_binary_metrics,
    infer_results_dir_from_prediction_path,
    load_labels_from_cell_instances,
    load_prediction,
    plot_confusion,
)


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings.to(torch.float32), p=2, dim=1)


def build_metric_summary(
    split_name: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    id_col: str,
) -> Dict[str, Any]:
    metrics = (
        get_binary_metrics(logits, labels)
        if len(class_names) == 2
        else get_all_metrics(logits, labels, num_classes=len(class_names))
    )
    cm = confusion_matrix(
        labels.numpy(),
        logits.argmax(dim=1).numpy(),
        labels=np.arange(len(class_names)),
    )
    return {
        "split": split_name,
        "num_samples": int(labels.shape[0]),
        "id_column": id_col,
        "metrics": metrics,
        "cm": cm.tolist(),
    }


def evaluate_logits(
    pred: Dict[str, Any],
    logits: torch.Tensor,
    class_names: List[str],
) -> tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    instance_df = build_instance_dataframe(pred, logits, class_names)
    mosaic_df = aggregate_mosaic_votes(
        instance_df,
        num_classes=len(class_names),
        class_names=class_names,
    )
    mosaic_logits = torch.as_tensor(
        np.asarray(mosaic_df["logits"].tolist()),
        dtype=torch.float32,
    )
    mosaic_labels = torch.as_tensor(
        mosaic_df["label"].to_numpy(),
        dtype=torch.long,
    )
    return (
        instance_df,
        build_metric_summary(
            "instance",
            logits,
            pred["label"],
            class_names,
            "path",
        ),
        mosaic_df,
        build_metric_summary(
            "mosaic_vote",
            mosaic_logits,
            mosaic_labels,
            class_names,
            "slide_id",
        ),
    )


def predict_logits(
    model: nn.Module,
    embeddings: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], batch_size):
            end = min(start + batch_size, embeddings.shape[0])
            outputs.append(
                model(embeddings[start:end].to(device, non_blocking=True)).cpu()
            )
    return torch.vstack(outputs)


def fit_linear_probe(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    cfg: Dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame]:
    set_reproducible_seed(cfg["seed"])
    model = nn.Linear(train_embeddings.shape[1], num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["learning_rate"],
        momentum=0.9,
        weight_decay=cfg["weight_decay"],
    )
    generator = torch.Generator().manual_seed(cfg["seed"])
    batch_size = min(cfg["batch_size"], len(train_labels))

    history = []
    for epoch in tqdm(
        range(cfg["num_epochs"]),
        desc=f'{cfg.get("trial_name", "linear")} epochs',
        leave=False,
    ):
        model.train()
        total_loss = 0.0
        total_seen = 0
        indices = torch.randperm(len(train_labels), generator=generator)
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            batch_embeddings = train_embeddings[idx].to(device, non_blocking=True)
            batch_labels = train_labels[idx].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch_embeddings), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_labels.shape[0]
            total_seen += batch_labels.shape[0]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / total_seen,
                "learning_rate": cfg["learning_rate"],
                "num_epochs": cfg["num_epochs"],
                "seed": cfg["seed"],
            }
        )
    return model, pd.DataFrame(history)


def build_trial_cfgs(
    class_names: List[str],
    experiments_dir: str,
    linear_params: Dict[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    base_cfg = {
        "class_names": class_names,
        "device": device,
        "batch_size": linear_params["batch_size"],
        "weight_decay": linear_params["weight_decay"],
        "selection_metric": linear_params["selection_metric"],
        "seed": linear_params["seed"],
    }
    trial_cfgs = []
    for learning_rate, num_epochs in product(
        linear_params["learning_rates"],
        linear_params["num_epochs_list"],
    ):
        trial_name = (
            "lr"
            f"{f'{float(learning_rate):.8g}'.replace('-', 'm').replace('.', 'd')}"
            f"_ep{int(num_epochs):03d}"
        )
        trial_cfgs.append(
            {
                **base_cfg,
                "trial_name": trial_name,
                "trial_dir": opj(experiments_dir, trial_name),
                "learning_rate": float(learning_rate),
                "num_epochs": int(num_epochs),
            }
        )
    return trial_cfgs


def run_linear_eval(
    train_pred: Dict[str, Any],
    test_pred: Dict[str, Any],
    class_names: List[str],
    cfg: Dict[str, Any],
    split_names: tuple[str, str],
) -> tuple[
    nn.Module,
    pd.DataFrame,
    tuple[str, pd.DataFrame, Dict[str, Any]],
    tuple[str, pd.DataFrame, Dict[str, Any]],
]:
    model, history_df = fit_linear_probe(
        train_pred["embeddings"],
        train_pred["label"],
        len(class_names),
        cfg,
        cfg["device"],
    )
    instance_df, instance_metrics, mosaic_df, mosaic_metrics = evaluate_logits(
        test_pred,
        predict_logits(
            model, test_pred["embeddings"], cfg["batch_size"], cfg["device"]
        ),
        class_names,
    )
    return (
        model,
        history_df,
        (split_names[0], instance_df, instance_metrics),
        (split_names[1], mosaic_df, mosaic_metrics),
    )


def save_trial_outputs(
    trial_dir: str,
    outputs: List[tuple[str, pd.DataFrame, Dict[str, Any]]],
    split_class_names: Dict[str, List[str]],
    history_df: pd.DataFrame,
    binary_history_df: Optional[pd.DataFrame],
    trial_summary: Dict[str, Any],
    model: nn.Module,
    binary_model: Optional[nn.Module],
) -> None:
    os.makedirs(trial_dir, exist_ok=True)
    history_df.to_csv(opj(trial_dir, "training_history.csv"), index=False)
    if binary_history_df is not None:
        binary_history_df.to_csv(
            opj(trial_dir, "training_history_binary.csv"), index=False
        )
    for split_name, prediction_df, metrics in outputs:
        prediction_df.to_csv(
            opj(trial_dir, f"{split_name}_predictions.csv"), index=False
        )
        pd.DataFrame([metrics["metrics"]]).to_csv(
            opj(trial_dir, f"{split_name}_metrics.csv"),
            index=False,
        )
        plot_confusion(
            confusion=np.asarray(metrics["cm"], dtype=np.int64),
            out_file=opj(trial_dir, f"{split_name}_confusion.png"),
            class_names=split_class_names[split_name],
        )
        with open(
            opj(trial_dir, f"{split_name}_all_metrics.json"),
            "w",
            encoding="utf-8",
        ) as fd:
            json.dump(metrics, fd, indent=2)
    with open(opj(trial_dir, "summary.json"), "w", encoding="utf-8") as fd:
        json.dump(trial_summary, fd, indent=2)
    torch.save(
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        opj(trial_dir, "model.pt"),
    )
    if binary_model is not None:
        torch.save(
            {k: v.detach().cpu() for k, v in binary_model.state_dict().items()},
            opj(trial_dir, "model_binary.pt"),
        )


def run_linear_trial(
    cfg: Dict[str, Any],
    databank_pred: Dict[str, Any],
    test_pred: Dict[str, Any],
    binary_databank_pred: Optional[Dict[str, Any]],
    binary_test_pred: Optional[Dict[str, Any]],
    binary_class_names: Optional[List[str]],
) -> Dict[str, Any]:
    logging.info("Training %s", cfg["trial_name"])
    model, history_df, instance_out, mosaic_out = run_linear_eval(
        databank_pred,
        test_pred,
        cfg["class_names"],
        cfg,
        ("instance", "mosaic_vote"),
    )
    outputs = [instance_out, mosaic_out]
    split_class_names = {
        "instance": cfg["class_names"],
        "mosaic_vote": cfg["class_names"],
    }

    binary_model = None
    binary_history_df = None
    if binary_databank_pred is not None:
        binary_model, binary_history_df, binary_instance_out, binary_mosaic_out = (
            run_linear_eval(
                binary_databank_pred,
                binary_test_pred,
                binary_class_names,
                cfg,
                ("instance_binary", "mosaic_vote_binary"),
            )
        )
        outputs.extend([binary_instance_out, binary_mosaic_out])
        split_class_names["instance_binary"] = binary_class_names
        split_class_names["mosaic_vote_binary"] = binary_class_names

    trial_summary = {
        "trial_name": cfg["trial_name"],
        "trial_dir": cfg["trial_dir"],
        "learning_rate": cfg["learning_rate"],
        "num_epochs": cfg["num_epochs"],
        "seed": cfg["seed"],
        "train_num_samples": int(databank_pred["label"].shape[0]),
        "test_num_samples": int(test_pred["label"].shape[0]),
        **{split_name: metrics for split_name, _, metrics in outputs},
    }
    scope, metric = cfg["selection_metric"].split(".", maxsplit=1)
    trial_summary["selection_metric"] = cfg["selection_metric"]
    trial_summary["selection_value"] = float(trial_summary[scope]["metrics"][metric])

    save_trial_outputs(
        cfg["trial_dir"],
        outputs,
        split_class_names,
        history_df,
        binary_history_df,
        trial_summary,
        model,
        binary_model,
    )
    return trial_summary


def evaluate_run(
    name: str,
    databank_pred_path: str,
    test_pred_path: str,
    databank_gt_csv_path: Optional[str],
    test_gt_csv_path: Optional[str],
    out_dir: str,
    class_names: Optional[List[str]] = None,
    negative_class: Optional[str] = None,
    gt_label_column: str = "label",
    linear_params: Optional[Dict[str, Any]] = None,
    binary_sampling_seed: int = 0,
) -> None:
    linear_params = {
        "learning_rates": [1e-4, 5e-4, 1e-3],
        "num_epochs_list": [1],
        "batch_size": 8192,
        "weight_decay": 0.0,
        "selection_metric": "instance.mca",
        "seed": 0,
        **(linear_params or {}),
    }

    logging.info("Evaluating run with linear probes: %s", name)
    os.makedirs(out_dir, exist_ok=True)
    databank_pred = load_prediction(databank_pred_path)
    test_pred = load_prediction(test_pred_path)

    logging.info("Loading GT labels for %s", name)
    databank_gt_labels = (
        databank_pred["label"]
        if databank_gt_csv_path is None
        else load_labels_from_cell_instances(
            databank_pred["path"],
            databank_gt_csv_path,
            gt_label_column,
        )
    )
    test_gt_labels = (
        test_pred["label"]
        if test_gt_csv_path is None
        else load_labels_from_cell_instances(
            test_pred["path"],
            test_gt_csv_path,
            gt_label_column,
        )
    )

    databank_pred["label"], test_pred["label"], class_names = encode_prediction_labels(
        databank_gt_labels,
        test_gt_labels,
        class_names,
    )
    databank_pred["embeddings"] = normalize_embeddings(databank_pred["embeddings"])
    test_pred["embeddings"] = normalize_embeddings(test_pred["embeddings"])
    binary_databank_pred = None
    binary_test_pred = None
    binary_class_names = None
    if negative_class is not None:
        (
            binary_databank_pred,
            binary_test_pred,
            binary_class_names,
        ) = build_balanced_binary_predictions(
            databank_pred,
            test_pred,
            class_names.index(negative_class),
            negative_class,
            binary_sampling_seed,
        )

    selection_scope, selection_metric_name = linear_params["selection_metric"].split(
        ".",
        maxsplit=1,
    )
    selection_column = f"{selection_scope}_{selection_metric_name}"
    tie_breaker_acc_column = f"{selection_scope}_acc"
    experiments_dir = opj(out_dir, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)

    trial_summaries = [
        run_linear_trial(
            cfg,
            databank_pred,
            test_pred,
            binary_databank_pred,
            binary_test_pred,
            binary_class_names,
        )
        for cfg in tqdm(
            build_trial_cfgs(
                class_names,
                experiments_dir,
                linear_params,
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            ),
            desc="Linear trials",
            leave=False,
        )
    ]
    trial_metrics_df = pd.DataFrame(
        [
            {
                key: summary[key]
                for key in (
                    "trial_name",
                    "trial_dir",
                    "learning_rate",
                    "num_epochs",
                    "seed",
                    "train_num_samples",
                    "test_num_samples",
                )
            }
            | {
                f"{split_name}_{metric_name}": metric_value
                for split_name in (
                    "instance",
                    "mosaic_vote",
                    "instance_binary",
                    "mosaic_vote_binary",
                )
                if split_name in summary
                for metric_name, metric_value in summary[split_name]["metrics"].items()
            }
            for summary in trial_summaries
        ]
    )
    best_trial = trial_metrics_df.sort_values(
        [selection_column, tie_breaker_acc_column, "num_epochs", "learning_rate"],
        ascending=[False, False, True, True],
        ignore_index=True,
    ).iloc[0]

    best_summary = next(
        summary
        for summary in trial_summaries
        if summary["trial_name"] == str(best_trial["trial_name"])
    )
    for split_name in (
        "instance",
        "mosaic_vote",
        "instance_binary",
        "mosaic_vote_binary",
    ):
        if split_name not in best_summary:
            continue
        for suffix in (
            "_all_metrics.json",
            "_metrics.csv",
            "_predictions.csv",
            "_confusion.png",
            "_confusion.png.pdf",
        ):
            src = opj(best_summary["trial_dir"], f"{split_name}{suffix}")
            if os.path.exists(src):
                shutil.copy2(src, opj(out_dir, os.path.basename(src)))

    best_selection = {
        "selection_metric": linear_params["selection_metric"],
        "selection_column": selection_column,
        "best_hyperparams": {
            "learning_rate": float(best_trial["learning_rate"]),
            "num_epochs": int(best_trial["num_epochs"]),
        },
        "best_trial": {
            "trial_name": str(best_trial["trial_name"]),
            "trial_dir": str(best_trial["trial_dir"]),
            "seed": int(best_trial["seed"]),
            "selection_value": float(best_trial[selection_column]),
        },
    }
    with open(opj(out_dir, "best_selection.json"), "w", encoding="utf-8") as fd:
        json.dump(best_selection, fd, indent=2)

    summary = {
        "name": name,
        "method": "linear_probe",
        "databank_pred_path": databank_pred_path,
        "test_pred_path": test_pred_path,
        "class_names": class_names,
        "negative_class": negative_class,
        "linear_params": linear_params,
        "linear_selection_dataset": "test",
        "binary_sampling_seed": binary_sampling_seed,
        "selection": best_selection,
        "instance": best_summary["instance"],
        "mosaic_vote": best_summary["mosaic_vote"],
    }
    if "instance_binary" in best_summary:
        summary["instance_binary"] = best_summary["instance_binary"]
        summary["mosaic_vote_binary"] = best_summary["mosaic_vote_binary"]
    with open(opj(out_dir, "summary.json"), "w", encoding="utf-8") as fd:
        json.dump(summary, fd, indent=2)

    logging.info(
        "Best multiclass instance metrics for %s:\n%s",
        name,
        pd.DataFrame([best_summary["instance"]["metrics"]]),
    )
    logging.info(
        "Best multiclass mosaic metrics for %s:\n%s",
        name,
        pd.DataFrame([best_summary["mosaic_vote"]["metrics"]]),
    )
    if "instance_binary" in best_summary:
        logging.info(
            "Best binary instance metrics for %s:\n%s",
            name,
            pd.DataFrame([best_summary["instance_binary"]["metrics"]]),
        )
        logging.info(
            "Best binary mosaic metrics for %s:\n%s",
            name,
            pd.DataFrame([best_summary["mosaic_vote_binary"]["metrics"]]),
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    databank_gt_csv_path = "/nfs/turbo/umms-tocho/data/chengjia/silica_databank/instances_labels/srh7_1dot4m_.csv"
    test_gt_csv_path = "/nfs/turbo/umms-tocho/data/chengjia/silica_databank/instances_labels/srh7_test_.csv"

    class_names = [
        "hgg",
        "lgg",
        "mening",
        "metast",
        "normal",
        "pituita",
        "schwan",
    ]
    negative_class = "normal"
    linear_params = {
        "learning_rates": [1e-4, 1e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5],
        "num_epochs_list": [10],
        "batch_size": 128,
        "weight_decay": 0.0,
        "selection_metric": "instance.mca",
        "seed": 0,
    }
    run_sets = [
        {
            "exp_name": "b1a0cbe3_Apr07-21-09-04_sd1000_nomaskobw_lr13_tune0",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        },
        {
            "exp_name": "844ffd45_Apr06-12-07-47_sd1000_maskobw_lr43_tune1",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        },
        {
            "exp_name": "ca187b7c_Apr05-03-07-13_sd1000_nomaskobw_lr43_tune0",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        },
        # {
        #   "exp_name": "04e0bf39_Apr05-03-07-21_sd1000_dinov2_lr43_tune0",
        #   "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #   "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #    "exp_name": "4fb55301_Apr09-01-59-24_sd1000_nomaskobw_lr54_tune0",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #   "exp_name": "78d57cfc_Apr06-12-13-26_sd1000_dinov2_rmbg_lr43_tune0",
        #   "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #   "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #   "exp_name": "a2706135_dinov2",
        #   "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #   "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
    ]
    runs = build_runs_from_sets(
        exp_root="/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset2/",
        ckpt="training_124999",
        run_sets=run_sets,
        run_key_prefix="PERTURB",
    )

    for cfg in tqdm(runs, desc="Linear evaluation runs"):
        evaluate_run(
            databank_gt_csv_path=databank_gt_csv_path,
            test_gt_csv_path=test_gt_csv_path,
            out_dir=infer_results_dir_from_prediction_path(
                cfg["test_pred_path"],
                run_dir_prefix="linear",
            ),
            class_names=class_names,
            negative_class=negative_class,
            linear_params=linear_params,
            **cfg,
        )


if __name__ == "__main__":
    main()
