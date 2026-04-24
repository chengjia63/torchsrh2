import json
import logging
import os
from os.path import join as opj
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from ts2.utils.silica_sc_cls.eval_cell_inference_knn import (
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
from ts2.utils.silica_sc_eval.gmm_inference import (
    center_and_normalize_embeddings,
    load_db_mean,
    load_gmm,
)


def resolve_gmm_artifact_paths(gmm_path: str, k: int) -> Dict[str, str]:
    if not gmm_path:
        raise ValueError("Expected a non-empty gmm_path")

    model_path = opj(gmm_path, "models", f"gmm_g2m_m{k}.pkl")
    metrics_path = opj(gmm_path, "stats", "gmm_g2m_metrics.csv")
    db_mean_path = opj(gmm_path, "stats", "db_mean.pt")

    required_paths = {
        "artifact_dir": gmm_path,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "db_mean_path": db_mean_path,
    }
    for key, path in required_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing GMM {key}: {path}")
    return required_paths


def load_component_class_matrix(
    metrics_path: str,
    class_names: List[str],
    k: int,
) -> torch.Tensor:
    metrics = pd.read_csv(metrics_path)
    if "k" not in metrics.columns:
        raise KeyError(f"Expected `k` column in metrics CSV: {metrics_path}")
    if "class_proportions" not in metrics.columns:
        raise KeyError(
            f"Expected `class_proportions` column in metrics CSV: {metrics_path}"
        )

    metric_row = metrics.loc[metrics["k"] == k]
    if metric_row.empty:
        raise ValueError(f"No metrics row found for k={k} in {metrics_path}")

    raw_value = metric_row.iloc[0]["class_proportions"]
    class_proportions = (
        json.loads(raw_value) if isinstance(raw_value, str) else raw_value
    )
    if not isinstance(class_proportions, dict):
        raise TypeError(
            f"Expected class_proportions to deserialize to a dict for k={k}, got {type(class_proportions)}"
        )

    normalized_class_proportions = {
        int(str(component_idx).strip()): component_scores
        for component_idx, component_scores in class_proportions.items()
    }
    missing_components = sorted(set(range(k)) - set(normalized_class_proportions))
    if missing_components:
        raise ValueError(
            f"Missing class proportions for components {missing_components[:10]} in {metrics_path}"
        )

    matrix = np.zeros((k, len(class_names)), dtype=np.float32)
    for component_idx in range(k):
        component_scores = normalized_class_proportions[component_idx]
        if not isinstance(component_scores, dict):
            raise TypeError(
                f"Expected class scores for component {component_idx} to be a dict, got {type(component_scores)}"
            )
        missing_classes = sorted(set(class_names) - set(component_scores))
        if missing_classes:
            raise ValueError(
                f"Component {component_idx} in {metrics_path} is missing classes: {missing_classes}"
            )
        extra_classes = sorted(set(component_scores) - set(class_names))
        if extra_classes:
            raise ValueError(
                f"Component {component_idx} in {metrics_path} has unexpected classes: {extra_classes}"
            )
        matrix[component_idx] = np.asarray(
            [float(component_scores[class_name]) for class_name in class_names],
            dtype=np.float32,
        )
    return torch.from_numpy(matrix)


def compute_gmm_logits(
    test_pred: Dict[str, Any],
    gmm_model_path: str,
    db_mean_path: str,
    metrics_path: str,
    class_names: List[str],
    k: int,
) -> torch.Tensor:
    logging.info("Loading GMM model from %s", gmm_model_path)
    gmm = load_gmm(gmm_model_path)
    if gmm.n_components != k:
        raise ValueError(
            f"Expected GMM with k={k} components, got {gmm.n_components} from {gmm_model_path}"
        )

    db_mean = load_db_mean(db_mean_path)
    _, test_embs_norm = center_and_normalize_embeddings(
        db_mean=db_mean,
        inf_embs=test_pred["embeddings"],
    )
    responsibilities = torch.from_numpy(
        gmm.predict_proba(test_embs_norm.cpu().numpy())
    ).to(torch.float32)
    if responsibilities.shape != (len(test_pred["path"]), k):
        raise ValueError(
            "Unexpected responsibility matrix shape: "
            f"expected {(len(test_pred['path']), k)}, got {tuple(responsibilities.shape)}"
        )

    class_matrix = load_component_class_matrix(
        metrics_path=metrics_path,
        class_names=class_names,
        k=k,
    )
    logits = responsibilities @ class_matrix
    row_sums = logits.sum(dim=1)
    if not torch.all(row_sums > 0):
        raise ValueError(
            "Found non-positive per-cell GMM score sums after class aggregation"
        )
    return torch.nn.functional.normalize(logits, p=1, dim=1)


def aggregate_slide_pooling(
    instance_df: pd.DataFrame, num_classes: int, class_names: List[str]
) -> pd.DataFrame:
    rows = []
    grouped = instance_df.groupby("slide_id", sort=True)
    for slide_id, group in tqdm(
        grouped,
        total=instance_df["slide_id"].nunique(),
        desc="Slide pooling",
        leave=False,
    ):
        labels = group["label"].unique().tolist()
        if len(labels) != 1:
            raise ValueError(
                f"Found multiple ground-truth labels in slide {slide_id}: {labels}"
            )

        logits = torch.as_tensor(
            np.asarray(group["logits"].tolist()), dtype=torch.float32
        )
        pooled_logits = logits.mean(dim=0)
        if pooled_logits.numel() != num_classes:
            raise ValueError(
                f"Unexpected pooled logit length for slide {slide_id}: "
                f"expected {num_classes}, got {pooled_logits.numel()}"
            )

        label = int(labels[0])
        pred = int(pooled_logits.argmax().item())
        rows.append(
            {
                "slide_id": slide_id,
                "num_instances": len(group),
                "label": label,
                "pred": pred,
                "label_name": class_names[label],
                "pred_name": class_names[pred],
                "logits": pooled_logits.tolist(),
            }
        )

    return pd.DataFrame(rows).sort_values("slide_id", ignore_index=True)


def save_eval_outputs_no_save(
    out_dir: str,
    split_name: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    prediction_df: pd.DataFrame,
    id_col: str,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    logging.info("Computing metrics for %s", split_name)
    if len(class_names) == 2:
        metrics = get_binary_metrics(logits, labels)
        metric_type = "binary"
    else:
        metrics = get_all_metrics(logits, labels, num_classes=len(class_names))
        metric_type = "multiclass"
    logging.info("Computing confusion matrix for %s", split_name)
    preds = logits.argmax(dim=1)
    cm = confusion_matrix(
        labels.numpy(), preds.numpy(), labels=np.arange(len(class_names))
    )

    logging.info("Formatting metric table for %s", split_name)
    metrics_df = pd.DataFrame([metrics])
    prediction_df.to_csv(opj(out_dir, f"{split_name}_predictions.csv"), index=False)
    metrics_df.to_csv(opj(out_dir, f"{split_name}_metrics.csv"), index=False)
    plot_confusion(cm, opj(out_dir, f"{split_name}_confusion.png"), class_names)

    print(f"{split_name} {metric_type} metrics")
    print(metrics_df)
    all_metrics = {
        "split": split_name,
        "num_samples": int(labels.shape[0]),
        "id_column": id_col,
        "metrics": metrics,
        "cm": cm.tolist(),
    }
    print(all_metrics)
    return all_metrics


def evaluate_run(
    name: str,
    databank_pred_path: str,
    test_pred_path: str,
    gmm_path: str,
    databank_gt_csv_path: Optional[str],
    test_gt_csv_path: Optional[str],
    out_dir: str,
    class_names: Optional[List[str]] = None,
    gt_label_column: str = "label",
    gmm_k: int = 256,
) -> None:
    logging.info("Evaluating run with GMM: %s", name)
    databank_pred = load_prediction(databank_pred_path)
    test_pred = load_prediction(test_pred_path)

    logging.info("Loading GT labels for %s", name)
    if databank_gt_csv_path is None:
        logging.info(
            "No databank GT CSV provided for %s; using labels from prediction file",
            name,
        )
        databank_gt_labels = databank_pred["label"]
    else:
        databank_gt_labels = load_labels_from_cell_instances(
            databank_pred["path"],
            databank_gt_csv_path,
            gt_label_column,
        )

    if test_gt_csv_path is None:
        logging.info(
            "No test GT CSV provided for %s; using labels from prediction file",
            name,
        )
        test_gt_labels = test_pred["label"]
    else:
        test_gt_labels = load_labels_from_cell_instances(
            test_pred["path"],
            test_gt_csv_path,
            gt_label_column,
        )

    logging.info("Encoding labels for %s", name)
    _, test_pred["label"], class_names = encode_prediction_labels(
        databank_gt_labels,
        test_gt_labels,
        class_names,
    )
    artifact_paths = resolve_gmm_artifact_paths(gmm_path=gmm_path, k=gmm_k)

    logging.info("Running GMM inference for %s", name)
    logits = compute_gmm_logits(
        test_pred=test_pred,
        gmm_model_path=artifact_paths["model_path"],
        db_mean_path=artifact_paths["db_mean_path"],
        metrics_path=artifact_paths["metrics_path"],
        class_names=class_names,
        k=gmm_k,
    )

    logging.info("Building instance dataframe for %s", name)
    instance_df = build_instance_dataframe(test_pred, logits, class_names)
    logging.info("Pooling slide scores for %s", name)
    slide_df = aggregate_slide_pooling(
        instance_df, num_classes=len(class_names), class_names=class_names
    )

    instance_metrics = save_eval_outputs_no_save(
        out_dir=out_dir,
        split_name="instance_gmm",
        logits=logits,
        labels=test_pred["label"],
        class_names=class_names,
        prediction_df=instance_df,
        id_col="path",
    )

    slide_logits = torch.as_tensor(
        np.asarray(slide_df["logits"].tolist()), dtype=torch.float32
    )
    slide_labels = torch.as_tensor(slide_df["label"].to_numpy(), dtype=torch.long)
    slide_metrics = save_eval_outputs_no_save(
        out_dir=out_dir,
        split_name="slide_pool",
        logits=slide_logits,
        labels=slide_labels,
        class_names=class_names,
        prediction_df=slide_df,
        id_col="slide_id",
    )

    summary = {
        "name": name,
        "databank_pred_path": databank_pred_path,
        "test_pred_path": test_pred_path,
        "class_names": class_names,
        "gmm_path": gmm_path,
        "gmm_k": gmm_k,
        "gmm_artifacts": artifact_paths,
        "instance_gmm": instance_metrics,
        "slide_pool": slide_metrics,
    }
    with open(opj(out_dir, "summary.json"), "w", encoding="utf-8") as fd:
        json.dump(summary, fd, indent=2)

    logging.info(
        "Instance GMM metrics for %s:\n%s",
        name,
        pd.DataFrame([instance_metrics["metrics"]]),
    )
    logging.info(
        "Slide pooled GMM metrics for %s:\n%s",
        name,
        pd.DataFrame([slide_metrics["metrics"]]),
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

    run_sets = [
        {
            "exp_name": "3122d0c0_Mar20-19-19-03_sd1000_dev_dinov2_lr43_tune0",
            "gmm_path": "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/utils/silica_sc_eval/srh7v1sp1dot4m_3122d0c0",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB05_*",
            "gmm_k": 256,
        },
        {
            "exp_name": "bead0872_Mar22-23-45-20_sd1000_dev_nomaskobw_lr43_tune0",
            "gmm_path": "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/utils/silica_sc_eval/srh7v1sp1dot4m_bead0872",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB05_*",
            "gmm_k": 256,
        },
        {
            "exp_name": "1dfffb8f_Mar22-23-45-20_sd1000_dev_maskobw_lr43_tune1",
            "gmm_path": "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/utils/silica_sc_eval/srh7v1sp1dot4m_1dfffb8f",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB05_*",
            "gmm_k": 256,
        },
        {
            "exp_name": "3122d0c0_Mar20-19-19-03_sd1000_dev_dinov2_lr43_tune0",
            "gmm_path": "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/utils/silica_sc_eval/srh7v1sp1dot4m_3122d0c0",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB00_*",
            "gmm_k": 256,
        },
        {
            "exp_name": "bead0872_Mar22-23-45-20_sd1000_dev_nomaskobw_lr43_tune0",
            "gmm_path": "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/utils/silica_sc_eval/srh7v1sp1dot4m_bead0872",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB00_*",
            "gmm_k": 256,
        },
        {
            "exp_name": "1dfffb8f_Mar22-23-45-20_sd1000_dev_maskobw_lr43_tune1",
            "gmm_path": "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/utils/silica_sc_eval/srh7v1sp1dot4m_1dfffb8f",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB00_*",
            "gmm_k": 256,
        },
    ]

    exp_root = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset/"
    ckpt = "training_124999"
    run_key_prefix = "PERTURB"

    for run_set in tqdm(run_sets, desc="Evaluation run sets"):
        try:
            resolve_gmm_artifact_paths(
                gmm_path=run_set["gmm_path"],
                k=run_set["gmm_k"],
            )
        except (FileNotFoundError, ValueError) as exc:
            logging.warning(
                "Skipping %s because GMM artifacts are unavailable: %s",
                run_set["exp_name"],
                exc,
            )
            continue

        runs = build_runs_from_sets(
            exp_root=exp_root,
            ckpt=ckpt,
            run_sets=[run_set],
            run_key_prefix=run_key_prefix,
        )
        for cfg in runs:
            out_dir = infer_results_dir_from_prediction_path(
                cfg["test_pred_path"],
                run_dir_prefix="gmm",
            )
            evaluate_run(
                name=cfg["name"],
                databank_pred_path=cfg["databank_pred_path"],
                test_pred_path=cfg["test_pred_path"],
                gmm_path=run_set["gmm_path"],
                gmm_k=run_set["gmm_k"],
                databank_gt_csv_path=databank_gt_csv_path,
                test_gt_csv_path=test_gt_csv_path,
                out_dir=out_dir,
                class_names=class_names,
            )


if __name__ == "__main__":
    main()
