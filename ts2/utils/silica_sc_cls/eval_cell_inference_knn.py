import json
import logging
import os
from glob import glob
from os.path import join as opj
import re
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torchmetrics import AveragePrecision, Accuracy, AUROC, F1Score, Recall, Specificity
from tqdm import tqdm


def load_prediction(pred_path: str) -> Dict[str, Any]:
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file does not exist: {pred_path}")

    logging.info("Loading predictions from %s", pred_path)
    pred = torch.load(pred_path, map_location="cpu")
    if not isinstance(pred, dict):
        raise TypeError(f"Expected prediction file to contain a dict, got {type(pred)}")

    required_keys = {"embeddings", "label", "path"}
    missing = required_keys - set(pred.keys())
    if missing:
        raise KeyError(
            f"Prediction file {pred_path} is missing required keys: {sorted(missing)}"
        )

    pred["embeddings"] = ensure_tensor_2d(pred["embeddings"], "embeddings", pred_path)
    pred["label"] = ensure_label_tensor(pred["label"], pred_path)
    pred["path"] = ensure_path_list(pred["path"], pred_path)

    n = pred["embeddings"].shape[0]
    if len(pred["label"]) != n or len(pred["path"]) != n:
        raise ValueError(
            f"Inconsistent prediction lengths in {pred_path}: "
            f"embeddings={n}, labels={len(pred['label'])}, paths={len(pred['path'])}"
        )
    return pred


def ensure_tensor_2d(x: Any, key: str, pred_path: str) -> torch.Tensor:
    if isinstance(x, list):
        if not x:
            raise ValueError(f"{pred_path} has empty list for {key}")
        x = torch.as_tensor(np.asarray(x))
    elif not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if x.ndim != 2:
        raise ValueError(
            f"{pred_path} expected {key} to be rank-2, got shape {tuple(x.shape)}"
        )
    return x.to(torch.float32)


def ensure_label_tensor(x: Any, pred_path: str) -> List[Any]:
    if isinstance(x, torch.Tensor):
        if x.ndim != 1:
            raise ValueError(
                f"{pred_path} expected label to be rank-1, got shape {tuple(x.shape)}"
            )
        return x.cpu().tolist()

    x_np = np.asarray(x)
    if x_np.ndim != 1:
        raise ValueError(
            f"{pred_path} expected label to be rank-1, got shape {tuple(x_np.shape)}"
        )
    return x_np.tolist()


def encode_prediction_labels(
    databank_labels: List[Any],
    test_labels: List[Any],
    class_names: Optional[List[str]] = None,
) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    databank_labels = [str(x) for x in databank_labels]
    test_labels = [str(x) for x in test_labels]
    unique_labels = list(dict.fromkeys(databank_labels + test_labels))

    if class_names is not None:
        class_names = [str(x) for x in class_names]
        label_to_idx = {name: idx for idx, name in enumerate(class_names)}
        missing = sorted(set(unique_labels) - set(label_to_idx))
        if missing:
            raise ValueError(
                f"Found labels not present in class_names: {missing}. "
                f"class_names={class_names}"
            )
        return (
            torch.as_tensor(
                [label_to_idx[x] for x in databank_labels], dtype=torch.long
            ),
            torch.as_tensor([label_to_idx[x] for x in test_labels], dtype=torch.long),
            class_names,
        )

    all_numeric = all(label.lstrip("+-").isdigit() for label in unique_labels)
    if all_numeric:
        databank_tensor = torch.as_tensor(
            [int(x) for x in databank_labels], dtype=torch.long
        )
        test_tensor = torch.as_tensor([int(x) for x in test_labels], dtype=torch.long)
        num_classes = int(max(databank_tensor.max(), test_tensor.max()).item()) + 1
        inferred_class_names = [str(i) for i in range(num_classes)]
        return databank_tensor, test_tensor, inferred_class_names

    inferred_class_names = unique_labels
    label_to_idx = {name: idx for idx, name in enumerate(inferred_class_names)}
    return (
        torch.as_tensor([label_to_idx[x] for x in databank_labels], dtype=torch.long),
        torch.as_tensor([label_to_idx[x] for x in test_labels], dtype=torch.long),
        inferred_class_names,
    )


def ensure_path_list(x: Any, pred_path: str) -> List[str]:
    if isinstance(x, list):
        out = [str(v) for v in x]
    else:
        raise TypeError(f"{pred_path} expected path to be a list, got {type(x)}")
    return out


def parse_tuple_string(s: str) -> tuple[int, int]:
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        raise ValueError(f"Input must be a string representation of a tuple, got {s!r}")
    parts = [part.strip() for part in s[1:-1].split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected a 2-tuple string, got {s!r}")
    return int(parts[0]), int(parts[1])


def encode_cell_path(patch: str, proposal: Any) -> str:
    if isinstance(proposal, str):
        row, col = parse_tuple_string(proposal)
    elif isinstance(proposal, (tuple, list)) and len(proposal) == 2:
        row, col = int(proposal[0]), int(proposal[1])
    else:
        raise ValueError(
            f"Could not parse proposal coordinates from value: {proposal!r}"
        )
    return f"{patch}#{row}_{col}"


def load_labels_from_cell_instances(
    pred_paths: List[str],
    cell_instances_csv_path: str,
    label_column: str,
) -> List[Any]:
    if not os.path.exists(cell_instances_csv_path):
        raise FileNotFoundError(f"GT CSV does not exist: {cell_instances_csv_path}")

    logging.info("Loading GT labels from %s", cell_instances_csv_path)
    label_source = pd.read_csv(cell_instances_csv_path, dtype=str)
    required_cols = {"patch", "proposal", label_column}
    missing = required_cols - set(label_source.columns)
    if missing:
        raise KeyError(
            f"GT CSV {cell_instances_csv_path} is missing required columns: {sorted(missing)}"
        )

    label_source = label_source.copy()
    label_source["path"] = [
        encode_cell_path(patch, proposal)
        for patch, proposal in tqdm(
            zip(label_source["patch"], label_source["proposal"]),
            total=len(label_source),
            desc=f"Build GT paths: {os.path.basename(cell_instances_csv_path)}",
            leave=False,
        )
    ]
    label_source = label_source.rename(columns={label_column: "gt_label"})[
        ["path", "gt_label"]
    ]

    labels_per_path = label_source.groupby("path")["gt_label"].nunique()
    ambiguous_paths = labels_per_path[labels_per_path > 1]
    if not ambiguous_paths.empty:
        raise ValueError(
            "Cell instances CSV produced paths with conflicting labels: "
            + ", ".join(ambiguous_paths.index[:10])
        )

    label_source = label_source.drop_duplicates(subset=["path", "gt_label"])
    pred_df = pd.DataFrame({"path": pred_paths})
    merged = pred_df.merge(label_source, on="path", how="left", validate="many_to_one")
    missing_label_count = int(merged["gt_label"].isna().sum())
    if missing_label_count != 0:
        raise ValueError(
            f"Could not match {missing_label_count} prediction rows to labels in {cell_instances_csv_path}"
        )
    return merged["gt_label"].tolist()


def knn_predict(
    feature: torch.Tensor,
    feature_bank: torch.Tensor,
    feature_labels: torch.Tensor,
    classes: int,
    knn_k: int,
    knn_t: float,
) -> torch.Tensor:
    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(
        feature_labels.expand(feature.shape[0], -1),
        dim=-1,
        index=sim_indices,
    )
    sim_weight = (sim_weight / knn_t).exp()
    one_hot_label = torch.zeros(
        feature.shape[0] * knn_k,
        classes,
        device=sim_labels.device,
    )
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    return pred_scores


def compute_knn_logits(
    databank_pred: Dict[str, Any],
    test_pred: Dict[str, Any],
    num_classes: int,
    knn_k: int,
    knn_t: float,
    batch_size: int,
) -> torch.Tensor:
    if databank_pred["embeddings"].shape[0] < knn_k:
        raise ValueError(
            f"kNN requires at least k databank samples, got k={knn_k} and "
            f"{databank_pred['embeddings'].shape[0]} databank embeddings"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_embs = torch.nn.functional.normalize(
        databank_pred["embeddings"], p=2, dim=1
    ).T.to(device)
    test_embs = torch.nn.functional.normalize(test_pred["embeddings"], p=2, dim=1)
    train_labels = databank_pred["label"].to(device)

    all_scores = []
    total_batches = (test_embs.shape[0] + batch_size - 1) // batch_size
    with torch.no_grad():
        for start in tqdm(
            range(0, test_embs.shape[0], batch_size),
            total=total_batches,
            desc="kNN batches",
            leave=False,
        ):
            end = min(start + batch_size, test_embs.shape[0])
            batch = test_embs[start:end].to(device)
            scores = knn_predict(
                feature=batch,
                feature_bank=train_embs,
                feature_labels=train_labels,
                classes=num_classes,
                knn_k=knn_k,
                knn_t=knn_t,
            )
            all_scores.append(torch.nn.functional.normalize(scores, p=1, dim=1).cpu())
    return torch.vstack(all_scores)


def subset_prediction(pred: Dict[str, Any], indices: torch.Tensor) -> Dict[str, Any]:
    if indices.ndim != 1:
        raise ValueError(f"indices must be rank-1, got shape {tuple(indices.shape)}")
    indices = indices.to(torch.long).cpu()
    return {
        "embeddings": pred["embeddings"][indices],
        "label": pred["label"][indices],
        "path": [pred["path"][i] for i in indices.tolist()],
    }


def build_balanced_binary_predictions(
    databank_pred: Dict[str, Any],
    test_pred: Dict[str, Any],
    negative_class_idx: int,
    negative_class_name: str,
    seed: int = 0,
) -> tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    databank_labels = databank_pred["label"]
    test_labels = test_pred["label"]

    databank_neg_idx = torch.nonzero(databank_labels == negative_class_idx).flatten()
    databank_pos_idx = torch.nonzero(databank_labels != negative_class_idx).flatten()
    if databank_neg_idx.numel() == 0:
        raise ValueError(
            f"Cannot build binary databank for {negative_class_name}: no negative-class samples found"
        )
    if databank_pos_idx.numel() == 0:
        raise ValueError(
            f"Cannot build binary databank for {negative_class_name}: no positive-class samples found"
        )
    if databank_pos_idx.numel() < databank_neg_idx.numel():
        raise ValueError(
            f"Cannot balance binary databank for {negative_class_name}: "
            f"positives={databank_pos_idx.numel()} < negatives={databank_neg_idx.numel()}"
        )

    rng = np.random.default_rng(seed)
    sampled_pos = rng.choice(
        databank_pos_idx.numpy(), size=databank_neg_idx.numel(), replace=False
    )
    selected_idx = torch.as_tensor(
        np.sort(np.concatenate([databank_neg_idx.numpy(), sampled_pos])),
        dtype=torch.long,
    )

    binary_databank_pred = subset_prediction(databank_pred, selected_idx)
    binary_test_pred = subset_prediction(
        test_pred, torch.arange(test_labels.shape[0], dtype=torch.long)
    )

    binary_databank_pred["label"] = (
        binary_databank_pred["label"] != negative_class_idx
    ).to(torch.long)
    binary_test_pred["label"] = (binary_test_pred["label"] != negative_class_idx).to(
        torch.long
    )
    binary_class_names = [negative_class_name, f"not_{negative_class_name}"]

    logging.info(
        "Built balanced binary databank for %s with %d negative and %d positive samples (seed=%d)",
        negative_class_name,
        databank_neg_idx.numel(),
        databank_neg_idx.numel(),
        seed,
    )
    return binary_databank_pred, binary_test_pred, binary_class_names


def infer_slide_id(path: str) -> str:
    base = path.split("@")[0].split("#")[0]
    parts = base.split("-")
    if len(parts) < 2:
        raise ValueError(f"Could not infer slide/mosaic id from path: {path}")
    return "-".join(parts[:2])


def build_instance_dataframe(
    test_pred: Dict[str, Any], logits: torch.Tensor, class_names: List[str]
) -> pd.DataFrame:
    pred_labels = logits.argmax(dim=1)
    paths = test_pred["path"]
    slide_ids = [
        infer_slide_id(p) for p in tqdm(paths, desc="Infer slide ids", leave=False)
    ]
    label_list = test_pred["label"].tolist()
    pred_list = pred_labels.tolist()
    return pd.DataFrame(
        {
            "path": paths,
            "slide_id": slide_ids,
            "label": label_list,
            "pred": pred_list,
            "label_name": [
                class_names[i]
                for i in tqdm(label_list, desc="Label names", leave=False)
            ],
            "pred_name": [
                class_names[i] for i in tqdm(pred_list, desc="Pred names", leave=False)
            ],
            "logits": logits.tolist(),
        }
    )


def aggregate_mosaic_votes(
    instance_df: pd.DataFrame, num_classes: int, class_names: List[str]
) -> pd.DataFrame:
    rows = []
    grouped = instance_df.groupby("slide_id", sort=True)
    import pdb

    # pdb.set_trace()
    for slide_id, group in tqdm(
        grouped,
        total=instance_df["slide_id"].nunique(),
        desc="Mosaic voting",
        leave=False,
    ):
        labels = group["label"].unique().tolist()
        if len(labels) != 1:
            raise ValueError(
                f"Found multiple ground-truth labels in mosaic {slide_id}: {labels}"
            )

        logits = torch.as_tensor(
            np.asarray(group["logits"].tolist()), dtype=torch.float32
        )
        vote_logits = logits.sum(dim=0)
        pred = int(vote_logits.argmax().item())
        label = int(labels[0])
        if vote_logits.numel() != num_classes:
            raise ValueError(
                f"Unexpected vote logit length for mosaic {slide_id}: "
                f"expected {num_classes}, got {vote_logits.numel()}"
            )

        rows.append(
            {
                "slide_id": slide_id,
                "num_instances": len(group),
                "label": label,
                "pred": pred,
                "label_name": class_names[label],
                "pred_name": class_names[pred],
                "logits": vote_logits.tolist(),
            }
        )

    return pd.DataFrame(rows).sort_values("slide_id", ignore_index=True)


def get_all_metrics(
    logits: torch.Tensor, label: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    if logits.ndim != 2 or label.ndim != 1:
        raise ValueError(
            f"Expected logits [N, C] and labels [N], got {tuple(logits.shape)} and {tuple(label.shape)}"
        )
    if logits.shape[0] != label.shape[0]:
        raise ValueError(
            f"Mismatched logits/label lengths: {logits.shape[0]} vs {label.shape[0]}"
        )
    if logits.shape[1] != num_classes:
        raise ValueError(
            f"Expected {num_classes} classes in logits, got {logits.shape[1]}"
        )

    acc = Accuracy(task="multiclass", num_classes=num_classes, top_k=1, average="micro")
    mca = Accuracy(task="multiclass", num_classes=num_classes, top_k=1, average="macro")
    auprc = AveragePrecision(
        task="multiclass", num_classes=num_classes, average="macro"
    )
    auroc = AUROC(task="multiclass", num_classes=num_classes, average="macro")
    sen = Recall(task="multiclass", num_classes=num_classes, top_k=1, average="micro")
    spec = Specificity(
        task="multiclass", num_classes=num_classes, top_k=1, average="micro"
    )
    f1 = F1Score(task="multiclass", num_classes=num_classes, top_k=1, average="macro")

    metrics = {
        "acc": float(acc(logits, label).item()),
        "t2": float(topk_accuracy_or_one(logits, label, num_classes, 2)),
        "t3": float(topk_accuracy_or_one(logits, label, num_classes, 3)),
        "mca": float(mca(logits, label).item()),
        "map": float(auprc(logits, label).item()),
        "auroc": float(auroc(logits, label).item()),
        "sen": float(sen(logits, label).item()),
        "sepc": float(spec(logits, label).item()),
        "f1": float(f1(logits, label).item()),
    }
    return metrics


def collapse_to_negative_vs_all(
    logits: torch.Tensor,
    labels: torch.Tensor,
    negative_class_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or labels.ndim != 1:
        raise ValueError(
            f"Expected logits [N, C] and labels [N], got {tuple(logits.shape)} and {tuple(labels.shape)}"
        )
    if not (0 <= negative_class_idx < logits.shape[1]):
        raise ValueError(
            f"negative_class_idx={negative_class_idx} is out of bounds for logits with {logits.shape[1]} classes"
        )
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Mismatched logits/label lengths: {logits.shape[0]} vs {labels.shape[0]}"
        )

    binary_logits = torch.stack(
        [
            logits[:, negative_class_idx],
            logits.sum(dim=1) - logits[:, negative_class_idx],
        ],
        dim=1,
    )
    binary_labels = (labels != negative_class_idx).to(torch.long)
    return binary_logits, binary_labels


def get_binary_metrics(logits: torch.Tensor, label: torch.Tensor) -> Dict[str, float]:
    if logits.ndim != 2 or label.ndim != 1:
        raise ValueError(
            f"Expected logits [N, 2] and labels [N], got {tuple(logits.shape)} and {tuple(label.shape)}"
        )
    if logits.shape[0] != label.shape[0]:
        raise ValueError(
            f"Mismatched logits/label lengths: {logits.shape[0]} vs {label.shape[0]}"
        )
    if logits.shape[1] != 2:
        raise ValueError(
            f"Expected binary logits with 2 columns, got {logits.shape[1]}"
        )

    probs = torch.nn.functional.normalize(logits, p=1, dim=1)
    pos_scores = probs[:, 1]
    acc = Accuracy(task="binary", average="micro")
    mca = Accuracy(task="multiclass", num_classes=2, average="macro")
    auprc = AveragePrecision(task="binary")
    auroc = AUROC(task="binary")
    sen = Recall(task="binary")
    spec = Specificity(task="binary")
    f1 = F1Score(task="binary")

    return {
        "acc": float(acc(pos_scores, label).item()),
        "t2": 1.0,
        "t3": 1.0,
        "mca": float(mca(probs, label).item()),
        "map": float(auprc(pos_scores, label).item()),
        "auroc": float(auroc(pos_scores, label).item()),
        "sen": float(sen(pos_scores, label).item()),
        "sepc": float(spec(pos_scores, label).item()),
        "f1": float(f1(pos_scores, label).item()),
    }


def resolve_negative_class_idx(
    negative_class: Optional[str], class_names: List[str]
) -> Optional[int]:
    if negative_class is None:
        return None

    if negative_class not in class_names:
        raise ValueError(
            f"negative_class={negative_class!r} not found in class_names={class_names}"
        )
    return class_names.index(negative_class)


def topk_accuracy_or_one(
    logits: torch.Tensor, label: torch.Tensor, num_classes: int, top_k: int
) -> float:
    if num_classes < top_k:
        return 1.0
    metric = Accuracy(
        task="multiclass", num_classes=num_classes, top_k=top_k, average="micro"
    )
    return metric(logits, label).item()


def plot_confusion(
    confusion: np.ndarray, out_file: str, class_names: List[str]
) -> None:
    nc = confusion.shape[0]
    fig, ax = plt.subplots(1, 1)

    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_normalized = np.divide(
        confusion,
        row_sums,
        out=np.zeros_like(confusion, dtype=float),
        where=row_sums != 0,
    )

    im = ax.imshow(confusion_normalized, cmap=plt.get_cmap("Blues"))
    ax.set_xticks(np.arange(nc))
    ax.set_yticks(np.arange(nc))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("none")
    im.set_clim(0, 1)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thres = np.max(confusion_normalized) / 2 if confusion_normalized.size else 0.5
    fontsize = (
        8
        if np.max(confusion) > 999 or ((nc > 7) and (np.max(confusion) > 99))
        else "small"
    )
    for i in range(nc):
        for j in range(nc):
            if confusion[i, j] == 0:
                continue
            color = "k" if confusion_normalized[i, j] < thres else "w"
            ax.text(
                j,
                i,
                confusion[i, j],
                ha="center",
                va="center",
                color=color,
                fontdict={"size": fontsize},
            )

    fig.tight_layout()
    plt.savefig(out_file)
    plt.savefig(out_file + ".pdf")
    plt.close(fig)


def save_eval_outputs(
    out_dir: str,
    split_name: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    prediction_df: pd.DataFrame,
    id_col: str,
    # negative_class: Optional[str] = None,
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

    logging.info("Writing prediction tables for %s", split_name)
    # prediction_df.to_pickle(opj(out_dir, f"{split_name}_predictions.pkl"))
    prediction_df.to_csv(opj(out_dir, f"{split_name}_predictions.csv"), index=False)

    logging.info("Formatting metric table for %s", split_name)
    metrics_df = pd.DataFrame([metrics])
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

    # if negative_class is not None and len(class_names) > 2:
    #    negative_class_idx = resolve_negative_class_idx(negative_class, class_names)
    #    binary_logits, binary_labels = collapse_to_negative_vs_all(
    #        logits=logits,
    #        labels=labels,
    #        negative_class_idx=negative_class_idx,
    #    )
    #    binary_preds = binary_logits.argmax(dim=1)
    #    binary_cm = confusion_matrix(
    #        binary_labels.numpy(), binary_preds.numpy(), labels=np.arange(2)
    #    )
    #    all_metrics["negative_vs_all"] = {
    #        "negative_class": negative_class,
    #        "class_names": [negative_class, f"not_{negative_class}"],
    #        "metrics": get_binary_metrics(binary_logits, binary_labels),
    #        "cm": binary_cm.tolist(),
    #    }
    #    print(f"{split_name} negative-vs-all metrics ({negative_class} vs rest)")
    #    print(pd.DataFrame([all_metrics["negative_vs_all"]["metrics"]]))

    with open(
        opj(out_dir, f"{split_name}_all_metrics.json"), "w", encoding="utf-8"
    ) as fd:
        json.dump(all_metrics, fd, indent=2)
    print(all_metrics)
    return all_metrics


def infer_results_dir_from_prediction_path(
    pred_path: str,
    run_dir_prefix: str = "run",
) -> str:
    pred_dir = os.path.dirname(pred_path)
    if os.path.basename(pred_dir) != "predictions":
        raise ValueError(
            f"Expected prediction path under a predictions directory, got {pred_path}"
        )
    if not re.fullmatch(r"[A-Za-z]+", run_dir_prefix):
        raise ValueError(
            f"Expected run_dir_prefix to contain only letters, got {run_dir_prefix!r}"
        )
    eval_root = os.path.dirname(pred_dir)
    results_root = opj(eval_root, "results")
    existing_run_ids = []
    if os.path.isdir(results_root):
        for entry in os.listdir(results_root):
            entry_path = opj(results_root, entry)
            if not os.path.isdir(entry_path):
                continue
            match = re.fullmatch(rf"{re.escape(run_dir_prefix)}(\d{{4}})", entry)
            if match is not None:
                existing_run_ids.append(int(match.group(1)))

    next_run_id = 0 if not existing_run_ids else max(existing_run_ids) + 1
    return opj(results_root, f"{run_dir_prefix}{next_run_id:04d}")


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
    knn_k: int = 20,
    knn_t: float = 0.07,
    knn_batch_size: int = 8192,
    binary_sampling_seed: int = 0,
) -> None:
    logging.info("Evaluating run: %s", name)
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
    databank_pred["label"], test_pred["label"], class_names = encode_prediction_labels(
        databank_gt_labels,
        test_gt_labels,
        class_names,
    )
    num_classes = len(class_names)
    negative_class_idx = resolve_negative_class_idx(negative_class, class_names)

    logging.info("Running kNN for %s", name)
    logits = compute_knn_logits(
        databank_pred=databank_pred,
        test_pred=test_pred,
        num_classes=num_classes,
        knn_k=knn_k,
        knn_t=knn_t,
        batch_size=knn_batch_size,
    )

    logging.info("Building instance dataframe for %s", name)
    instance_df = build_instance_dataframe(test_pred, logits, class_names)
    logging.info("Aggregating mosaic votes for %s", name)
    mosaic_df = aggregate_mosaic_votes(
        instance_df, num_classes=num_classes, class_names=class_names
    )

    instance_metrics = save_eval_outputs(
        out_dir=out_dir,
        split_name="instance_knn",
        logits=logits,
        labels=test_pred["label"],
        class_names=class_names,
        prediction_df=instance_df,
        id_col="path",
        # negative_class=negative_class,
    )

    mosaic_logits = torch.as_tensor(
        np.asarray(mosaic_df["logits"].tolist()), dtype=torch.float32
    )
    mosaic_labels = torch.as_tensor(mosaic_df["label"].to_numpy(), dtype=torch.long)
    mosaic_metrics = save_eval_outputs(
        out_dir=out_dir,
        split_name="mosaic_vote",
        logits=mosaic_logits,
        labels=mosaic_labels,
        class_names=class_names,
        prediction_df=mosaic_df,
        id_col="slide_id",
        # negative_class=negative_class,
    )

    binary_instance_metrics = None
    binary_mosaic_metrics = None
    if negative_class_idx is not None:
        logging.info("Running balanced binary kNN for %s", name)
        binary_databank_pred, binary_test_pred, binary_class_names = (
            build_balanced_binary_predictions(
                databank_pred=databank_pred,
                test_pred=test_pred,
                negative_class_idx=negative_class_idx,
                negative_class_name=negative_class,
                seed=binary_sampling_seed,
            )
        )
        binary_logits = compute_knn_logits(
            databank_pred=binary_databank_pred,
            test_pred=binary_test_pred,
            num_classes=2,
            knn_k=knn_k,
            knn_t=knn_t,
            batch_size=knn_batch_size,
        )
        binary_instance_df = build_instance_dataframe(
            binary_test_pred, binary_logits, binary_class_names
        )
        binary_mosaic_df = aggregate_mosaic_votes(
            binary_instance_df, num_classes=2, class_names=binary_class_names
        )
        binary_instance_metrics = save_eval_outputs(
            out_dir=out_dir,
            split_name="instance_knn_binary",
            logits=binary_logits,
            labels=binary_test_pred["label"],
            class_names=binary_class_names,
            prediction_df=binary_instance_df,
            id_col="path",
        )
        binary_mosaic_logits = torch.as_tensor(
            np.asarray(binary_mosaic_df["logits"].tolist()), dtype=torch.float32
        )
        binary_mosaic_labels = torch.as_tensor(
            binary_mosaic_df["label"].to_numpy(), dtype=torch.long
        )
        binary_mosaic_metrics = save_eval_outputs(
            out_dir=out_dir,
            split_name="mosaic_vote_binary",
            logits=binary_mosaic_logits,
            labels=binary_mosaic_labels,
            class_names=binary_class_names,
            prediction_df=binary_mosaic_df,
            id_col="slide_id",
        )

    summary = {
        "name": name,
        "databank_pred_path": databank_pred_path,
        "test_pred_path": test_pred_path,
        "class_names": class_names,
        "negative_class": negative_class,
        "knn_k": knn_k,
        "knn_t": knn_t,
        "knn_batch_size": knn_batch_size,
        "binary_sampling_seed": binary_sampling_seed,
        "instance_knn": instance_metrics,
        "mosaic_vote": mosaic_metrics,
    }
    if binary_instance_metrics is not None:
        summary["instance_knn_binary"] = binary_instance_metrics
    if binary_mosaic_metrics is not None:
        summary["mosaic_vote_binary"] = binary_mosaic_metrics
    with open(opj(out_dir, "summary.json"), "w", encoding="utf-8") as fd:
        json.dump(summary, fd, indent=2)

    logging.info(
        "Instance metrics for %s:\n%s",
        name,
        pd.DataFrame([instance_metrics["metrics"]]),
    )
    logging.info(
        "Mosaic metrics for %s:\n%s",
        name,
        pd.DataFrame([mosaic_metrics["metrics"]]),
    )
    # if "negative_vs_all" in instance_metrics:
    #    logging.info(
    #        "Instance negative-vs-all metrics for %s (%s vs rest):\n%s",
    #        name,
    #        instance_metrics["negative_vs_all"]["negative_class"],
    #        pd.DataFrame([instance_metrics["negative_vs_all"]["metrics"]]),
    #    )
    # if "negative_vs_all" in mosaic_metrics:
    #    logging.info(
    #        "Mosaic negative-vs-all metrics for %s (%s vs rest):\n%s",
    #        name,
    #        mosaic_metrics["negative_vs_all"]["negative_class"],
    #        pd.DataFrame([mosaic_metrics["negative_vs_all"]["metrics"]]),
    #    )
    if binary_instance_metrics is not None:
        logging.info(
            "Instance binary kNN metrics for %s:\n%s",
            name,
            pd.DataFrame([binary_instance_metrics["metrics"]]),
        )
    if binary_mosaic_metrics is not None:
        logging.info(
            "Mosaic binary kNN metrics for %s:\n%s",
            name,
            pd.DataFrame([binary_mosaic_metrics["metrics"]]),
        )


def find_matching_prediction_paths(
    pattern: str,
) -> List[str]:
    if not pattern:
        raise ValueError("Expected a non-empty glob pattern")

    matches = sorted(set(glob(pattern)))
    if not matches:
        raise ValueError(f"No prediction paths matched glob pattern: {pattern}")

    for path in matches:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Matched prediction path is not a file: {path}")

    return matches


def extract_perturb_key(path: str, prefix: str) -> str:
    if not prefix:
        raise ValueError("Expected a non-empty run key prefix")

    key_match = re.search(rf"({re.escape(prefix)}\d+)_", path)
    if key_match is None:
        raise ValueError(
            f"Could not extract run key with prefix {prefix!r} from path: {path}"
        )
    return key_match.group(1)


def build_runs_from_sets(
    exp_root: str,
    ckpt: str,
    run_sets: List[Dict[str, str]],
    run_key_prefix: str,
) -> List[Dict[str, str]]:
    if not exp_root:
        raise ValueError("Expected a non-empty exp_root")
    if not ckpt:
        raise ValueError("Expected a non-empty ckpt")

    runs: List[Dict[str, str]] = []
    for run_set in run_sets:
        required_keys = {
            "exp_name",
            "databank_pred_glob",
            "test_pred_glob",
        }
        missing = required_keys - set(run_set)
        if missing:
            raise KeyError(f"Run set is missing required keys: {sorted(missing)}")

        exp_name = run_set["exp_name"]
        base_eval_dir = opj(exp_root, exp_name, "models", "eval", ckpt)
        databank_glob = opj(
            base_eval_dir,
            run_set["databank_pred_glob"],
            "predictions",
            "pred.pt",
        )
        test_glob = opj(
            base_eval_dir,
            run_set["test_pred_glob"],
            "predictions",
            "pred.pt",
        )

        databank_matches = find_matching_prediction_paths(databank_glob)
        if len(databank_matches) != 1:
            raise ValueError(
                f"Expected exactly one databank prediction for run set {exp_name}, "
                f"found {len(databank_matches)}: {databank_matches}"
            )
        test_matches = find_matching_prediction_paths(test_glob)
        databank_pred_path = databank_matches[0]
        run_set_runs: List[Dict[str, str]] = []
        for test_pred_path in test_matches:
            perturb_key = extract_perturb_key(
                test_pred_path,
                prefix=run_key_prefix,
            )
            run_set_runs.append(
                {
                    "perturb_key": perturb_key,
                    "name": f"{exp_name}_{perturb_key}",
                    "databank_pred_path": databank_pred_path,
                    "test_pred_path": test_pred_path,
                }
            )
        perturb_key_to_test_pred_path = {
            run["perturb_key"]: run["test_pred_path"] for run in run_set_runs
        }
        if len(perturb_key_to_test_pred_path) != len(run_set_runs):
            raise ValueError(
                f"Expected unique perturb keys for run set {exp_name}, found duplicates "
                f"in test_pred_path values: {[run['test_pred_path'] for run in run_set_runs]}"
            )
        run_set_runs = sorted(
            run_set_runs,
            key=lambda run: (run["perturb_key"], run["test_pred_path"]),
        )
        runs.extend(
            {
                "name": run["name"],
                "databank_pred_path": run["databank_pred_path"],
                "test_pred_path": run["test_pred_path"],
            }
            for run in run_set_runs
        )

    return runs


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
    run_dir_prefix = "debug"
    run_sets = [
        # {
        #    "exp_name": "04e0bf39_Apr05-03-07-21_sd1000_dinov2_lr43_tune0",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #    "exp_name": "ca187b7c_Apr05-03-07-13_sd1000_nomaskobw_lr43_tune0",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #    "exp_name": "a2706135_dinov2",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #    "exp_name": "78d57cfc_Apr06-12-13-26_sd1000_dinov2_rmbg_lr43_tune0",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #    "exp_name": "844ffd45_Apr06-12-07-47_sd1000_maskobw_lr43_tune1",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        {
            "exp_name": "b1a0cbe3_Apr07-21-09-04_sd1000_nomaskobw_lr13_tune0",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        }
        # {
        #    "exp_name": "3122d0c0_Mar20-19-19-03_sd1000_dev_dinov2_lr43_tune0",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #    "exp_name": "bead0872_Mar22-23-45-20_sd1000_dev_nomaskobw_lr43_tune0",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #    "exp_name": "1dfffb8f_Mar22-23-45-20_sd1000_dev_maskobw_lr43_tune1",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #    "exp_name": "1526bfe8_Mar24-15-02-22_sd1000_dev_nomaskobw_lr13_tune0",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
        # {
        #    "exp_name": "8751a922_Mar24-15-02-22_sd1000_dev_maskobw_lr13_tune1",
        #    "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
        #    "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        # },
    ]
    runs = build_runs_from_sets(
        exp_root="/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset2/",
        ckpt="training_124999",
        run_sets=run_sets,
        run_key_prefix="PERTURB",
    )

    for cfg in tqdm(runs, desc="Evaluation runs"):
        out_dir = infer_results_dir_from_prediction_path(
            cfg["test_pred_path"], run_dir_prefix=run_dir_prefix
        )
        evaluate_run(
            databank_gt_csv_path=databank_gt_csv_path,
            test_gt_csv_path=test_gt_csv_path,
            out_dir=out_dir,
            class_names=class_names,
            negative_class=negative_class,
            **cfg,
        )


if __name__ == "__main__":
    main()

    #    {
    #        "name": "bead0872_Mar22-23-45-20_sd1000_dev_nomaskobw_lr43_tune0",
    #        "databank_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset/bead0872_Mar22-23-45-20_sd1000_dev_nomaskobw_lr43_tune0/models/eval/training_124999/c5408b55_Mar24-10-50-44_sd1000_INF_srh7v1sp1dot4m_dev/predictions/pred.pt",
    #        "test_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset/bead0872_Mar22-23-45-20_sd1000_dev_nomaskobw_lr43_tune0/models/eval/training_124999/63196916_Mar24-11-09-21_sd1000_INF_srh7v1test_dev/predictions/pred.pt",
    #    },
    #    {
    #        "name": "1dfffb8f_Mar22-23-45-20_sd1000_dev_maskobw_lr43_tune1",
    #        "databank_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset/1dfffb8f_Mar22-23-45-20_sd1000_dev_maskobw_lr43_tune1/models/eval/training_124999/331492fd_Mar24-11-32-12_sd1000_INF_srh7v1sp1dot4m_dev/predictions/pred.pt",
    #        "test_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset/1dfffb8f_Mar22-23-45-20_sd1000_dev_maskobw_lr43_tune1/models/eval/training_124999/e17e378d_Mar24-11-49-56_sd1000_INF_srh7v1test_dev/predictions/pred.pt",
    #    },
    #    {
    #        "name": "3122d0c0_Mar20-19-19-03_sd1000_dev_dinov2_lr43_tune0",
    #        "databank_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset/3122d0c0_Mar20-19-19-03_sd1000_dev_dinov2_lr43_tune0/models/eval/training_124999/728ba2bc_Mar24-00-36-25_sd1000_INF_srh7v1sp1dot4m_dev/predictions/pred.pt",
    #        "test_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset/3122d0c0_Mar20-19-19-03_sd1000_dev_dinov2_lr43_tune0/models/eval/training_124999/5791644f_Mar24-00-57-48_sd1000_INF_srh7v1test_dev/predictions/pred.pt",
    #    },

    #        {
    #        "name": "6778e5d1_May27-15-59-58_sd1000_dev_tune0",
    #        #"databank_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/ec3c473c_Mar23-23-00-56_sd1000_INF_srh7v1sp1dot4m_dev/predictions/pred.pt",
    #        "databank_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/91ae79af_Mar25-04-30-35_sd1000_INF_srh7v1sp1dot4m_perturbed_dev/predictions/pred.pt",
    #        "test_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/c7929ff4_Mar25-04-12-15_sd1000_INF_srh7v1tests64_dev/predictions/pred.pt",
    #        #"test_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/d8354750_Mar25-01-43-09_sd1000_INF_srh7v1tests64_dev/predictions/pred.pt",
    #        #"test_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/01b41ec3_Mar23-23-18-24_sd1000_INF_srh7v1test_dev/predictions/pred.pt",
    #    },
    #    {
    #        "name": "89d3ad98_May23-13-58-49_sd1000_dev_tune0",
    #        #"databank_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/89d3ad98_May23-13-58-49_sd1000_dev_tune0/models/eval/training_124999/b5568912_Mar24-03-11-55_sd1000_INF_srh7v1sp1dot4m_dev/predictions/pred.pt",
    #        "databank_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/89d3ad98_May23-13-58-49_sd1000_dev_tune0/models/eval/training_124999/0eb38b4c_Mar25-04-27-24_sd1000_INF_srh7v1sp1dot4m_perturbed_dev/predictions/pred.pt",
    #        "test_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/89d3ad98_May23-13-58-49_sd1000_dev_tune0/models/eval/training_124999/bbf81b9c_Mar25-04-09-02_sd1000_INF_srh7v1tests64_dev/predictions/pred.pt",
    #        #"test_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/89d3ad98_May23-13-58-49_sd1000_dev_tune0/models/eval/training_124999/fdca610e_Mar24-03-30-54_sd1000_INF_srh7v1test_dev/predictions/pred.pt",
    #        #"test_pred_path": "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/89d3ad98_May23-13-58-49_sd1000_dev_tune0/models/eval/training_124999/5beb42c2_Mar25-01-48-48_sd1000_INF_srh7v1tests64_dev/predictions/pred.pt",
    #    },
