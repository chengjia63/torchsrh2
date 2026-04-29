import json
import logging
import os

import hydra
import matplotlib
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
import matplotlib as mpl


from ts3.train.infra import config_loggers, prepare_config, register_resolvers
from ts3.train.main import assert_not_hydra_sweep, build_dataloader


register_resolvers()
matplotlib.use("Agg")


def setup_inference_infra(cf: DictConfig) -> None:
    pl.seed_everything(cf.infra.seed, workers=True)
    torch.set_float32_matmul_precision(
        cf.infra.get("float32_matmul_precision", "medium")
    )
    config_loggers(cf.infra.output_dir)


def load_lightning_module(cf: DictConfig):
    checkpoint_path = cf.inference.checkpoint_path
    if checkpoint_path is None:
        raise ValueError("Set inference.checkpoint_path to a Lightning .ckpt file")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = instantiate(cf.meta_arch.model)
    lightning_module = instantiate(cf.meta_arch, model=model)
    lightning_module.load_state_dict(checkpoint["state_dict"])
    return lightning_module


def collect_predictions(prediction_batches):
    rows = []
    full = {
        "path": [],
        "label": [],
        "logits": [],
        "score": [],
        "raw_score": [],
        "pred_label": [],
        "score_pred_label": [],
        "attention": [],
        "pooled_embeddings": [],
    }

    for batch in prediction_batches:
        logits = batch["logits"].detach().cpu()
        score = batch["score"].detach().cpu()
        raw_score = (
            batch["raw_score"].detach().cpu() if "raw_score" in batch else None
        )
        pred_label = batch["pred_label"].detach().cpu()
        score_pred_label = (
            batch["score_pred_label"].detach().cpu()
            if "score_pred_label" in batch
            else None
        )
        label = batch["label"].detach().cpu()

        paths = list(batch["path"])
        full["path"].extend(paths)
        full["label"].append(label)
        full["logits"].append(logits)
        full["score"].append(score)
        if raw_score is not None:
            full["raw_score"].append(raw_score)
        full["pred_label"].append(pred_label)
        if score_pred_label is not None:
            full["score_pred_label"].append(score_pred_label)
        full["attention"].extend(
            [attention.detach().cpu() for attention in batch["attention"]]
        )
        full["pooled_embeddings"].append(batch["pooled_embeddings"].detach().cpu())

        for idx, path in enumerate(paths):
            row = {
                "path": path,
                "label": int(label[idx]),
                "pred_label": int(pred_label[idx]),
                "score": float(score[idx]),
            }
            if raw_score is not None:
                row["raw_score"] = float(raw_score[idx])
                row["sigmoid_sum_score"] = float(score[idx])
            if score_pred_label is not None:
                row["score_pred_label"] = int(score_pred_label[idx])
            for logit_idx, logit in enumerate(logits[idx].tolist()):
                row[f"logit_{logit_idx}"] = float(logit)
            rows.append(row)

    for key in (
        "label",
        "logits",
        "score",
        "pred_label",
        "pooled_embeddings",
    ):
        full[key] = torch.cat(full[key], dim=0)
    if full["raw_score"]:
        full["raw_score"] = torch.cat(full["raw_score"], dim=0)
    if full["score_pred_label"]:
        full["score_pred_label"] = torch.cat(full["score_pred_label"], dim=0)

    return rows, full


def save_predictions(rows, full, output_dir: str) -> None:
    output_path = os.path.join(output_dir, "predictions.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logging.info("Saved %d predictions to %s", len(rows), output_path)


def save_gt_vs_ours_plot(
    rows, num_classes: int, output_dir: str, score_key: str
) -> None:
    import altair as alt

    alt.data_transformers.disable_max_rows()

    df = pd.DataFrame(rows)
    jitter = (pd.util.hash_pandas_object(df["path"], index=False) % 1000) / 999.0
    df["x_jitter"] = df["label"] + (jitter - 0.5) * 0.18
    plot_score_title = score_key.replace("_", " ").capitalize()

    x_axis = alt.X(
        "label:O",
        title="Ground truth",
        sort=list(range(num_classes)),
        axis=alt.Axis(ticks=False),
    )
    y_axis = alt.Y(
        f"{score_key}:Q",
        title=plot_score_title,
        axis=alt.Axis(ticks=False),
    )
    tooltip = [
        alt.Tooltip("path:N", title="Slide"),
        alt.Tooltip("label:Q", title="Ground truth"),
        alt.Tooltip("pred_label:Q", title="Predicted"),
        alt.Tooltip(f"{score_key}:Q", title=plot_score_title, format=".4f"),
    ]
    if score_key != "score":
        tooltip.append(alt.Tooltip("score:Q", title="Score", format=".4f"))
    if "raw_score" in df and score_key != "raw_score":
        tooltip.append(alt.Tooltip("raw_score:Q", title="Raw score", format=".4f"))
    if "sigmoid_sum_score" in df and score_key != "sigmoid_sum_score":
        tooltip.append(
            alt.Tooltip(
                "sigmoid_sum_score:Q",
                title="Sigmoid sum score",
                format=".4f",
            )
        )
    if "score_pred_label" in df:
        tooltip.append(alt.Tooltip("score_pred_label:Q", title="Score predicted"))

    box = (
        alt.Chart(df)
        .mark_boxplot(
            extent="min-max",
            size=48,
            color="#1F1F1F",
            box={"fillOpacity": 0, "stroke": "#1F1F1F"},
            median={"color": "#1F1F1F"},
            rule={"stroke": "#1F1F1F"},
            ticks={"strokeOpacity": 0},
        )
        .encode(
            x=x_axis,
            y=y_axis,
        )
    )

    points = (
        alt.Chart(df)
        .mark_circle(size=42, opacity=0.7, color="#1F1F1F")
        .encode(
            x=alt.X(
                "x_jitter:Q",
                title="Ground truth",
                scale=alt.Scale(domain=[-0.5, num_classes - 0.5]),
                axis=alt.Axis(values=list(range(num_classes)), ticks=False),
            ),
            y=y_axis,
            tooltip=tooltip,
        )
    )

    chart = (box + points).properties(
        width=520,
        height=420,
        title=f"Ground truth vs {plot_score_title.lower()}",
    )
    for extension in ("html", "png", "pdf"):
        output_path = os.path.join(output_dir, f"gt_vs_ours_{score_key}.{extension}")
        chart.save(output_path)
        logging.info("Saved GT vs ours plot to %s", output_path)


def confusion_matrix_from_rows(
    rows, num_classes: int, pred_label_key: str = "pred_label"
) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for row in rows:
        matrix[int(row["label"]), int(row[pred_label_key])] += 1
    return matrix


def save_confusion_matrix_plot(
    rows,
    num_classes: int,
    output_dir: str,
    pred_label_key: str = "pred_label",
    filename: str = "confusion_matrix",
    title: str = "Confusion matrix",
) -> None:
    import matplotlib.pyplot as plt

    output_path = os.path.join(output_dir, filename)

    matrix = confusion_matrix_from_rows(rows, num_classes, pred_label_key).numpy()
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=160)
    image = ax.imshow(matrix, cmap="Blues")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(length=0)

    labels = list(range(num_classes))
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    ax.tick_params(axis="both", length=0)

    threshold = matrix.max() / 2.0
    for gt_label in labels:
        for predicted_label in labels:
            count = matrix[gt_label, predicted_label]
            text_color = "white" if count > threshold else "black"
            ax.text(
                predicted_label,
                gt_label,
                str(int(count)),
                ha="center",
                va="center",
                color=text_color,
            )

    fig.tight_layout()
    fig.savefig(f"{output_path}.png")
    fig.savefig(f"{output_path}.svg")
    fig.savefig(f"{output_path}.pdf")
    plt.close(fig)
    logging.info("Saved confusion matrix to %s", output_path)


def _json_float(value):
    if pd.isna(value):
        return None
    return float(value)


def mean_class_accuracy(label, pred_label, num_classes: int):
    class_accuracies = []
    for class_idx in range(num_classes):
        in_class = label == class_idx
        if in_class.any():
            class_accuracies.append((pred_label[in_class] == label[in_class]).mean())
    return _json_float(pd.Series(class_accuracies).mean())


def threshold_auroc_metrics(df: pd.DataFrame, num_classes: int) -> dict:
    from sklearn.metrics import roc_auc_score

    labels = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    logits = torch.tensor(
        df[[f"logit_{idx}" for idx in range(num_classes - 1)]].to_numpy(),
        dtype=torch.float32,
    )
    if f"logit_{num_classes - 1}" in df:
        logits = torch.tensor(
            df[[f"logit_{idx}" for idx in range(num_classes)]].to_numpy(),
            dtype=torch.float32,
        )

    metrics = {}
    for threshold in range(num_classes - 1):
        target = (labels > threshold).numpy()

        if logits.shape[1] == num_classes - 1:
            binary_score = logits[:, threshold]
        else:
            negative_score = torch.logsumexp(logits[:, : threshold + 1], dim=1)
            positive_score = torch.logsumexp(logits[:, threshold + 1 :], dim=1)
            binary_score = positive_score - negative_score

        negative_name = "".join(str(idx) for idx in range(threshold + 1))
        positive_name = "".join(str(idx) for idx in range(threshold + 1, num_classes))
        metrics[f"auroc_{negative_name}_vs_{positive_name}"] = _json_float(
            roc_auc_score(target, binary_score.numpy())
        )

    return metrics


def compute_metrics(rows, num_classes: int) -> dict:
    df = pd.DataFrame(rows)
    label = df["label"]
    score = df["score"]
    pred_label = df["pred_label"]

    metrics = {
        "num_datapoints": int(len(df)),
        "confusion_matrix": confusion_matrix_from_rows(rows, num_classes).tolist(),
        "pred_label_accuracy": _json_float((pred_label == label).mean()),
        "pred_label_mean_class_accuracy": mean_class_accuracy(
            label, pred_label, num_classes
        ),
        "score_mae": _json_float((score - label).abs().mean()),
        "score_mse": _json_float(((score - label) ** 2).mean()),
        "pred_label_mae": _json_float((pred_label - label).abs().mean()),
        "pred_label_mse": _json_float(((pred_label - label) ** 2).mean()),
    }
    if "sigmoid_sum_score" not in df:
        metrics.update(
            {
                "score_pearson": _json_float(score.corr(label, method="pearson")),
                "score_spearman": _json_float(score.corr(label, method="spearman")),
            }
        )
    if "raw_score" in df:
        metrics.update(
            {
                "raw_score_pearson": _json_float(
                    df["raw_score"].corr(label, method="pearson")
                ),
                "raw_score_spearman": _json_float(
                    df["raw_score"].corr(label, method="spearman")
                ),
            }
        )
    if "sigmoid_sum_score" in df:
        metrics.update(
            {
                "sigmoid_sum_score_pearson": _json_float(
                    df["sigmoid_sum_score"].corr(label, method="pearson")
                ),
                "sigmoid_sum_score_spearman": _json_float(
                    df["sigmoid_sum_score"].corr(label, method="spearman")
                ),
            }
        )
    metrics.update(threshold_auroc_metrics(df, num_classes))
    if "score_pred_label" in df:
        score_pred_label = df["score_pred_label"]
        metrics.update(
            {
                "score_confusion_matrix": confusion_matrix_from_rows(
                    rows, num_classes, "score_pred_label"
                ).tolist(),
                "score_pred_label_accuracy": _json_float(
                    (score_pred_label == label).mean()
                ),
                "score_pred_label_mean_class_accuracy": mean_class_accuracy(
                    label, score_pred_label, num_classes
                ),
                "score_pred_label_mae": _json_float(
                    (score_pred_label - label).abs().mean()
                ),
                "score_pred_label_mse": _json_float(
                    ((score_pred_label - label) ** 2).mean()
                ),
            }
        )
    return metrics


def save_metrics(metrics: dict, output_dir: str) -> None:
    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w", encoding="utf-8") as fd:
        json.dump(metrics, fd, sort_keys=True, indent=4)
    logging.info("Saved metrics to %s", output_path)


@hydra.main(
    version_base=None, config_path=".", config_name="config/slide_embedding_infer"
)
def main(cf: DictConfig):
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    assert_not_hydra_sweep()
    cf = prepare_config(cf)
    setup_inference_infra(cf)

    dataloader = build_dataloader(cf.data.splits[cf.inference.split])

    lightning_module = load_lightning_module(cf)
    lightning_module.eval()

    trainer = instantiate(cf.trainer)
    prediction_batches = trainer.predict(lightning_module, dataloaders=dataloader)

    rows, full = collect_predictions(prediction_batches)
    save_predictions(rows, full, output_dir=cf.infra.output_dir)
    if "sigmoid_sum_score" in rows[0]:
        save_gt_vs_ours_plot(
            rows,
            num_classes=cf.meta_arch.num_classes,
            output_dir=cf.infra.output_dir,
            score_key="raw_score",
        )
        save_gt_vs_ours_plot(
            rows,
            num_classes=cf.meta_arch.num_classes,
            output_dir=cf.infra.output_dir,
            score_key="sigmoid_sum_score",
        )
    else:
        save_gt_vs_ours_plot(
            rows,
            num_classes=cf.meta_arch.num_classes,
            output_dir=cf.infra.output_dir,
            score_key="score",
        )
    save_confusion_matrix_plot(
        rows,
        num_classes=cf.meta_arch.num_classes,
        output_dir=cf.infra.output_dir,
        title="Prediction confusion matrix",
    )
    if "score_pred_label" in rows[0]:
        save_confusion_matrix_plot(
            rows,
            num_classes=cf.meta_arch.num_classes,
            output_dir=cf.infra.output_dir,
            pred_label_key="score_pred_label",
            filename="score_confusion_matrix",
            title="Rounded score confusion matrix",
        )

    metrics = compute_metrics(rows, num_classes=cf.meta_arch.num_classes)
    save_metrics(metrics, output_dir=cf.infra.output_dir)


if __name__ == "__main__":
    main()
