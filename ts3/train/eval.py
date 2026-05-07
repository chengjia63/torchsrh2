import json
import logging
import os
from os.path import join as opj

from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import torch


from ts3.train.infra import prepare_config, register_resolvers, setup_inference_infra
from ts3.train.main import assert_not_hydra_sweep, build_dataloader

register_resolvers()


def load_lightning_module(cf: DictConfig):
    checkpoint_path = cf.inference.checkpoint_path
    if checkpoint_path is None:
        raise ValueError("Set inference.checkpoint_path to a Lightning .ckpt file")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = instantiate(cf.meta_arch.model)
    lightning_module = instantiate(cf.meta_arch, model=model)
    lightning_module.load_state_dict(checkpoint["state_dict"])
    return lightning_module


def _slide_key_from_embedding_path(path: str) -> str:
    slide_key = os.path.basename(os.path.dirname(path))
    if slide_key:
        return slide_key
    return os.path.splitext(os.path.basename(path))[0]


def collect_predictions(prediction_batches: list[dict]) -> dict:
    tensor_keys = ("label", "logits", "raw_score", "pred_label")
    per_slide_keys = ("attention", "cell_score", "cluster", "cluster_contribution")

    full = {key: [] for key in ("path", *tensor_keys, *per_slide_keys)}

    for batch in prediction_batches:
        full["path"].extend(batch["path"])
        for key in tensor_keys:
            full[key].append(batch[key].detach().cpu())
        for key in per_slide_keys:
            full[key].extend(t.detach().cpu() for t in batch[key])

    for key in tensor_keys:
        full[key] = torch.cat(full[key], dim=0)

    return full


def rows_from_full(full: dict) -> list[dict]:
    rows = []
    for idx, path in enumerate(full["path"]):
        row = {
            "path": _slide_key_from_embedding_path(path),
            "label": int(full["label"][idx]),
            "pred_label": int(full["pred_label"][idx]),
            "raw_score": float(full["raw_score"][idx]),
        }
        for logit_idx, logit in enumerate(full["logits"][idx].tolist()):
            row[f"logit_{logit_idx}"] = float(logit)
        rows.append(row)
    return rows


def save_slide_predictions(rows, output_dir: str) -> None:
    prediction_df = pd.DataFrame(rows)
    csv_output_path = opj(output_dir, "slide_predictions.csv")
    prediction_df.to_csv(csv_output_path, index=False)
    logging.info("Saved %d slide predictions to %s", len(rows), csv_output_path)


def save_slide_tensors(full: dict, output_dir: str) -> None:
    path = opj(output_dir, "slide_tensors.pt")
    torch.save(full, path)
    logging.info("Saved slide tensors to %s", path)


def save_gt_vs_ours_plot(
    rows: list[dict], num_classes: int, output_dir: str, score_key: str
) -> None:
    import altair as alt

    alt.data_transformers.disable_max_rows()

    df = pd.DataFrame(rows)
    df["label"] = pd.to_numeric(df["label"])
    df[score_key] = pd.to_numeric(df[score_key])
    jitter = (pd.util.hash_pandas_object(df["path"], index=False) % 1000) / 999.0
    df["x_jitter"] = df["label"] + (jitter - 0.5) * 0.18
    plot_score_title = score_key.replace("_", " ").capitalize()

    x_axis = alt.X(
        "label:O",
        title="Ground truth",
        sort=list(range(num_classes)),
        axis=alt.Axis(ticks=False),
    )
    y_axis = alt.Y(f"{score_key}:Q", title=plot_score_title, axis=alt.Axis(ticks=False))
    tooltip = [
        alt.Tooltip("path:N", title="Slide"),
        alt.Tooltip("label:Q", title="Ground truth"),
        alt.Tooltip("pred_label:Q", title="Predicted"),
        alt.Tooltip(f"{score_key}:Q", title=plot_score_title, format=".4f"),
    ]
    if "raw_score" in df and score_key != "raw_score":
        tooltip.append(alt.Tooltip("raw_score:Q", title="Raw score", format=".4f"))
    box = (
        alt.Chart(df)
        .mark_boxplot(
            extent="min-max",
            size=48,
            color="#1F1F1F",
            box={"fillOpacity": 0, "stroke": "#1F1F1F"},
            median={"color": "#1F1F1F"},
            rule={"stroke": "#1F1F1F"},
            ticks=False,
        )
        .encode(x=x_axis, y=y_axis)
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
        width=520, height=420, title=f"Ground truth vs {plot_score_title.lower()}"
    )
    for extension in ("html", "png", "pdf", "json"):
        output_path = opj(output_dir, f"gt_vs_ours_{score_key}.{extension}")
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
    output_path = opj(output_dir, filename)
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
                predicted_label, gt_label, str(int(count)),
                ha="center", va="center", color=text_color,
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
    sigmoid_raw_score = pd.Series(
        torch.sigmoid(
            torch.tensor(df["raw_score"].to_numpy(), dtype=torch.float32)
        ).numpy(),
        index=df.index,
    )
    pred_label = df["pred_label"]

    metrics = {
        "num_datapoints": int(len(df)),
        "confusion_matrix": confusion_matrix_from_rows(rows, num_classes).tolist(),
        "pred_label_accuracy": _json_float((pred_label == label).mean()),
        "pred_label_mean_class_accuracy": mean_class_accuracy(label, pred_label, num_classes),
        "pred_label_mae": _json_float((pred_label - label).abs().mean()),
        "pred_label_mse": _json_float(((pred_label - label) ** 2).mean()),
        "raw_score_pearson": _json_float(df["raw_score"].corr(label, method="pearson")),
        "raw_score_spearman": _json_float(df["raw_score"].corr(label, method="spearman")),
        "sigmoid_raw_score_pearson": _json_float(sigmoid_raw_score.corr(label, method="pearson")),
        "sigmoid_raw_score_spearman": _json_float(sigmoid_raw_score.corr(label, method="spearman")),
    }
    metrics.update(threshold_auroc_metrics(df, num_classes))
    return metrics


def save_metrics(metrics: dict, output_dir: str) -> None:
    output_path = opj(output_dir, "metrics.json")
    with open(output_path, "w", encoding="utf-8") as fd:
        json.dump(metrics, fd, sort_keys=True, indent=4)
    logging.info("Saved metrics to %s", output_path)


def compute_save_metrics(rows, cf: DictConfig, output_dir: str) -> dict:
    num_classes = cf.meta_arch.num_classes
    save_slide_predictions(rows, output_dir=output_dir)
    save_gt_vs_ours_plot(rows, num_classes=num_classes, output_dir=output_dir, score_key="raw_score")
    save_confusion_matrix_plot(
        rows, num_classes=num_classes, output_dir=output_dir, title="Prediction confusion matrix"
    )
    metrics = compute_metrics(rows, num_classes=num_classes)
    save_metrics(metrics, output_dir=output_dir)
    return metrics


@hydra.main(
    version_base=None, config_path=".", config_name="config/slide_embedding_infer"
)
def main(cf: DictConfig):
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    assert_not_hydra_sweep()
    cf = prepare_config(cf)
    output_dir = setup_inference_infra(cf)

    lightning_module = load_lightning_module(cf)
    lightning_module.eval()

    dataloader = build_dataloader(cf.data.splits[cf.inference.split])
    trainer = instantiate(cf.trainer)
    prediction_batches = trainer.predict(lightning_module, dataloaders=dataloader)

    full = collect_predictions(prediction_batches)
    rows = rows_from_full(full)
    save_slide_tensors(full, output_dir)
    compute_save_metrics(rows, cf, output_dir)
    print(f"Eval output dir: {output_dir}")


if __name__ == "__main__":
    main()
