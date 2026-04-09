import json
import os
import re
from glob import glob
from os.path import join as opj
from typing import Dict, List, Optional

import altair as alt
import pandas as pd
from tqdm import tqdm

from ts2.utils.silica_sc_cls.eval_cell_inference_knn import build_runs_from_sets


def extract_perturbation_percent(test_pred_path: str, run_key_prefix: str) -> int:
    match = re.search(rf"{re.escape(run_key_prefix)}(\d+)", test_pred_path)
    if match is None:
        raise ValueError(
            f"Could not parse perturbation value from test prediction path: {test_pred_path}"
        )
    return int(match.group(1))


def find_latest_summary_dir(
    test_pred_path: str,
    run_dir_prefix: str = "run",
) -> str:
    pred_dir = os.path.dirname(test_pred_path)
    if os.path.basename(pred_dir) != "predictions":
        raise ValueError(
            f"Expected test prediction path under a predictions directory, got {test_pred_path}"
        )
    if not re.fullmatch(r"[A-Za-z]+", run_dir_prefix):
        raise ValueError(
            f"Expected run_dir_prefix to contain only letters, got {run_dir_prefix!r}"
        )

    results_root = opj(os.path.dirname(pred_dir), "results")
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Results directory does not exist: {results_root}")

    run_dirs = []
    for path in glob(opj(results_root, f"{run_dir_prefix}*")):
        if not os.path.isdir(path):
            continue
        run_name = os.path.basename(path)
        match = re.fullmatch(rf"{re.escape(run_dir_prefix)}(\d{{4}})", run_name)
        if match is None:
            continue
        summary_path = opj(path, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        run_dirs.append((int(match.group(1)), path))

    if not run_dirs:
        raise FileNotFoundError(
            "No completed result directories with summary.json found under "
            f"{results_root} for prefix {run_dir_prefix!r}"
        )

    return max(run_dirs, key=lambda item: item[0])[1]


def load_summary(summary_dir: str) -> Dict:
    summary_path = opj(summary_dir, "summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as fd:
        return json.load(fd)


def build_metrics_df(
    exp_root: str,
    ckpt: str,
    run_key_prefix: str,
    run_sets: List[Dict[str, str]],
    panels: List[Dict[str, object]],
    display_name_by_exp: Dict[str, str],
    run_dir_prefix: str = "run",
) -> pd.DataFrame:
    runs = build_runs_from_sets(
        exp_root=exp_root,
        ckpt=ckpt,
        run_sets=run_sets,
        run_key_prefix=run_key_prefix,
    )

    rows = []
    for run in tqdm(runs, desc="Loading SC-CLS summaries"):
        summary_dir = find_latest_summary_dir(
            run["test_pred_path"], run_dir_prefix=run_dir_prefix
        )
        summary = load_summary(summary_dir)

        exp_name = run["name"].rsplit(f"_{run_key_prefix}", maxsplit=1)[0]
        model_name = display_name_by_exp.get(exp_name, exp_name)
        perturbation = extract_perturbation_percent(
            run["test_pred_path"], run_key_prefix=run_key_prefix
        )

        row = {
            "exp_name": exp_name,
            "model": model_name,
            "perturbation": perturbation,
            "summary_dir": summary_dir,
        }
        for panel in panels:
            summary_key = panel["summary_key"]
            metric_key = panel["metric_key"]
            if summary_key not in summary:
                raise KeyError(
                    f"Summary at {summary_dir} is missing key {summary_key!r}. "
                    "This usually means the evaluation was not run with the expected outputs."
                )
            metrics = summary[summary_key].get("metrics")
            if not isinstance(metrics, dict):
                raise TypeError(
                    f"Expected metrics dict under {summary_key!r} in {summary_dir}, got {type(metrics)}"
                )
            if metric_key not in metrics:
                raise KeyError(
                    f"Metrics at {summary_dir} under {summary_key!r} are missing {metric_key!r}"
                )
            row[panel["title"]] = float(metrics[metric_key])
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        raise ValueError("No metrics rows were collected")

    metrics_df = metrics_df.sort_values(
        ["model", "perturbation", "exp_name"], ignore_index=True
    )
    return metrics_df


def make_panel(
    metrics_df: pd.DataFrame,
    title: str,
    baseline: Optional[float],
    width: int,
    height: int,
    color_scale: alt.Scale,
) -> alt.Chart:
    plot_df = metrics_df[
        ["perturbation", "model", "exp_name", "summary_dir", title]
    ].rename(columns={title: "value"})

    lines = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "perturbation:Q",
                title="% Swapped Tokens",
                axis=alt.Axis(ticks=False),
            ),
            y=alt.Y(
                "value:Q",
                title="MCA",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(ticks=False),
            ),
            color=alt.Color(
                "model:N",
                title=None,
                scale=color_scale,
                legend=alt.Legend(labelLimit=0),
            ),
            detail="exp_name:N",
            tooltip=[
                "model:N",
                "exp_name:N",
                "perturbation:Q",
                alt.Tooltip("value:Q", format=".4f"),
                "summary_dir:N",
            ],
        )
    )

    chart = lines
    if baseline is not None:
        baseline_df = pd.DataFrame({"baseline": [baseline]})
        baseline_rule = (
            alt.Chart(baseline_df)
            .mark_rule(strokeDash=[6, 4], strokeWidth=2, color="grey")
            .encode(
                y="baseline:Q",
                tooltip=[alt.Tooltip("baseline:Q", format=".4f")],
            )
        )
        chart = lines + baseline_rule

    return chart.properties(title=title, width=width, height=height)


def build_chart(
    metrics_df: pd.DataFrame,
    panels: List[Dict[str, object]],
    color_range: List[str],
    width: int,
    height: int,
) -> alt.Chart:
    model_domain = list(metrics_df["model"].drop_duplicates())
    color_scale = alt.Scale(
        domain=model_domain,
        range=color_range[: len(model_domain)],
    )

    return (
        alt.hconcat(
            *[
                make_panel(
                    metrics_df=metrics_df,
                    title=panel["title"],
                    baseline=panel["baseline"],
                    width=width,
                    height=height,
                    color_scale=color_scale,
                )
                for panel in panels
            ]
        )
        .resolve_scale(color="shared")
        .configure_title(fontSize=22)
        .configure_axis(titleFontSize=18, labelFontSize=18)
        .configure_legend(labelFontSize=18)
    )


def save_chart(chart: alt.Chart, out_stem: str, formats: List[str]) -> None:
    out_dir = os.path.dirname(out_stem)
    if out_dir:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            print(f"Created output directory: {out_dir}")

    for fmt in formats:
        out_path = f"{out_stem}.{fmt}"
        try:
            chart.save(out_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to save chart to {out_path}. "
                "If you requested png/pdf, ensure Altair export dependencies such as "
                "`vl-convert-python` are installed."
            ) from exc


def main() -> None:
    exp_root = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_fixdset2"
    ckpt = "training_124999"
    run_key_prefix = "PERTURB"
    run_dir_prefix = "run"
    out_stem = "cell_inference_knn_perturbation"
    formats = ["html", "png", "pdf"]
    width = 250
    height = 300
    run_sets = [
        {
            "exp_name": "04e0bf39_Apr05-03-07-21_sd1000_dinov2_lr43_tune0",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        },
        {
            "exp_name": "ca187b7c_Apr05-03-07-13_sd1000_nomaskobw_lr43_tune0",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        },
        {
            "exp_name": "a2706135_dinov2",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        },
        {
            "exp_name": "78d57cfc_Apr06-12-13-26_sd1000_dinov2_rmbg_lr43_tune0",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        },
        {
            "exp_name": "844ffd45_Apr06-12-07-47_sd1000_maskobw_lr43_tune1",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        },
        {
            "exp_name": "b1a0cbe3_Apr07-21-09-04_sd1000_nomaskobw_lr13_tune0",
            "databank_pred_glob": "*_INF_srh7v1sp1dot4m_*",
            "test_pred_glob": "*_INF_srh7v1tests64_PERTURB*_*",
        },
    ]
    panels = [
        {
            "title": "SRH7 Cell",
            "summary_key": "instance_knn",
            "metric_key": "mca",
            "baseline": None,
        },
        {
            "title": "SRH7 Slide",
            "summary_key": "mosaic_vote",
            "metric_key": "mca",
            "baseline": 0.855,
        },
        {
            "title": "Tumor/Normal Cell",
            "summary_key": "instance_knn_binary",
            "metric_key": "mca",
            "baseline": None,
        },
        {
            "title": "Tumor/Normal Slide",
            "summary_key": "mosaic_vote_binary",
            "metric_key": "mca",
            "baseline": 0.958,
        },
    ]
    display_name_by_exp = {
        "a2706135_dinov2": "DINOv2 Meta",
        "04e0bf39_Apr05-03-07-21_sd1000_dinov2_lr43_tune0": "DINOv2 lr4e-3",
        "ca187b7c_Apr05-03-07-13_sd1000_nomaskobw_lr43_tune0": "Silica FullIm iBOT lr4e-3",
        "78d57cfc_Apr06-12-13-26_sd1000_dinov2_rmbg_lr43_tune0": "DINOv2 lr4e-3 RmBg",
        "844ffd45_Apr06-12-07-47_sd1000_maskobw_lr43_tune1": "Silica Inside iBOT lr4e-3",
        "b1a0cbe3_Apr07-21-09-04_sd1000_nomaskobw_lr13_tune0": "Silica FullIm iBOT lr1e-3",
    }

    color_range = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

    metrics_df = build_metrics_df(
        exp_root=exp_root,
        ckpt=ckpt,
        run_key_prefix=run_key_prefix,
        run_dir_prefix=run_dir_prefix,
        run_sets=run_sets,
        panels=panels,
        display_name_by_exp=display_name_by_exp,
    )
    chart = build_chart(
        metrics_df,
        panels=panels,
        color_range=color_range,
        width=width,
        height=height,
    )

    csv_path = f"{out_stem}.csv"
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            print(f"Created output directory: {out_dir}")
    metrics_df.to_csv(csv_path, index=False)

    save_chart(chart, out_stem=out_stem, formats=formats)
    print(f"Saved metrics table to {csv_path}")
    print("Saved chart files: " + ", ".join(f"{out_stem}.{fmt}" for fmt in formats))


if __name__ == "__main__":
    main()
