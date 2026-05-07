from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


SITE_DIR = Path(__file__).resolve().parent
STATIC_DIR = SITE_DIR / "static"
GROUND_TRUTH_PATH = SITE_DIR / "gt.csv"
DEFAULT_INITIAL_SLIDE_KEY = "NIO_UM_937b-4"
DEFAULT_SLIDE_PORTAL_DIR = (
    Path(__file__).resolve().parent / "out" / "b1a0cbe3" / "nio_mouse_1-1" / "portal"
)
LOGGER = logging.getLogger("silica.site")
PREDICTION_CHART_EXTENSION = ".json"
PREDICTION_METRICS_EXTENSION = ".json"
PREDICTION_CHART_SUFFIX = "_fg_score_charts"
PREDICTION_METRICS_SUFFIX = "_metrics"


def _natural_sort_key(value: str) -> list[str | int]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _normalize_slide_token(value: str) -> str:
    normalized = value.strip()
    normalized = re.sub(r"[\\/]+", "-", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _normalize_metadata_value(value: str | None) -> str:
    if value is None:
        return "UNK"
    normalized = value.strip()
    return normalized if normalized else "UNK"


def _load_ground_truth(ground_truth_path: str | Path) -> dict[str, dict[str, str]]:
    csv_path = Path(ground_truth_path).resolve()
    LOGGER.info("Loading ground-truth CSV: %s", csv_path)
    assert csv_path.is_file(), f"Ground-truth CSV not found: {csv_path}"

    metadata_by_slide: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"slide", "diagnosis", "infiltration"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        assert (
            not missing_columns
        ), f"Ground-truth CSV is missing required columns: {sorted(missing_columns)}"

        for row in reader:
            slide_value = row.get("slide")
            assert (
                slide_value is not None and slide_value.strip()
            ), "Ground-truth CSV contains a row with an empty slide value"
            normalized_slide = _normalize_slide_token(slide_value)
            metadata = {
                "diagnosis": _normalize_metadata_value(row.get("diagnosis")),
                "infiltration": _normalize_metadata_value(row.get("infiltration")),
            }

            previous_metadata = metadata_by_slide.get(normalized_slide)
            if previous_metadata is not None and previous_metadata != metadata:
                raise AssertionError(
                    "Conflicting ground-truth metadata for slide "
                    f"{slide_value!r}: {previous_metadata} vs {metadata}"
                )
            metadata_by_slide[normalized_slide] = metadata

    LOGGER.info("Loaded %d ground-truth slide rows", len(metadata_by_slide))
    return metadata_by_slide


def _discover_prediction_charts(
    prediction_chart_dir: str | Path,
) -> tuple[Path | None, list[dict[str, str]]]:
    root = Path(prediction_chart_dir).resolve()
    assert root.is_dir(), f"Prediction chart directory not found: {root}"
    chart_paths = [
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() == PREDICTION_CHART_EXTENSION
            and path.stem.endswith(PREDICTION_CHART_SUFFIX)
        )
    ]
    chart_paths.sort(key=lambda path: _natural_sort_key(path.relative_to(root).as_posix()))
    return root, [
        {
            "id": path.relative_to(root).as_posix(),
            "label": path.stem,
            "experiment": path.stem[: -len(PREDICTION_CHART_SUFFIX)],
            "filename": path.name,
        }
        for path in chart_paths
    ]


def _discover_prediction_metrics(
    prediction_metrics_dir: str | Path,
) -> tuple[Path | None, list[dict[str, str]]]:
    root = Path(prediction_metrics_dir).resolve()
    assert root.is_dir(), f"Prediction metrics directory not found: {root}"
    metrics_paths = [
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() == PREDICTION_METRICS_EXTENSION
            and path.stem.endswith(PREDICTION_METRICS_SUFFIX)
        )
    ]
    metrics_paths.sort(key=lambda path: _natural_sort_key(path.relative_to(root).as_posix()))
    return root, [
        {
            "id": path.relative_to(root).as_posix(),
            "label": path.stem,
            "experiment": path.stem[: -len(PREDICTION_METRICS_SUFFIX)],
            "filename": path.name,
        }
        for path in metrics_paths
    ]


def _read_prediction_chart_spec(chart_path: Path) -> dict:
    assert (
        chart_path.suffix.lower() == PREDICTION_CHART_EXTENSION
    ), f"Unsupported prediction chart type: {chart_path.suffix}"
    with chart_path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    assert isinstance(spec, dict), "Prediction chart JSON spec must be an object"
    return spec


def _read_prediction_metrics(metrics_path: Path) -> dict:
    assert (
        metrics_path.suffix.lower() == PREDICTION_METRICS_EXTENSION
    ), f"Unsupported prediction metrics type: {metrics_path.suffix}"
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    assert isinstance(metrics, dict), "Prediction metrics JSON must be an object"
    return metrics


def _resolve_slide_dzi_dir(
    portal_dir: Path,
    slide_key: str,
    slide_dzi_root: Path | None = None,
) -> Path:
    if slide_dzi_root is not None:
        return slide_dzi_root / slide_key / "dzi"
    return portal_dir.parent / "dzi"


def _make_slide_entry(
    slide_key: str,
    experiment_root: Path,
    experiment_name: str,
    slide_dzi_root: Path | None = None,
) -> dict:
    portal_dir = experiment_root / slide_key / "portal"
    return {
        "key": slide_key,
        "label": slide_key,
        "slide_id": slide_key,
        "experiment": experiment_name,
        "portal_dir": portal_dir,
        "dzi_dir": _resolve_slide_dzi_dir(
            portal_dir=portal_dir,
            slide_key=slide_key,
            slide_dzi_root=slide_dzi_root,
        ),
    }


def _discover_existing_slide_dirs(experiment_root: Path) -> set[str]:
    return {
        entry.name
        for entry in os.scandir(experiment_root)
        if entry.is_dir()
    }


def _assert_unique_slide_entries(slide_entries: list[dict]) -> None:
    seen_slide_keys: set[str] = set()
    duplicate_slide_keys: set[str] = set()
    for entry in slide_entries:
        slide_key = entry["key"]
        if slide_key in seen_slide_keys:
            duplicate_slide_keys.add(slide_key)
        seen_slide_keys.add(slide_key)
    assert (
        not duplicate_slide_keys
    ), "Duplicate slide keys were discovered: " + ", ".join(
        sorted(duplicate_slide_keys, key=_natural_sort_key)
    )


def _discover_slide_portals_from_slide_keys_for_root(
    experiment_root: Path,
    experiment_name: str,
    slide_keys: list[str],
    slide_dzi_root: Path | None = None,
) -> list[dict]:
    started_at = time.perf_counter()
    LOGGER.info(
        "Building slide entries: experiment=%s root=%s slides=%d",
        experiment_name,
        experiment_root,
        len(slide_keys),
    )
    slide_entries: list[dict] = []
    existing_slide_keys = _discover_existing_slide_dirs(experiment_root)
    for slide_key in slide_keys:
        if slide_key not in existing_slide_keys:
            continue
        slide_entries.append(
            _make_slide_entry(
                slide_key,
                experiment_root=experiment_root,
                experiment_name=experiment_name,
                slide_dzi_root=slide_dzi_root,
            )
        )

    LOGGER.info(
        "Sorting slide entries: experiment=%s count=%d",
        experiment_name,
        len(slide_entries),
    )
    slide_entries.sort(key=lambda entry: _natural_sort_key(entry["label"]))
    LOGGER.info(
        "Checking duplicate slide entries: experiment=%s count=%d",
        experiment_name,
        len(slide_entries),
    )
    _assert_unique_slide_entries(slide_entries)
    LOGGER.info(
        "Built slide entries: experiment=%s count=%d elapsed=%.2fs",
        experiment_name,
        len(slide_entries),
        time.perf_counter() - started_at,
    )
    return slide_entries


def discover_slide_portals(
    slide_portal_dir: str | Path,
    slide_keys: list[str],
    slide_dzi_root: str | Path | None = None,
    default_experiment: str | None = None,
) -> tuple[list[dict], list[str]]:
    root = Path(slide_portal_dir).resolve()
    LOGGER.info("Preparing slide portal records from root: %s", root)
    assert root.is_dir(), f"Slide portal directory not found: {root}"
    resolved_slide_dzi_root = None
    if slide_dzi_root is not None:
        resolved_slide_dzi_root = Path(slide_dzi_root).resolve()
        assert resolved_slide_dzi_root.is_dir(), (
            f"Slide DZI root not found: {resolved_slide_dzi_root}"
        )

    is_multi_experiment_root = (
        default_experiment is not None and (root / default_experiment).is_dir()
    )
    if default_experiment is None and slide_keys:
        first_slide_key = slide_keys[0]
        is_multi_experiment_root = not (
            (root / first_slide_key).exists()
            or (root / first_slide_key / "portal").exists()
        )

    if not is_multi_experiment_root:
        experiment_name = root.name or "default"
        LOGGER.info(
            "Using single-experiment portal layout: experiment=%s slides=%d",
            experiment_name,
            len(slide_keys),
        )
        slide_entries = _discover_slide_portals_from_slide_keys_for_root(
            experiment_root=root,
            experiment_name=experiment_name,
            slide_keys=slide_keys,
            slide_dzi_root=resolved_slide_dzi_root,
        )
        return slide_entries, [experiment_name]

    slide_entries: list[dict] = []
    discovered_experiments: list[str] = []
    LOGGER.info("Using multi-experiment portal layout")
    for experiment_root in sorted(
        root.iterdir(), key=lambda path: _natural_sort_key(path.name)
    ):
        if not experiment_root.is_dir():
            continue
        LOGGER.info(
            "Registering experiment=%s slides=%d",
            experiment_root.name,
            len(slide_keys),
        )
        experiment_started_at = time.perf_counter()
        experiment_slide_entries = _discover_slide_portals_from_slide_keys_for_root(
            experiment_root=experiment_root,
            experiment_name=experiment_root.name,
            slide_keys=slide_keys,
            slide_dzi_root=resolved_slide_dzi_root,
        )
        if not experiment_slide_entries:
            LOGGER.info(
                "Skipping experiment=%s because it has no matching slide portals",
                experiment_root.name,
            )
            continue
        discovered_experiments.append(experiment_root.name)
        slide_entries.extend(experiment_slide_entries)
        LOGGER.info(
            "Registered experiment=%s records=%d elapsed=%.2fs",
            experiment_root.name,
            len(experiment_slide_entries),
            time.perf_counter() - experiment_started_at,
        )

    assert discovered_experiments, f"No experiment directories were found under {root}"
    LOGGER.info(
        "Prepared %d slide/experiment records across %d experiments",
        len(slide_entries),
        len(discovered_experiments),
    )
    return slide_entries, sorted(set(discovered_experiments), key=_natural_sort_key)


def create_app(
    slide_portal_dir: str | Path | None = None,
    slide_dzi_root: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    default_experiment: str | None = None,
    prediction_chart_dir: str | Path | None = None,
    prediction_metrics_dir: str | Path | None = None,
) -> FastAPI:
    started_at = time.perf_counter()
    LOGGER.info("Creating Silica portal backend")
    ground_truth_by_slide = _load_ground_truth(ground_truth_path or GROUND_TRUTH_PATH)
    slide_entries, discovered_experiments = discover_slide_portals(
        slide_portal_dir or DEFAULT_SLIDE_PORTAL_DIR,
        slide_dzi_root=slide_dzi_root,
        slide_keys=list(ground_truth_by_slide.keys()),
        default_experiment=default_experiment,
    )
    LOGGER.info(
        "Prepared %d slide/experiment records from %d GT slides",
        len(slide_entries),
        len(ground_truth_by_slide),
    )
    diagnoses: set[str] = set()
    infiltrations: set[str] = set()
    slides_by_key: dict[str, dict] = {}
    for entry in slide_entries:
        matched_metadata = ground_truth_by_slide[entry["key"]]
        entry["diagnosis"] = matched_metadata["diagnosis"]
        entry["infiltration"] = matched_metadata["infiltration"]
        diagnoses.add(entry["diagnosis"])
        infiltrations.add(entry["infiltration"])
        slide_record = slides_by_key.setdefault(
            entry["key"],
            {
                "key": entry["key"],
                "label": entry["label"],
                "slide_id": entry["slide_id"],
                "diagnosis": entry["diagnosis"],
                "infiltration": entry["infiltration"],
                "available_experiments": [],
            },
        )
        slide_record["available_experiments"].append(entry["experiment"])
        assert (
            slide_record["diagnosis"] == entry["diagnosis"]
        ), f"Inconsistent diagnosis metadata for slide {entry['key']!r}"
        assert (
            slide_record["infiltration"] == entry["infiltration"]
        ), f"Inconsistent infiltration metadata for slide {entry['key']!r}"

    slide_variant_lookup = {
        (entry["key"], entry["experiment"]): entry for entry in slide_entries
    }
    for slide_record in slides_by_key.values():
        slide_record["available_experiments"] = sorted(
            set(slide_record["available_experiments"]),
            key=_natural_sort_key,
        )

    if default_experiment is not None:
        assert default_experiment in discovered_experiments, (
            f"Default experiment {default_experiment!r} was not found. "
            f"Available experiments: {discovered_experiments}"
        )
    resolved_default_experiment = default_experiment or discovered_experiments[0]

    candidate_default_slide_keys = [
        slide_key
        for slide_key in sorted(slides_by_key.keys(), key=_natural_sort_key)
        if resolved_default_experiment
        in slides_by_key[slide_key]["available_experiments"]
    ]
    assert (
        candidate_default_slide_keys
    ), f"No slides are available for default experiment {resolved_default_experiment!r}"
    default_slide_key = (
        DEFAULT_INITIAL_SLIDE_KEY
        if DEFAULT_INITIAL_SLIDE_KEY in candidate_default_slide_keys
        else candidate_default_slide_keys[0]
    )
    LOGGER.info(
        "Resolved default selection: experiment=%s slide=%s",
        resolved_default_experiment,
        default_slide_key,
    )
    assert (
        prediction_metrics_dir is not None
    ), "--prediction-metrics-dir is required for the predictions page"
    resolved_prediction_chart_dir = prediction_chart_dir or prediction_metrics_dir
    prediction_chart_root, prediction_charts = _discover_prediction_charts(
        resolved_prediction_chart_dir
    )
    prediction_metrics_root, prediction_metrics = _discover_prediction_metrics(
        prediction_metrics_dir
    )
    prediction_experiments = {
        artifact["experiment"] for artifact in prediction_charts + prediction_metrics
    }
    available_experiments = sorted(
        set(discovered_experiments) | prediction_experiments,
        key=_natural_sort_key,
    )
    assert STATIC_DIR.is_dir(), f"Static directory not found: {STATIC_DIR}"

    app = FastAPI(title="Silica Single-Cell Portal")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    LOGGER.info(
        "Backend ready: slides=%d experiments=%d diagnoses=%d infiltrations=%d elapsed=%.2fs",
        len(slides_by_key),
        len(available_experiments),
        len(diagnoses),
        len(infiltrations),
        time.perf_counter() - started_at,
    )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/slideviewer", include_in_schema=False)
    def slideviewer() -> FileResponse:
        return FileResponse(STATIC_DIR / "slideviewer.html")

    @app.get("/slideviewer/", include_in_schema=False)
    def slideviewer_slash() -> FileResponse:
        return FileResponse(STATIC_DIR / "slideviewer.html")

    @app.get("/predictions", include_in_schema=False)
    def predictions() -> FileResponse:
        return FileResponse(STATIC_DIR / "predictions.html")

    @app.get("/predictions/", include_in_schema=False)
    def predictions_slash() -> FileResponse:
        return FileResponse(STATIC_DIR / "predictions.html")

    @app.get("/api/slides")
    def slides() -> dict:
        LOGGER.info("Serving /api/slides")
        return {
            "slides": [
                slides_by_key[key]
                for key in sorted(slides_by_key.keys(), key=_natural_sort_key)
            ],
            "default_slide_key": default_slide_key,
            "default_experiment": resolved_default_experiment,
            "experiments": available_experiments,
            "filters": {
                "diagnosis": sorted(diagnoses, key=_natural_sort_key),
                "infiltration": sorted(infiltrations, key=_natural_sort_key),
            },
        }

    @app.get("/api/prediction-charts")
    def prediction_chart_index() -> dict:
        LOGGER.info("Serving /api/prediction-charts")
        return {"charts": prediction_charts}

    @app.get("/api/prediction-charts/{chart_id:path}")
    def prediction_chart_spec(chart_id: str) -> dict:
        if prediction_chart_root is None:
            raise HTTPException(
                status_code=404, detail="Prediction chart directory is not configured"
            )
        chart_path = (prediction_chart_root / chart_id).resolve()
        try:
            chart_path.relative_to(prediction_chart_root)
        except ValueError as error:
            raise HTTPException(
                status_code=404, detail="Chart path is outside chart root"
            ) from error
        if (
            not chart_path.is_file()
            or chart_path.suffix.lower() != PREDICTION_CHART_EXTENSION
        ):
            raise HTTPException(
                status_code=404, detail=f"Prediction chart not found: {chart_id}"
            )
        LOGGER.info("Serving prediction chart spec: %s", chart_id)
        return _read_prediction_chart_spec(chart_path)

    @app.get("/api/prediction-metrics")
    def prediction_metrics_index() -> dict:
        LOGGER.info("Serving /api/prediction-metrics")
        return {"metrics": prediction_metrics}

    @app.get("/api/prediction-metrics/{metrics_id:path}")
    def prediction_metrics_payload(metrics_id: str) -> dict:
        if prediction_metrics_root is None:
            raise HTTPException(
                status_code=404, detail="Prediction metrics directory is not configured"
            )
        metrics_path = (prediction_metrics_root / metrics_id).resolve()
        try:
            metrics_path.relative_to(prediction_metrics_root)
        except ValueError as error:
            raise HTTPException(
                status_code=404, detail="Metrics path is outside metrics root"
            ) from error
        if (
            not metrics_path.is_file()
            or metrics_path.suffix.lower() != PREDICTION_METRICS_EXTENSION
        ):
            raise HTTPException(
                status_code=404, detail=f"Prediction metrics not found: {metrics_id}"
            )
        LOGGER.info("Serving prediction metrics: %s", metrics_id)
        return _read_prediction_metrics(metrics_path)

    @app.get(
        "/portal-assets/{experiment_name}/{slide_key}/{asset_path:path}",
        include_in_schema=False,
    )
    def portal_asset(
        experiment_name: str, slide_key: str, asset_path: str
    ) -> FileResponse:
        slide_entry = slide_variant_lookup.get((slide_key, experiment_name))
        if slide_entry is None:
            LOGGER.info(
                "Unknown portal asset request: experiment=%s slide=%s asset=%s",
                experiment_name,
                slide_key,
                asset_path,
            )
            raise HTTPException(
                status_code=404,
                detail=f"Unknown slide/experiment combination: {slide_key} @ {experiment_name}",
            )

        portal_dir = slide_entry["portal_dir"]
        resolved_asset_path = (portal_dir / asset_path).resolve()
        if asset_path in {"slide_manifest.json", "cells.json"}:
            LOGGER.info(
                "Serving portal asset: experiment=%s slide=%s asset=%s",
                experiment_name,
                slide_key,
                asset_path,
            )
        try:
            resolved_asset_path.relative_to(portal_dir)
        except ValueError as error:
            raise HTTPException(
                status_code=404, detail="Asset path is outside slide portal"
            ) from error

        if not resolved_asset_path.exists() or not resolved_asset_path.is_file():
            LOGGER.info(
                "Missing portal asset: experiment=%s slide=%s asset=%s path=%s",
                experiment_name,
                slide_key,
                asset_path,
                resolved_asset_path,
            )
            raise HTTPException(
                status_code=404, detail=f"Asset not found: {asset_path}"
            )

        return FileResponse(resolved_asset_path)

    @app.get("/dzi-assets/{slide_key}/{asset_path:path}", include_in_schema=False)
    def dzi_asset(slide_key: str, asset_path: str) -> FileResponse:
        slide_candidates = [
            entry for entry in slide_entries if entry["key"] == slide_key
        ]
        if not slide_candidates:
            LOGGER.info(
                "Unknown DZI asset request: slide=%s asset=%s",
                slide_key,
                asset_path,
            )
            raise HTTPException(
                status_code=404, detail=f"Unknown slide key: {slide_key}"
            )
        slide_entry = slide_candidates[0]

        dzi_dir = slide_entry["dzi_dir"]
        resolved_asset_path = (dzi_dir / asset_path).resolve()
        if asset_path.endswith(".dzi"):
            LOGGER.info(
                "Serving DZI metadata: slide=%s asset=%s",
                slide_key,
                asset_path,
            )
        try:
            resolved_asset_path.relative_to(dzi_dir)
        except ValueError as error:
            raise HTTPException(
                status_code=404, detail="Asset path is outside slide DZI"
            ) from error

        if not resolved_asset_path.exists() or not resolved_asset_path.is_file():
            LOGGER.info(
                "Missing DZI asset: slide=%s asset=%s path=%s",
                slide_key,
                asset_path,
                resolved_asset_path,
            )
            raise HTTPException(
                status_code=404, detail=f"Asset not found: {asset_path}"
            )

        return FileResponse(resolved_asset_path)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slide-portal-dir",
        default=str(DEFAULT_SLIDE_PORTAL_DIR),
        help=(
            "Root containing slide portal assets as <slide>/portal or "
            "<experiment>/<slide>/portal. Slide names come from the "
            "ground-truth CSV."
        ),
    )
    parser.add_argument(
        "--slide-dzi-dir",
        default=None,
        help=(
            "Optional root directory for DZI assets stored separately from portal "
            "metadata. Each slide is resolved as <slide-dzi-dir>/<slide_key>/dzi/."
        ),
    )
    parser.add_argument(
        "--ground-truth-csv",
        default=str(GROUND_TRUTH_PATH),
        help="Ground-truth CSV used to populate portal filters.",
    )
    parser.add_argument(
        "--default-experiment",
        default=None,
        help=(
            "Optional experiment directory name to select by default when "
            "using a multi-experiment root. If omitted, the first experiment "
            "in natural-sort order is used."
        ),
    )
    parser.add_argument(
        "--prediction-chart-dir",
        default=None,
        help=(
            "Optional directory containing Altair prediction charts exported as "
            ".json Vega-Lite specs."
        ),
    )
    parser.add_argument(
        "--prediction-metrics-dir",
        required=True,
        help=(
            "Directory containing prediction artifacts as "
            "*_fg_score_charts.json and *_metrics.json files."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    LOGGER.info("Starting server on %s:%s", args.host, args.port)
    app = create_app(
        slide_portal_dir=args.slide_portal_dir,
        slide_dzi_root=args.slide_dzi_dir,
        ground_truth_path=args.ground_truth_csv,
        default_experiment=args.default_experiment,
        prediction_chart_dir=args.prediction_chart_dir,
        prediction_metrics_dir=args.prediction_metrics_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
