from __future__ import annotations

import argparse
import csv
import logging
import re
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from tqdm import tqdm
import uvicorn


STATIC_DIR = Path(__file__).resolve().parent / "static"
GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "gt.csv"
DEFAULT_INITIAL_SLIDE_KEY = "NIO_UM_937b-4"
DEFAULT_SLIDE_PORTAL_DIR = (
    Path(__file__).resolve().parent / "out" / "b1a0cbe3" / "nio_mouse_1-1" / "portal"
)
LOGGER = logging.getLogger("silica.site")


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
    for slide_key in tqdm(
        slide_keys,
        desc=f"register {experiment_name}",
        unit="slide",
        dynamic_ncols=True,
    ):
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
        discovered_experiments.append(experiment_root.name)
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
    assert STATIC_DIR.is_dir(), f"Static directory not found: {STATIC_DIR}"

    app = FastAPI(title="Silica Single-Cell Portal")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    LOGGER.info(
        "Backend ready: slides=%d experiments=%d diagnoses=%d infiltrations=%d elapsed=%.2fs",
        len(slides_by_key),
        len(discovered_experiments),
        len(diagnoses),
        len(infiltrations),
        time.perf_counter() - started_at,
    )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

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
            "experiments": discovered_experiments,
            "filters": {
                "diagnosis": sorted(diagnoses, key=_natural_sort_key),
                "infiltration": sorted(infiltrations, key=_natural_sort_key),
            },
        }

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
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
