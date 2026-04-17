from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn


STATIC_DIR = Path(__file__).resolve().parent / "static"
GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "gt.csv"
DEFAULT_INITIAL_SLIDE_KEY = "NIO_UM_937b-4"
DEFAULT_SLIDE_PORTAL_DIR = (
    Path(__file__).resolve().parent
    / "out"
    / "b1a0cbe3"
    / "nio_mouse_1-1"
    / "portal"
)


def _natural_sort_key(value: str) -> list[str | int]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _read_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    assert csv_path.is_file(), f"Ground-truth CSV not found: {csv_path}"

    metadata_by_slide: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"slide", "diagnosis", "infiltration"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        assert not missing_columns, (
            f"Ground-truth CSV is missing required columns: {sorted(missing_columns)}"
        )

        for row in reader:
            slide_value = row.get("slide")
            assert slide_value is not None and slide_value.strip(), (
                "Ground-truth CSV contains a row with an empty slide value"
            )
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

    return metadata_by_slide


def _resolve_slide_dzi_dir(
    portal_dir: Path,
    slide_key: str,
    slide_dzi_root: str | Path | None = None,
) -> Path:
    if slide_dzi_root is not None:
        dzi_root = Path(slide_dzi_root).resolve()
        assert dzi_root.is_dir(), f"Slide DZI root not found: {dzi_root}"
        candidate_dirs = [
            dzi_root / slide_key,
            dzi_root / slide_key / "dzi",
        ]
        dzi_dir = next(
            (
                path
                for path in candidate_dirs
                if path.is_dir() and (path / "color.dzi").is_file()
            ),
            None,
        )
        if dzi_dir is None:
            dzi_dir = next((path for path in candidate_dirs if path.is_dir()), None)
        assert dzi_dir is not None, (
            f"Could not find DZI directory for slide {slide_key!r} under {dzi_root}. "
            f"Expected one of: {candidate_dirs[0]} or {candidate_dirs[1]}"
        )
        return dzi_dir.resolve()

    sibling_dzi_dir = portal_dir.parent / "dzi"
    if sibling_dzi_dir.is_dir():
        return sibling_dzi_dir.resolve()
    return portal_dir.resolve()


def _make_slide_entry(
    portal_dir: Path,
    slide_dzi_root: str | Path | None = None,
) -> dict:
    manifest_path = portal_dir / "slide_manifest.json"
    assert manifest_path.exists(), f"Slide portal manifest not found: {manifest_path}"
    manifest = _read_manifest(manifest_path)
    slide_label = portal_dir.parent.name if portal_dir.name == "portal" else portal_dir.name
    slide_id = str(manifest.get("slide_id", slide_label))
    return {
        "key": slide_label,
        "label": slide_label,
        "slide_id": slide_id,
        "portal_dir": portal_dir.resolve(),
        "dzi_dir": _resolve_slide_dzi_dir(
            portal_dir=portal_dir,
            slide_key=slide_label,
            slide_dzi_root=slide_dzi_root,
        ),
    }


def discover_slide_portals(
    slide_portal_dir: str | Path,
    slide_dzi_root: str | Path | None = None,
) -> list[dict]:
    root = Path(slide_portal_dir).resolve()
    assert root.is_dir(), f"Slide portal directory not found: {root}"

    if (root / "slide_manifest.json").exists():
        slide_entries = [_make_slide_entry(root, slide_dzi_root=slide_dzi_root)]
    else:
        candidate_portals: list[Path] = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if (child / "slide_manifest.json").exists():
                candidate_portals.append(child)
                continue
            portal_dir = child / "portal"
            if (portal_dir / "slide_manifest.json").exists():
                candidate_portals.append(portal_dir)

        assert candidate_portals, (
            "No slide portal directories were found under "
            f"{root}. Expected either a direct portal directory with "
            "slide_manifest.json or child directories that contain portal/slide_manifest.json."
        )
        slide_entries = [
            _make_slide_entry(portal_dir, slide_dzi_root=slide_dzi_root)
            for portal_dir in candidate_portals
        ]

    slide_entries.sort(key=lambda entry: _natural_sort_key(entry["label"]))
    slide_keys = [entry["key"] for entry in slide_entries]
    duplicate_slide_keys = sorted({key for key in slide_keys if slide_keys.count(key) > 1})
    assert not duplicate_slide_keys, (
        "Duplicate slide keys were discovered: " + ", ".join(duplicate_slide_keys)
    )
    return slide_entries


def create_app(
    slide_portal_dir: str | Path | None = None,
    slide_dzi_root: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
) -> FastAPI:
    slide_entries = discover_slide_portals(
        slide_portal_dir or DEFAULT_SLIDE_PORTAL_DIR,
        slide_dzi_root=slide_dzi_root,
    )
    ground_truth_by_slide = _load_ground_truth(ground_truth_path or GROUND_TRUTH_PATH)
    diagnoses: set[str] = set()
    infiltrations: set[str] = set()
    for entry in slide_entries:
        candidate_tokens = {
            _normalize_slide_token(entry["key"]),
            _normalize_slide_token(entry["label"]),
            _normalize_slide_token(entry["slide_id"]),
        }
        matched_metadata = {
            "diagnosis": "UNK",
            "infiltration": "UNK",
        }
        for token in candidate_tokens:
            if token in ground_truth_by_slide:
                matched_metadata = ground_truth_by_slide[token]
                break
        entry["diagnosis"] = matched_metadata["diagnosis"]
        entry["infiltration"] = matched_metadata["infiltration"]
        diagnoses.add(entry["diagnosis"])
        infiltrations.add(entry["infiltration"])

    slide_lookup = {entry["key"]: entry for entry in slide_entries}
    assert STATIC_DIR.is_dir(), f"Static directory not found: {STATIC_DIR}"

    app = FastAPI(title="Silica Single-Cell Portal")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/slides")
    def slides() -> dict:
        default_slide_key = (
            DEFAULT_INITIAL_SLIDE_KEY
            if DEFAULT_INITIAL_SLIDE_KEY in slide_lookup
            else slide_entries[0]["key"]
        )
        return {
            "slides": [
                {
                    "key": entry["key"],
                    "label": entry["label"],
                    "slide_id": entry["slide_id"],
                    "diagnosis": entry["diagnosis"],
                    "infiltration": entry["infiltration"],
                }
                for entry in slide_entries
            ],
            "default_slide_key": default_slide_key,
            "filters": {
                "diagnosis": sorted(diagnoses, key=_natural_sort_key),
                "infiltration": sorted(infiltrations, key=_natural_sort_key),
            },
        }

    @app.get("/portal-assets/{slide_key}/{asset_path:path}", include_in_schema=False)
    def portal_asset(slide_key: str, asset_path: str) -> FileResponse:
        slide_entry = slide_lookup.get(slide_key)
        if slide_entry is None:
            raise HTTPException(status_code=404, detail=f"Unknown slide key: {slide_key}")

        portal_dir = slide_entry["portal_dir"]
        resolved_asset_path = (portal_dir / asset_path).resolve()
        try:
            resolved_asset_path.relative_to(portal_dir)
        except ValueError as error:
            raise HTTPException(status_code=404, detail="Asset path is outside slide portal") from error

        if not resolved_asset_path.exists() or not resolved_asset_path.is_file():
            raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}")

        return FileResponse(resolved_asset_path)

    @app.get("/dzi-assets/{slide_key}/{asset_path:path}", include_in_schema=False)
    def dzi_asset(slide_key: str, asset_path: str) -> FileResponse:
        slide_entry = slide_lookup.get(slide_key)
        if slide_entry is None:
            raise HTTPException(status_code=404, detail=f"Unknown slide key: {slide_key}")

        dzi_dir = slide_entry["dzi_dir"]
        resolved_asset_path = (dzi_dir / asset_path).resolve()
        try:
            resolved_asset_path.relative_to(dzi_dir)
        except ValueError as error:
            raise HTTPException(status_code=404, detail="Asset path is outside slide DZI") from error

        if not resolved_asset_path.exists() or not resolved_asset_path.is_file():
            raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}")

        return FileResponse(resolved_asset_path)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slide-portal-dir",
        default=str(DEFAULT_SLIDE_PORTAL_DIR),
        help=(
            "Either a single portal directory containing slide_manifest.json, or a "
            "parent directory whose child slide folders each contain portal/slide_manifest.json."
        ),
    )
    parser.add_argument(
        "--slide-dzi-dir",
        default=None,
        help=(
            "Optional root directory for DZI assets stored separately from portal "
            "metadata. Each slide is resolved as either <slide-dzi-dir>/<slide_key>/ "
            "or <slide-dzi-dir>/<slide_key>/dzi/."
        ),
    )
    parser.add_argument(
        "--ground-truth-csv",
        default=str(GROUND_TRUTH_PATH),
        help="Ground-truth CSV used to populate portal filters.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        slide_portal_dir=args.slide_portal_dir,
        slide_dzi_root=args.slide_dzi_dir,
        ground_truth_path=args.ground_truth_csv,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
