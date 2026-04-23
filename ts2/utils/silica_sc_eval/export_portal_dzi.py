import argparse
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

from ts2.utils.silica_sc_eval.generate_gmm_visualization import load_mosaic_image
from ts2.utils.silica_sc_eval.run_single_cell_inference import (
    infer_mosaic_io_paths,
    parse_mosaic_run,
    select_mosaic_runs_for_task,
)

logger = logging.getLogger(__name__)

try:
    import pyvips
except Exception:  # pragma: no cover - optional runtime dependency
    pyvips = None


_LANCZOS = (
    Image.Resampling.LANCZOS
    if hasattr(Image, "Resampling")
    else Image.LANCZOS
)


def _ensure_dir(path: str | os.PathLike[str]) -> Path:
    out_path = Path(path)
    out_path.mkdir(parents=True, exist_ok=True)
    assert out_path.is_dir(), f"Failed to create directory: {out_path}"
    return out_path


def _write_dzi_descriptor(
    descriptor_path: Path,
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
    tile_format: str,
) -> None:
    descriptor = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Image TileSize="{tile_size}" Overlap="{overlap}" Format="{tile_format}" '
        'xmlns="http://schemas.microsoft.com/deepzoom/2008">\n'
        f'  <Size Width="{width}" Height="{height}"/>\n'
        "</Image>\n"
    )
    descriptor_path.write_text(descriptor, encoding="utf-8")
    assert descriptor_path.exists(), f"Failed to write DZI descriptor: {descriptor_path}"


def _save_tile_pyramid_with_pil(
    image: Image.Image,
    output_prefix: Path,
    tile_size: int,
    overlap: int,
    jpeg_quality: int,
    tile_format: str = "jpg",
) -> tuple[str, str]:
    width, height = image.size
    max_level = (
        int(math.ceil(math.log2(max(width, height)))) if max(width, height) > 1 else 0
    )
    descriptor_path = output_prefix.with_suffix(".dzi")
    files_dir = output_prefix.parent / f"{output_prefix.name}_files"
    _ensure_dir(files_dir)
    _write_dzi_descriptor(
        descriptor_path=descriptor_path,
        width=width,
        height=height,
        tile_size=tile_size,
        overlap=overlap,
        tile_format=tile_format,
    )

    current_image = image.copy()
    current_level = max_level
    while True:
        level_dir = files_dir / str(current_level)
        _ensure_dir(level_dir)
        level_width, level_height = current_image.size
        num_cols = int(math.ceil(level_width / tile_size))
        num_rows = int(math.ceil(level_height / tile_size))

        for row in range(num_rows):
            for col in range(num_cols):
                left = col * tile_size
                top = row * tile_size
                right = min(left + tile_size, level_width)
                bottom = min(top + tile_size, level_height)
                crop_left = max(0, left - (overlap if col > 0 else 0))
                crop_top = max(0, top - (overlap if row > 0 else 0))
                crop_right = min(
                    level_width,
                    right + (overlap if col < num_cols - 1 else 0),
                )
                crop_bottom = min(
                    level_height,
                    bottom + (overlap if row < num_rows - 1 else 0),
                )
                tile = current_image.crop((crop_left, crop_top, crop_right, crop_bottom))
                tile_path = level_dir / f"{col}_{row}.{tile_format}"
                tile.save(tile_path, format="JPEG", quality=jpeg_quality)

        if current_level == 0:
            break
        next_size = (
            max(1, int(math.ceil(level_width / 2.0))),
            max(1, int(math.ceil(level_height / 2.0))),
        )
        current_image = current_image.resize(next_size, _LANCZOS)
        current_level -= 1

    return descriptor_path.name, files_dir.name


def _save_tile_pyramid_with_pyvips(
    image: Image.Image,
    output_prefix: Path,
    tile_size: int,
    overlap: int,
    jpeg_quality: int,
) -> tuple[str, str]:
    assert pyvips is not None, "pyvips is not available."
    rgb = np.asarray(image, dtype=np.uint8)
    assert rgb.ndim == 3 and rgb.shape[2] == 3, (
        f"Expected RGB image array, got shape {rgb.shape}"
    )
    vips_image = pyvips.Image.new_from_memory(
        rgb.tobytes(),
        rgb.shape[1],
        rgb.shape[0],
        rgb.shape[2],
        format="uchar",
    )
    vips_image.dzsave(
        str(output_prefix),
        tile_size=tile_size,
        overlap=overlap,
        layout="dz",
        suffix=f".jpg[Q={jpeg_quality}]",
    )
    descriptor_path = output_prefix.with_suffix(".dzi")
    files_dir = output_prefix.parent / f"{output_prefix.name}_files"
    assert (
        descriptor_path.exists()
    ), f"Missing DZI descriptor after pyvips export: {descriptor_path}"
    assert files_dir.is_dir(), f"Missing DZI tile directory after pyvips export: {files_dir}"
    return descriptor_path.name, files_dir.name


def save_image_as_dzi(
    image: Image.Image,
    output_prefix: str | os.PathLike[str],
    tile_size: int = 256,
    overlap: int = 0,
    jpeg_quality: int = 88,
) -> tuple[str, str]:
    output_prefix = Path(output_prefix)
    _ensure_dir(output_prefix.parent)
    image = image.convert("RGB")
    if pyvips is not None:
        logger.info("Saving DZI with pyvips: %s", output_prefix)
        return _save_tile_pyramid_with_pyvips(
            image=image,
            output_prefix=output_prefix,
            tile_size=tile_size,
            overlap=overlap,
            jpeg_quality=jpeg_quality,
        )

    logger.warning(
        "pyvips is unavailable; falling back to PIL DZI export for %s. "
        "This is correct but slower for large slides.",
        output_prefix,
    )
    return _save_tile_pyramid_with_pil(
        image=image,
        output_prefix=output_prefix,
        tile_size=tile_size,
        overlap=overlap,
        jpeg_quality=jpeg_quality,
    )


def export_slide_dzi_assets(
    *,
    slide_id: str,
    mosaic_dicom_path: str,
    dzi_dir: str,
    srhrgb_image_path: str | None = None,
    skip_existing: bool = False,
    strip_padding: int = 50,
    tile_size: int = 256,
    overlap: int = 0,
    jpeg_quality: int = 88,
) -> dict[str, str]:
    logger.info("Exporting DZI assets for %s", slide_id)
    dzi_path = _ensure_dir(dzi_dir)
    if skip_existing and _dzi_asset_exists(dzi_path, "srhvhe"):
        srhvhe_dzi_name = "srhvhe.dzi"
        srhvhe_tiles_dir = "srhvhe_files"
    else:
        mosaic_rgb = load_mosaic_image(
            mosaic_dicom_path=mosaic_dicom_path,
            strip_padding=strip_padding,
        )
        srhvhe_image = Image.fromarray(mosaic_rgb)
        srhvhe_dzi_name, srhvhe_tiles_dir = save_image_as_dzi(
            image=srhvhe_image,
            output_prefix=dzi_path / "srhvhe",
            tile_size=tile_size,
            overlap=overlap,
            jpeg_quality=jpeg_quality,
        )

    outputs = {
        "dzi_dir": str(dzi_path),
        "srhvhe_dzi_path": str(dzi_path / srhvhe_dzi_name),
        "srhvhe_tiles_dir": str(dzi_path / srhvhe_tiles_dir),
    }

    if srhrgb_image_path is not None:
        if skip_existing and _dzi_asset_exists(dzi_path, "srhrgb"):
            srhrgb_dzi_name = "srhrgb.dzi"
            srhrgb_tiles_dir = "srhrgb_files"
        else:
            srhrgb_path = Path(srhrgb_image_path)
            assert (
                srhrgb_path.is_file()
            ), f"SRH RGB mosaic PNG not found: {srhrgb_path}"
            srhrgb_image = Image.open(srhrgb_path)
            srhrgb_dzi_name, srhrgb_tiles_dir = save_image_as_dzi(
                image=srhrgb_image,
                output_prefix=dzi_path / "srhrgb",
                tile_size=tile_size,
                overlap=overlap,
                jpeg_quality=jpeg_quality,
            )
        outputs.update(
            {
                "srhrgb_dzi_path": str(dzi_path / srhrgb_dzi_name),
                "srhrgb_tiles_dir": str(dzi_path / srhrgb_tiles_dir),
            }
        )

    logger.info("DZI assets for %s written under %s", slide_id, dzi_path)
    return outputs


def _dzi_artifact_paths(dzi_dir: str) -> dict[str, str]:
    return {
        "srhvhe_dzi": os.path.join(dzi_dir, "srhvhe.dzi"),
        "srhvhe_tiles": os.path.join(dzi_dir, "srhvhe_files"),
        "srhrgb_dzi": os.path.join(dzi_dir, "srhrgb.dzi"),
        "srhrgb_tiles": os.path.join(dzi_dir, "srhrgb_files"),
    }


def _dzi_asset_exists(dzi_dir: str | os.PathLike[str], prefix: str) -> bool:
    dzi_dir = str(dzi_dir)
    return os.path.isfile(os.path.join(dzi_dir, f"{prefix}.dzi")) and os.path.isdir(
        os.path.join(dzi_dir, f"{prefix}_files")
    )


def _dzi_asset_partially_exists(dzi_dir: str | os.PathLike[str], prefix: str) -> bool:
    dzi_dir = str(dzi_dir)
    descriptor_path = os.path.join(dzi_dir, f"{prefix}.dzi")
    tiles_dir = os.path.join(dzi_dir, f"{prefix}_files")
    return (os.path.exists(descriptor_path) or os.path.exists(tiles_dir)) and not (
        os.path.isfile(descriptor_path) and os.path.isdir(tiles_dir)
    )


def _dzi_outputs_exist(dzi_dir: str) -> bool:
    artifact_paths = _dzi_artifact_paths(dzi_dir)
    return (
        os.path.isfile(artifact_paths["srhvhe_dzi"])
        and os.path.isdir(artifact_paths["srhvhe_tiles"])
        and os.path.isfile(artifact_paths["srhrgb_dzi"])
        and os.path.isdir(artifact_paths["srhrgb_tiles"])
    )


def _dzi_outputs_partially_exist(dzi_dir: str) -> bool:
    return _dzi_asset_partially_exists(
        dzi_dir, "srhvhe"
    ) or _dzi_asset_partially_exists(dzi_dir, "srhrgb")


def _clear_existing_dzi_outputs(dzi_dir: str) -> None:
    artifact_paths = _dzi_artifact_paths(dzi_dir)
    for path in artifact_paths.values():
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)


def _resolve_dzi_output_dir(cf, mosaic_run: dict, viz_candidate_name: str) -> str:
    dzi_root = cf.infra.get("dzi_root")
    if dzi_root:
        return os.path.join(str(dzi_root), viz_candidate_name, "dzi")

    portal_out_root = cf.infra.get("portal_out_root")
    if portal_out_root:
        return mosaic_run.get(
            "dzi_dir",
            os.path.join(str(portal_out_root), viz_candidate_name, "portal", "dzi"),
        )

    assert cf.infra.out_root, "Expected infra.out_root when infra.dzi_root is not set."
    return mosaic_run.get(
        "dzi_dir",
        os.path.join(cf.infra.out_root, viz_candidate_name, "dzi"),
    )


def _resolve_srhrgb_image_path(cf, mosaic_run: dict, viz_candidate_name: str) -> str:
    static_output_dir = mosaic_run.get("static_output_dir")
    if static_output_dir is None:
        static_infra_out_root = cf.infra.get("static_infra_out_root")
        assert static_infra_out_root, (
            "Expected infra.static_infra_out_root or mosaic_run.static_output_dir "
            "to locate the SRH RGB mosaic PNG."
        )
        static_output_dir = os.path.join(str(static_infra_out_root), viz_candidate_name)

    return os.path.join(
        str(static_output_dir),
        f"{viz_candidate_name}-srhrgb.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the cell inference YAML config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate DZI outputs even when complete DZI artifacts already exist.",
    )
    return parser.parse_args()


def main(config_path: str, overwrite: bool = False) -> None:
    logging_format_str = (
        "[%(levelname)-s|%(asctime)s|%(name)s|"
        + "%(filename)s:%(lineno)d|%(funcName)s] %(message)s"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=logging_format_str,
    )
    assert config_path, "Expected a non-empty config path."
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    cf = OmegaConf.load(config_path)
    assert (
        cf.infra.get("dzi_root")
        or cf.infra.get("portal_out_root")
        or cf.infra.get("out_root")
    ), "Expected one of infra.dzi_root, infra.portal_out_root, or infra.out_root in config."

    selected_mosaic_runs = select_mosaic_runs_for_task(
        cf.data.mosaic_runs,
        cf.infra.get("num_mosaic_per_task"),
    )

    generated_slides = []
    skipped_slides = []
    failed_mosaic_runs = []

    for mosaic_run_entry in tqdm(
        selected_mosaic_runs,
        desc="Portal DZI export",
    ):
        mosaic_label = str(mosaic_run_entry)
        try:
            mosaic_run = parse_mosaic_run(mosaic_run_entry)
            mouse_id = mosaic_run["mouse_id"]
            mosaic_id = mosaic_run["mosaic_id"]
            viz_candidate_name = f"{mouse_id}-{mosaic_id}"
            mosaic_label = f"{mouse_id}/{mosaic_id}"
            dzi_dir = _resolve_dzi_output_dir(
                cf=cf,
                mosaic_run=mosaic_run,
                viz_candidate_name=viz_candidate_name,
            )
            srhrgb_image_path = _resolve_srhrgb_image_path(
                cf=cf,
                mosaic_run=mosaic_run,
                viz_candidate_name=viz_candidate_name,
            )

            if _dzi_outputs_exist(dzi_dir) and not overwrite:
                logger.info(
                    "[%s] Skipping DZI export because complete outputs already exist at %s",
                    viz_candidate_name,
                    dzi_dir,
                )
                skipped_slides.append(viz_candidate_name)
                continue

            if _dzi_outputs_partially_exist(dzi_dir):
                if not overwrite:
                    raise RuntimeError(
                        "Found partial DZI outputs under "
                        f"{dzi_dir}. Re-run with --overwrite to replace them."
                    )
                logger.info(
                    "[%s] Removing existing DZI artifacts under %s before regeneration",
                    viz_candidate_name,
                    dzi_dir,
                )
                _clear_existing_dzi_outputs(dzi_dir)

            inferred_paths = infer_mosaic_io_paths(
                mouse_id=mouse_id,
                mosaic_id=mosaic_id,
                data_root=cf.data.data_root,
            )
            export_slide_dzi_assets(
                slide_id=viz_candidate_name,
                mosaic_dicom_path=inferred_paths["mosaic_dicom_path"],
                dzi_dir=dzi_dir,
                srhrgb_image_path=srhrgb_image_path,
                skip_existing=not overwrite,
                strip_padding=50,
            )
            generated_slides.append(viz_candidate_name)
        except Exception as exc:
            logger.exception("Portal DZI export failed for %s", mosaic_label)
            failed_mosaic_runs.append(
                {
                    "mosaic_run": mosaic_label,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    logger.info(
        "Generated DZI outputs for %d mosaic runs: %s",
        len(generated_slides),
        ", ".join(generated_slides) if generated_slides else "(none)",
    )
    logger.info(
        "Skipped %d mosaic runs with existing DZI outputs: %s",
        len(skipped_slides),
        ", ".join(skipped_slides) if skipped_slides else "(none)",
    )
    if failed_mosaic_runs:
        failed_summary = "; ".join(
            f"{failure['mosaic_run']} ({failure['error_type']}: {failure['error']})"
            for failure in failed_mosaic_runs
        )
        raise RuntimeError(
            f"{len(failed_mosaic_runs)} of {len(selected_mosaic_runs)} mosaic runs "
            f"failed during portal DZI export: {failed_summary}"
        )


if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config, overwrite=args.overwrite)
