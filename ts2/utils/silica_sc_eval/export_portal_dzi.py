import argparse
import logging
import os
import shutil

from omegaconf import OmegaConf
from tqdm.auto import tqdm

from ts2.utils.silica_sc_eval.portal_assets import export_slide_dzi_assets
from ts2.utils.silica_sc_eval.run_single_cell_inference import (
    infer_mosaic_io_paths,
    parse_mosaic_run,
    resolve_mosaic_output_dirs,
    select_mosaic_runs_for_task,
)

logger = logging.getLogger(__name__)


def _dzi_artifact_paths(dzi_dir: str) -> dict[str, str]:
    return {
        "color_dzi": os.path.join(dzi_dir, "color.dzi"),
        "color_tiles": os.path.join(dzi_dir, "color_files"),
    }


def _dzi_outputs_exist(dzi_dir: str) -> bool:
    artifact_paths = _dzi_artifact_paths(dzi_dir)
    return (
        os.path.isfile(artifact_paths["color_dzi"])
        and os.path.isdir(artifact_paths["color_tiles"])
    )


def _dzi_outputs_partially_exist(dzi_dir: str) -> bool:
    artifact_paths = _dzi_artifact_paths(dzi_dir)
    return any(os.path.exists(path) for path in artifact_paths.values())


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

    assert cf.infra.out_root, "Expected infra.out_root when infra.dzi_root is not set."
    assert (
        cf.infra.static_infra_out_root
    ), "Expected infra.static_infra_out_root when infra.dzi_root is not set."
    output_dirs = resolve_mosaic_output_dirs(
        mosaic_run=mosaic_run,
        out_root=cf.infra.out_root,
        static_infra_out_root=cf.infra.static_infra_out_root,
        viz_candidate_name=viz_candidate_name,
    )
    return output_dirs["dzi_dir"]


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
    assert cf.infra.get("dzi_root") or cf.infra.get(
        "out_root"
    ), "Expected either infra.dzi_root or infra.out_root in config."

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
