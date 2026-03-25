import json
import logging
import os
from os.path import join as opj

import pandas as pd
from tqdm import tqdm

from ts2.data.meta_parser import CachedCSVParser
from ts2.data.slide_dataset import SRHPatchCoordMapper


def build_all_instances(
    data_root: str,
    cached_parser_dir: str,
    sc_proposal_root: str,
    seg_model: str = "03207B00",
    tile_size: int = 48,
) -> pd.DataFrame:
    slide_instances, tensor_shape_map = CachedCSVParser(cache_dir=cached_parser_dir)()
    instances = []

    for inst in tqdm(slide_instances, desc="Building cell instances"):
        slide_name = inst["name"]
        patient, mosaic = slide_name.rsplit("-", 1)
        mmap_info = tensor_shape_map[slide_name]

        path_parts = mmap_info["path"].split("/")
        assert len(path_parts) >= 4, f"Unexpected mmap path format: {mmap_info['path']}"
        institution = path_parts[0]

        pt_meta_fname = opj(data_root, institution, patient, f"{patient}_meta.json")
        if not os.path.exists(pt_meta_fname):
            logging.warning("No pt meta file - %s", pt_meta_fname)
            continue

        with open(pt_meta_fname) as fd:
            slide_meta_all_inf = json.load(fd)["slides"][mosaic]["predictions"]

        if seg_model in slide_meta_all_inf:
            slide_meta = slide_meta_all_inf[seg_model]
        else:
            logging.warning("No %s inference - %s", seg_model, pt_meta_fname)
            continue

        patch_type_per_patch = {}
        for patch_type, subdict in slide_meta["patches"].items():
            for patch_flat in subdict:
                patch_name = patch_flat.removeprefix(f"{slide_name}-")
                patch_type_per_patch[patch_name] = patch_type

        slide_meta = opj(sc_proposal_root, f"{slide_name}-meta.csv")
        assert os.path.exists(slide_meta), f"Missing proposal file: {slide_meta}"

        try:
            cp = pd.read_csv(slide_meta)
        except pd.errors.EmptyDataError:
            logging.warning("No cells before filtering - %s", slide_name)
            continue

        cp["centroid_r"] = cp["centroid_r"].round().astype(int)
        cp["centroid_c"] = cp["centroid_c"].round().astype(int)

        cp_filt = cp[
            cp["celltype"].isin({"nuclei", "mp"})
            & (cp["score"] > 0.5)
            & (tile_size / 2 <= cp["centroid_r"])
            & (cp["centroid_r"] <= 300 - tile_size / 2)
            & (tile_size / 2 <= cp["centroid_c"])
            & (cp["centroid_c"] <= 300 - tile_size / 2)
        ]

        if len(cp_filt) == 0:
            logging.warning("No cells after filtering - %s", slide_name)
            continue

        cp_filt = pd.DataFrame(
            {
                "patch": cp_filt["patch"],
                "proposal": list(zip(cp_filt["centroid_r"], cp_filt["centroid_c"])),
            }
        )
        cp_filt.loc[:, "patch_flat"] = cp_filt["patch"].apply(
            SRHPatchCoordMapper.to_universal_patch_name
        )
        cp_filt.loc[:, "patch_name"] = cp_filt["patch_flat"].apply(
            lambda x: "-".join(x.split("-")[-2:])
        )

        patch_summary = cp_filt.groupby("patch_name").agg(
            proposal=("proposal", list),
            patch=("patch", lambda values: values.iloc[0]),
            patch_flat=("patch_flat", lambda values: values.iloc[0]),
        )
        patch_keep = set(patch_summary.index)

        kept_patches = [
            patch for patch in inst["patches"] if patch["patch_name"] in patch_keep
        ]

        records = []
        for patch in kept_patches:
            patch_name = patch["patch_name"]
            patch_info = patch_summary.loc[patch_name]
            patch_type = patch_type_per_patch.get(patch_name)
            assert (
                patch_type is not None
            ), f"Missing patch_type for {slide_name} patch {patch_name}"

            for proposal in patch_info["proposal"]:
                records.append(
                    {
                        "patch": patch_info["patch"],
                        "proposal": proposal,
                        "patch_flat": patch_info["patch_flat"],
                        "institution": institution,
                        "patient": patient,
                        "mosaic": mosaic,
                        "mmap_idx": patch["patch_idx"],
                        "patch_type": patch_type,
                        "tensor_shape": tuple(mmap_info["shape"]),
                    }
                )

        if records:
            instances.append(pd.DataFrame.from_records(records))

    assert instances, "No cell instances were built."
    out_df = pd.concat(instances).reset_index(drop=True)
    missing_columns = {
        "patch",
        "proposal",
        "patch_flat",
        "institution",
        "patient",
        "mosaic",
        "mmap_idx",
        "patch_type",
        "tensor_shape",
    }.difference(out_df.columns)
    assert not missing_columns, f"Missing expected columns: {sorted(missing_columns)}"
    return out_df[
        [
            "patch",
            "proposal",
            "patch_flat",
            "institution",
            "patient",
            "mosaic",
            "mmap_idx",
            "patch_type",
            "tensor_shape",
        ]
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    data_root = "/nfs/turbo/umms-tocho-snr/data/root_histology_db/srh"
    sc_proposal_root = "/nfs/turbo/umms-tocho-snr/exp/chengjia/scsrh_repl_root_gen2"
    #cached_parser_dir = "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/data/tsmeta/srh7v1/7b5f27a8_slide_train"
    cached_parser_dir = "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/data/tsmeta/srh7v1_diagnostic/962e2262_slide_test"
    #cached_parser_dir = "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/data/tsmeta/srhum/bf99b89c_slide_all"
    output_csv = "./test_srh7v1_diagnostic_test_all_instances.csv"
    tile_size = 64

    out_df = build_all_instances(
        data_root=data_root,
        sc_proposal_root=sc_proposal_root,
        cached_parser_dir=cached_parser_dir,
        tile_size=tile_size,
    )
    out_df.to_csv(output_csv, index=False)
    assert os.path.exists(output_csv), f"Expected output CSV at {output_csv}"
    logging.info("Wrote %d rows to %s", len(out_df), output_csv)


if __name__ == "__main__":
    main()
