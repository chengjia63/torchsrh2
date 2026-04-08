import argparse
import logging

# import openslide
import matplotlib.pyplot as plt
import numpy as np
import glob
import json
import pandas as pd
from tqdm import tqdm
from typing import List
import yaml
import os
from os.path import join as opj
import re

# from matplotlib_venn import venn2, venn3
from datetime import datetime, timedelta
import warnings
import matplotlib


def sanitize_string(string):
    return re.sub(r"[^a-zA-Z0-9]", "", string)


def read_cat_csvs_with_fn(fns: List[str]) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(mfn, dtype=str, comment="#") for mfn in fns]
    ).reset_index(drop=True)


def read_cat_csvs_with_fn_keep_fn(fns: List[str]) -> pd.DataFrame:
    all_dfs = []
    for mfn in fns:
        curr_df = pd.read_csv(mfn, dtype=str, comment="#")
        curr_df["qr_meta_fn"] = mfn.split("/")[-1]
        all_dfs.append(curr_df)
    return pd.concat(all_dfs).reset_index(drop=True)


def read_cat_csvs(fn_pattern: str) -> pd.DataFrame:
    meta_fns = glob.glob(fn_pattern)
    return pd.concat(
        [pd.read_csv(mfn, dtype=str, comment="#") for mfn in meta_fns]
    ).reset_index(drop=True)


def mxa_code_cast(x):
    if x == "":
        return ""
    return f"{int(x):02d}"


def process_metadata_mx_match(cf):
    metas = read_cat_csvs_with_fn(cf["mx_match"]["labportal"])
    metas = metas[~metas["Outside Accession"].isna()].copy()
    metas["m_num"] = metas["Outside Accession"].apply(
        lambda x: int(x.lower().split("x")[0].removeprefix("m"))
    )
    metas["x_num"] = metas["Outside Accession"].apply(
        lambda x: int(x.lower().split("x")[1])
    )

    ocr_metas = read_cat_csvs_with_fn_keep_fn(cf["mx_match"]["ocr"]).copy()
    ocr_metas_mxonly = ocr_metas[["m_code", "x_code"]]
    assert not ocr_metas_mxonly.duplicated().any()

    ocr_metas = ocr_metas.rename(
        columns={
            "Unnamed: 4": "tile",
            "Unnamed: 5": "comment",
        }
    )
    ocr_metas = ocr_metas.drop("mx_code", axis=1)
    ocr_metas["m_num"] = ocr_metas["m_code"].apply(
        lambda x: int(x.lower().removeprefix("m"))
    )
    ocr_metas["x_num"] = ocr_metas["x_code"].apply(
        lambda x: int(x.lower().removeprefix("x"))
    )

    out = pd.merge(
        left=ocr_metas,
        right=metas,
        left_on=["m_num", "x_num"],
        right_on=["m_num", "x_num"],
        how="outer",
    )

    out["a_num"] = -1
    out["mxa_code"] = out.fillna("").apply(
        lambda x: f'M{x["m_num"]}x{x["x_num"]:02d}', axis=1
    )
    out = out.drop(["m_code", "x_code"], axis=1)
    out = out[cf["infra"]["meta_col_order"]].sort_values(["m_num", "x_num"])
    return out


def process_metadata_mxa_match(cf):
    metas = read_cat_csvs_with_fn(cf["mxa_match"]["labportal"]).copy()
    metas["a_num"] = (
        metas["A_number"]
        .fillna("a-1")
        .apply(lambda x: int(x.lower().replace("missing", "a-2").removeprefix("a")))
    )
    metas["m_num"] = metas["Outside Accession"].apply(
        lambda x: int(x.lower().split("x")[0].removeprefix("m"))
    )
    metas["x_num"] = metas["Outside Accession"].apply(
        lambda x: int(x.lower().split("x")[1])
    )

    metas_amatch = metas[metas["a_num"] != -1].copy()
    metas_mxmatch = metas[metas["a_num"] == -1].drop("a_num", axis=1).copy()

    qrd_csv = read_cat_csvs_with_fn_keep_fn(cf["mxa_match"]["ocr"]).copy()
    qrd_csv["a_num"] = qrd_csv["a_code"].apply(
        lambda x: int(x.lower().replace("x", "a-1").removeprefix("a"))
    )
    qrd_csv_mxmatch = qrd_csv[qrd_csv["a_num"] == -1].copy()
    qrd_csv_amatch = qrd_csv[qrd_csv["a_num"] != -1].copy()

    qrd_csv_mxmatch["m_num"] = qrd_csv_mxmatch["m_code"].apply(
        lambda x: int(x.lower().removeprefix("m"))
    )
    qrd_csv_mxmatch["x_num"] = qrd_csv_mxmatch["x_code"].apply(
        lambda x: int(x.lower().removeprefix("x"))
    )

    mxmatch_out = pd.merge(
        left=qrd_csv_mxmatch,
        right=metas_mxmatch,
        left_on=["m_num", "x_num"],
        right_on=["m_num", "x_num"],
        how="outer",
    )
    mxmatch_out["mxa_code"] = mxmatch_out.fillna("").apply(
        lambda x: f'M{x["m_num"]}x{x["x_num"]:02d}', axis=1
    )
    mxmatch_out["tile"] = ""
    mxmatch_out["comment"] = ""
    mxmatch_out = mxmatch_out.drop(["a_code", "m_code", "x_code", "A_number"], axis=1)
    mxmatch_out = mxmatch_out[cf["infra"]["meta_col_order"]].sort_values(
        ["m_num", "x_num"]
    )

    amatch_out = pd.merge(
        left=qrd_csv_amatch,
        right=metas_amatch,
        left_on=["a_num"],
        right_on=["a_num"],
        how="outer",
    )
    amatch_out["mxa_code"] = amatch_out.fillna("").apply(
        lambda x: (
            f'M{mxa_code_cast(x["m_num"])}'
            f'x{mxa_code_cast(x["x_num"])}'
            f'A{x["a_num"]}'
        ),
        axis=1,
    )
    amatch_out["tile"] = ""
    amatch_out["comment"] = ""
    amatch_out["x_num"] = amatch_out["x_num"].fillna(-2).astype(int)
    amatch_out["m_num"] = amatch_out["m_num"].fillna(-2).astype(int)
    amatch_out["a_num"] = amatch_out["a_num"].fillna(-2).astype(int)
    amatch_out = amatch_out.drop(["a_code", "m_code", "x_code", "A_number"], axis=1)
    amatch_out = amatch_out[cf["infra"]["meta_col_order"]].sort_values(
        ["m_num", "x_num", "a_num"]
    )

    return mxmatch_out, amatch_out


def process_metadata_a_match(cf):
    metas_nn = read_cat_csvs_with_fn(cf["a_match"]["labportal"]).copy()
    metas_nn["a_num"] = metas_nn["Outside Accession"].apply(
        lambda x: int(x.lower().removeprefix("a"))
    )
    metas_nn["m_num"] = -2
    metas_nn["x_num"] = -2

    qrd_nn = read_cat_csvs_with_fn_keep_fn(cf["a_match"]["ocr"]).copy()
    qrd_nn["a_num"] = qrd_nn["a_code"].apply(
        lambda x: int(x.lower().replace("x", "a-1").removeprefix("a"))
    )

    out = pd.merge(
        left=qrd_nn,
        right=metas_nn,
        left_on=["a_num"],
        right_on=["a_num"],
        how="outer",
    )
    out["mxa_code"] = out.fillna("").apply(lambda x: f'MxA{x["a_num"]}', axis=1)
    out["tile"] = ""
    out["comment"] = ""
    out["x_num"] = -1
    out["m_num"] = -1
    out["a_num"] = out["a_num"].fillna(-1).astype(int)
    out = out.drop(["a_code", "m_code", "x_code", "A_number"], axis=1)
    out = out[cf["infra"]["meta_col_order"]].sort_values(["m_num", "x_num", "a_num"])
    return out


def get_filename(path):
    if not isinstance(path, str) or not path:
        return ""
    return re.split(r"[\\/]", path)[-1]


def get_filename_stem(path):
    return os.path.splitext(get_filename(path))[0]


def path_relative_to(path, root):
    path_abs = os.path.abspath(path)
    root_abs = os.path.abspath(root)
    if os.path.commonpath([path_abs, root_abs]) != root_abs:
        return path
    return os.path.relpath(path_abs, root_abs)


def list_phase_c_svs_files(svs_dirs, svs_root):
    rows = []
    for svs_dir in svs_dirs:
        if not os.path.isdir(svs_dir):
            raise NotADirectoryError(f"Phase C SVS directory does not exist: {svs_dir}")

        for fn in os.listdir(svs_dir):
            svs_path = opj(svs_dir, fn)
            if os.path.isfile(svs_path) and fn.lower().endswith(".svs"):
                rows.append(
                    {
                        "phase_c_filename": fn,
                        "path": path_relative_to(svs_path, svs_root),
                    }
                )

    svs_files = pd.DataFrame(rows)
    if svs_files.empty:
        return pd.DataFrame(columns=["phase_c_filename", "path"])

    duplicated_fns = svs_files.loc[
        svs_files["phase_c_filename"].duplicated(), "phase_c_filename"
    ]
    if len(duplicated_fns):
        raise ValueError(
            "Phase C SVS filenames must be unique. Duplicates: "
            f"{sorted(duplicated_fns.unique().tolist())}"
        )

    return svs_files


def process_metadata_phase_c(cf):
    phase_c_cf = cf.get("phase_c")
    if not phase_c_cf or (
        not phase_c_cf.get("metadata") and not phase_c_cf.get("svs_dirs")
    ):
        return pd.DataFrame(columns=cf["infra"]["meta_col_order"])
    if not phase_c_cf.get("metadata") or not phase_c_cf.get("svs_dirs"):
        raise ValueError("Phase C config must set both metadata and svs_dirs")

    metas = read_cat_csvs_with_fn_keep_fn(phase_c_cf["metadata"]).copy()
    if "File Location" not in metas.columns:
        raise KeyError('Phase C metadata must include column "File Location"')

    metas["phase_c_filename"] = metas["File Location"].apply(get_filename)
    empty_file_location = metas["phase_c_filename"] == ""
    if empty_file_location.any():
        raise ValueError(
            "Phase C metadata has empty File Location values in rows: "
            f"{metas.index[empty_file_location].tolist()}"
        )

    duplicated_meta_fns = metas.loc[
        metas["phase_c_filename"].duplicated(), "phase_c_filename"
    ]
    if len(duplicated_meta_fns):
        raise ValueError(
            "Phase C metadata filenames must be unique. Duplicates: "
            f"{sorted(duplicated_meta_fns.unique().tolist())}"
        )

    metas = metas.rename(
        columns={
            "Barcode ID": "Barcode",
            "Accession / ID #": "UM Accession",
            "Outside Acc#": "Outside Accession",
            "Block ID": "Block",
            "Organ": "Site/Organ",
        }
    )

    svs_root = opj(cf["infra"]["neuroslides_root"], "svs")
    svs_files = list_phase_c_svs_files(phase_c_cf["svs_dirs"], svs_root)
    out = pd.merge(
        left=metas,
        right=svs_files,
        on="phase_c_filename",
        how="outer",
    )

    out["mxa_code"] = out["phase_c_filename"].apply(get_filename_stem)
    out["m_num"] = 0
    out["x_num"] = 0
    out["a_num"] = 0
    out["tile"] = ""
    out["comment"] = ""
    out["Diagnosis"] = ""
    out["original_path"] = out["File Location"]

    return out[cf["infra"]["meta_col_order"] + ["original_path"]].sort_values(
        "mxa_code"
    )


def normalize_metadata_paths(df):
    def proc_path_str(x):
        if isinstance(x, str):
            return (
                x.removeprefix("/nfs/turbo/umms-tocho/data/he.mlins_scan.svs.raw/")
                .removeprefix("/nfs/mm-isilon/brainscans/dropbox/Slide_Incoming/")
                .removeprefix("/nfs/umms-tocho-mr/dropbox/Slide_Incoming/")
                .replace("20241108", "2024-11-08")
                .replace("20241109", "2024-11-08")
                .replace("20241110", "2024-11-08")
                .replace("2024-11-22 evening", "2024-11-22e")
                .replace("2024-11-23 evening", "2024-11-23e")
                .replace("2024-11-24 evening", "2024-11-24e")
                .replace("2024-11-26 evening", "2024-11-26e")
                .replace("2024-12-03 evening", "2024-12-03e")
            )
        return x

    df = df.copy()
    df["path"] = df["path"].apply(proc_path_str)
    return df


def log_phase_c_stats(phase_c_allmeta):
    log_section("Phase C stats")
    logging.info("Rows: %d", len(phase_c_allmeta))
    logging.info(
        "Unique UM Accession values: %d",
        len(phase_c_allmeta["UM Accession"].drop_duplicates()),
    )
    logging.info(
        "Rows with UM Accession: %d",
        phase_c_allmeta["UM Accession"].notna().sum(),
    )
    logging.info("Rows with scanned path: %d", phase_c_allmeta["path"].notna().sum())
    logging.info("Rows missing scanned path: %d", phase_c_allmeta["path"].isna().sum())


def process_metadata(cf):
    first_batch_allmeta = process_metadata_mx_match(cf)
    mxa_mxmatch_allmeta, mxa_amatch_allmeta = process_metadata_mxa_match(cf)
    amatch_allmeta = process_metadata_a_match(cf)
    phase_c_allmeta = process_metadata_phase_c(cf)
    log_phase_c_stats(phase_c_allmeta)

    all_meta = pd.concat(
        [
            first_batch_allmeta,
            mxa_mxmatch_allmeta,
            mxa_amatch_allmeta,
            amatch_allmeta,
            phase_c_allmeta,
        ]
    ).sort_values(["a_num", "m_num", "x_num"])
    if "original_path" not in all_meta.columns:
        all_meta["original_path"] = ""

    all_meta = normalize_metadata_paths(all_meta)
    return all_meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the matched SU/MxA metadata spreadsheet."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def setup_logging(log_level):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with open(config_path, "r") as f:
        cf = yaml.safe_load(f)

    if cf is None:
        raise ValueError(f"Config file is empty: {config_path}")
    normalize_config_paths(cf)
    return cf


def join_neuroslides_path(neuroslides_root, path):
    if os.path.isabs(path):
        return path
    return opj(neuroslides_root, path)


def normalize_config_paths(cf):
    neuroslides_root = cf["infra"].get("neuroslides_root")
    if not neuroslides_root:
        raise ValueError("Config must set infra.neuroslides_root")

    for section_name in ["mx_match", "mxa_match", "a_match"]:
        for key in ["labportal", "ocr"]:
            cf[section_name][key] = [
                join_neuroslides_path(neuroslides_root, path)
                for path in cf[section_name][key]
            ]

    if "phase_c" in cf:
        cf["phase_c"]["metadata"] = [
            join_neuroslides_path(neuroslides_root, path)
            for path in cf["phase_c"]["metadata"]
        ]
        cf["phase_c"]["svs_dirs"] = [
            join_neuroslides_path(neuroslides_root, path)
            for path in cf["phase_c"]["svs_dirs"]
        ]


def format_table(df):
    with pd.option_context("display.max_colwidth", None):
        return df.to_string(index=False)


def resolve_svs_path(svs_root, path):
    if os.path.isabs(path):
        return path
    return opj(svs_root, path)


def log_section(title):
    logging.info("")
    logging.info("=" * 88)
    logging.info(title)
    logging.info("=" * 88)


def path_parent_dir(path):
    if not isinstance(path, str) or not path:
        return ""
    return os.path.dirname(path)


def log_check(title, df, note=None, sort_by=None, directory_column=None):
    logging.info("")
    logging.info("- %s", title)
    if note:
        logging.info("  Note: %s", note)

    out_df = df
    if sort_by is not None:
        out_df = out_df.sort_values(sort_by, na_position="last")

    if len(out_df):
        logging.error("  Rows: %d", len(out_df))
        print(format_table(out_df))

        if directory_column is not None:
            directories = sorted(
                {
                    directory
                    for directory in out_df[directory_column].apply(path_parent_dir)
                    if directory
                }
            )
            logging.error("  Directories containing these files:")
            print("\n".join(directories))

    else:
        logging.info("  OK")


def main():
    args = parse_args()
    setup_logging(args.log_level)

    logging.info("Loading config: %s", args.config)
    cf = load_config(args.config)
    all_meta = process_metadata(cf)
    logging.info("Processed metadata rows: %d", len(all_meta))

    fsx_annotation = pd.read_csv(cf["fsx_annot"])
    mxa_fsx = set(
        fsx_annotation.loc[fsx_annotation["x_p"].isin({"x", "tp"}), "mxa"].tolist()
    )
    all_meta.loc[all_meta["mxa_code"].isin(mxa_fsx), "Block"] = (
        all_meta.loc[all_meta["mxa_code"].isin(mxa_fsx), "Block"].str.removesuffix("x")
        + "x"
    )

    export_cols = [col for col in cf["infra"]["meta_col_order"] if col != "qr_meta_fn"]
    all_meta[export_cols].to_csv(f'{cf["infra"]["out_fn"]}.csv', index=False)
    logging.info("Wrote CSV: %s.csv", cf["infra"]["out_fn"])
    all_meta[export_cols].to_excel(f'{cf["infra"]["out_fn"]}.xlsx', index=False)
    logging.info("Wrote Excel: %s.xlsx", cf["infra"]["out_fn"])

    pd.set_option("display.max_rows", None)

    log_section("Sanity checks - these tables should all be empty")

    kns = all_meta[all_meta["path"].isna()]
    log_check(
        "Known but not scanned - need to rescan",
        kns[["original_path"]],
        sort_by="original_path",
    )

    snk = all_meta[all_meta["UM Accession"].isna()]
    log_check(
        "Scanned but not known - need to upload / check LabPortal CSV",
        snk,
        sort_by="path",
        directory_column="path",
    )

    snkb = all_meta[all_meta["Block"].isna()]
    log_check("Missing block", snkb)

    check_select_stain = all_meta[all_meta["Stain"] == "Select Stain"]
    log_check("Select Stain values", check_select_stain)

    ms_mx = (
        all_meta[(all_meta["m_num"] > 0) & (all_meta["x_num"] > 0)]
        .groupby(["m_num", "x_num"])
        .filter(lambda x: len(x) > 1)
    )
    log_check(
        "Multiple scans by Mx numbers - need to decide which to keep",
        ms_mx,
        note="Includes the ones with A numbers. Comment them out in the OCR results.",
    )

    ms_a = (
        all_meta[(all_meta["m_num"] == -1) & (all_meta["x_num"] == -1)]
        .groupby(["a_num"])
        .filter(lambda x: len(x) > 1)
    )
    log_check("Multiple scans by A numbers only - need to decide which to keep", ms_a)

    m_msu = (
        all_meta.loc[all_meta["m_num"] > 0, ["m_num", "UM Accession"]]
        .drop_duplicates()
        .groupby("m_num")
        .filter(lambda x: len(x) > 1)
    )
    log_check("M maps to multiple SU - need to give them a new M number", m_msu)

    su_mm = (
        all_meta.loc[all_meta["m_num"] > 0, ["m_num", "UM Accession"]]
        .drop_duplicates()
        .groupby("UM Accession")
        .filter(lambda x: len(x) > 1)
    )
    log_check(
        "SU maps to multiple M",
        su_mm,
        note=(
            "Use the earlier M number, change the labportal csv, and note the "
            "original Mx number as well. This is currently OK because M number "
            "is not used anymore."
        ),
    )

    log_section("File existence check")
    svs_root = opj(cf["infra"]["neuroslides_root"], "svs")
    missing_svs = []
    missing_path_meta = []
    for i, s in tqdm(all_meta.iterrows(), total=len(all_meta)):
        if type(s["path"]) == str:
            spath = resolve_svs_path(svs_root, s["path"])
            if not os.path.exists(spath):
                missing_svs.append(spath)
        else:
            missing_path_meta.append(s["mxa_code"])

    if missing_path_meta:
        logging.warning(
            "Skipped rows missing path metadata: %d", len(missing_path_meta)
        )
        logging.warning("\n%s", "\n".join(missing_path_meta))

    if missing_svs:
        logging.error("Missing SVS files: %d", len(missing_svs))
        logging.error("\n%s", "\n".join(missing_svs))
        raise FileNotFoundError(f"Missing {len(missing_svs)} SVS files")

    logging.info("All referenced SVS files exist")


if __name__ == "__main__":
    main()
