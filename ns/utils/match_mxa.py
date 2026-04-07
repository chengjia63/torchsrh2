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


def process_metadata(cf):
    first_batch_all_meta = process_metadata_mx_match(cf)
    mxa_batch_mxmatch_allmeta, mxa_batch_mamatch_allmeta = process_metadata_mxa_match(
        cf
    )
    mxa_newnormal_allmeta = process_metadata_a_match(cf)

    all_meta = pd.concat(
        [
            first_batch_all_meta,
            mxa_batch_mxmatch_allmeta,
            mxa_batch_mamatch_allmeta,
            mxa_newnormal_allmeta,
        ]
    ).sort_values(["a_num", "m_num", "x_num"])

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


def format_table(df):
    return df.to_string(index=False)


def log_section(title):
    logging.info("")
    logging.info("=" * 88)
    logging.info(title)
    logging.info("=" * 88)


def log_check(title, df, note=None):
    logging.info("")
    logging.info("- %s", title)
    if note:
        logging.info("  Note: %s", note)
    logging.info("  Rows: %d", len(df))
    if len(df):
        logging.warning("\n%s", format_table(df))
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

    all_meta.drop("qr_meta_fn", axis=1).to_csv(
        f'{cf["infra"]["out_fn"]}.csv', index=False
    )
    logging.info("Wrote CSV: %s.csv", cf["infra"]["out_fn"])
    all_meta.drop("qr_meta_fn", axis=1).to_excel(
        f'{cf["infra"]["out_fn"]}.xlsx', index=False
    )
    logging.info("Wrote Excel: %s.xlsx", cf["infra"]["out_fn"])

    pd.set_option("display.max_rows", None)

    log_section("Sanity checks - these tables should all be empty")

    kns = all_meta[all_meta["path"].isna()]
    log_check("Known but not scanned - need to rescan", kns)

    snk = all_meta[all_meta["UM Accession"].isna()]
    log_check("Scanned but not known - need to upload / check LabPortal CSV", snk)

    snkb = all_meta[all_meta["Block"].isna()]
    log_check("Missing block", snkb)

    check_select_stain = all_meta[all_meta["Stain"] == "Select Stain"]
    log_check("Select Stain values", check_select_stain)

    ms_mx = (
        all_meta[(all_meta["m_num"] != -1) & (all_meta["x_num"] != -1)]
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
        all_meta.loc[all_meta["m_num"] != -1, ["m_num", "UM Accession"]]
        .drop_duplicates()
        .groupby("m_num")
        .filter(lambda x: len(x) > 1)
    )
    log_check("M maps to multiple SU - need to give them a new M number", m_msu)

    su_mm = (
        all_meta.loc[all_meta["m_num"] != -1, ["m_num", "UM Accession"]]
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
            spath = opj(svs_root, s["path"])
            if not os.path.exists(spath):
                missing_svs.append(spath)
        else:
            missing_path_meta.append(s["mxa_code"])

    if missing_path_meta:
        logging.warning("Skipped rows missing path metadata: %d", len(missing_path_meta))
        logging.warning("\n%s", "\n".join(missing_path_meta))

    if missing_svs:
        logging.error("Missing SVS files: %d", len(missing_svs))
        logging.error("\n%s", "\n".join(missing_svs))
        raise FileNotFoundError(f"Missing {len(missing_svs)} SVS files")

    logging.info("All referenced SVS files exist")
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
