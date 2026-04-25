import pandas as pd
import glob
import os

import pydicom
from tqdm import tqdm


def get_first_mosaic_dicom_path(slide_root: str) -> str:
    mosaics_dir_candidates = [
        os.path.join(slide_root, "mosaics"),
        os.path.join(slide_root, "mosaic"),
    ]
    mosaics_dir = next(
        (path for path in mosaics_dir_candidates if os.path.isdir(path)),
        None,
    )
    if mosaics_dir is None:
        raise FileNotFoundError(
            f"Could not find mosaic directory under {slide_root}. "
            "Expected either `mosaics/` or `mosaic/`."
        )

    dicom_paths = sorted(glob.glob(os.path.join(mosaics_dir, "*.dcm")))
    if not dicom_paths:
        raise FileNotFoundError(f"No DICOM files found under {mosaics_dir}")

    return dicom_paths[0]


def normalize_barcode(value) -> str | None:
    if pd.isna(value):
        return None

    value = str(value).strip()
    if not value:
        return None

    return value


def read_first_mosaic_dicom(slide_root: str):
    dicom_path = get_first_mosaic_dicom_path(slide_root)
    return pydicom.dcmread(dicom_path, stop_before_pixels=True)


def get_container_identifier(slide_root: str, container_identifier_tag: tuple) -> str:
    dicom = read_first_mosaic_dicom(slide_root)
    container_identifier = dicom.get(container_identifier_tag)
    if container_identifier is None:
        raise KeyError(
            f"Missing Container Identifier tag {container_identifier_tag} "
            f"in first mosaic DICOM under {slide_root}"
        )

    return normalize_barcode(container_identifier.value)


def load_ucsf_meta(
    ucsf_meta_path: str,
    ucsf_meta_column_map: dict,
    date_corrections: dict,
) -> pd.DataFrame:
    meta = pd.read_csv(ucsf_meta_path)
    meta.columns = meta.columns.str.strip()
    missing_columns = set(ucsf_meta_column_map) - set(meta.columns)
    if missing_columns:
        raise KeyError(
            f"Missing required UCSF metadata columns: {sorted(missing_columns)}"
        )

    meta = meta[list(ucsf_meta_column_map)].rename(columns=ucsf_meta_column_map)
    meta["ucsfnio"] = meta["ucsfnio"].ffill()
    meta["date"] = meta["date"].ffill()

    missing_ucsf_nio = meta["ucsfnio"].isna()
    if missing_ucsf_nio.any():
        raise ValueError(
            "Found missing `ucsfnio` values before any populated row in "
            f"{ucsf_meta_path}"
        )

    missing_surgery_date = meta["date"].isna()
    if missing_surgery_date.any():
        raise ValueError(
            "Found missing `date` values before any populated row in "
            f"{ucsf_meta_path}"
        )

    meta["ucsfnio"] = meta["ucsfnio"].astype(str).str.strip().str.removeprefix("NIO")
    meta["barcode"] = meta["barcode"].apply(normalize_barcode)
    for (ucsfnio, wrong_date), correct_date in date_corrections.items():
        meta.loc[
            (meta["ucsfnio"] == ucsfnio)
            & (meta["date"].astype(str).str.strip() == wrong_date),
            "date",
        ] = correct_date

    return meta


def load_data(
    data_filenames: list[str],
    slide_root: str,
    container_identifier_tag: tuple,
) -> pd.DataFrame:
    data = pd.concat(
        [pd.read_csv(filename) for filename in data_filenames],
        ignore_index=True,
    )
    data["slide_root"] = data.apply(
        lambda x: os.path.join(slide_root, str(x["patient"]), str(x["mosaic"])),
        axis=1,
    )
    data["container_identifier"] = data["slide_root"].progress_apply(
        lambda slide_root: get_container_identifier(
            slide_root,
            container_identifier_tag=container_identifier_tag,
        )
    )

    return data


def unique_list(series: pd.Series) -> list:
    return sorted(series.dropna().drop_duplicates().tolist())


def select_single_candidate(candidates: list):
    if len(candidates) == 1:
        return candidates[0]

    return "NONE"


def normalize_date(value, fallback_date: str) -> str:
    if pd.isna(value):
        return fallback_date

    return pd.Timestamp(value).strftime("%Y-%m-%d")


def build_patient_meta_map(
    data: pd.DataFrame,
    meta: pd.DataFrame,
    patient_ucsfnio_overrides: dict,
    fallback_date: str,
) -> pd.DataFrame:

    mapped = data.merge(
        meta,
        left_on="container_identifier",
        right_on="barcode",
        how="left",
    )
    mapped = mapped.loc[~mapped["patient"].isin(patient_ucsfnio_overrides)]

    manual_rows = []
    data_patients = set(data["patient"].dropna())
    for patient, ucsfnio_values in patient_ucsfnio_overrides.items():
        if patient not in data_patients:
            continue

        patient_meta = meta.loc[meta["ucsfnio"].isin(ucsfnio_values)]
        if patient_meta.empty:
            for ucsfnio in ucsfnio_values:
                manual_rows.append(
                    {
                        "patient": patient,
                        "container_identifier": None,
                        "ucsfnio": ucsfnio,
                        "date": None,
                        "barcode": None,
                    }
                )
            continue

        for row in patient_meta.itertuples(index=False):
            manual_rows.append(
                {
                    "patient": patient,
                    "container_identifier": row.barcode,
                    "ucsfnio": row.ucsfnio,
                    "date": row.date,
                    "barcode": row.barcode,
                }
            )

    if manual_rows:
        mapped = pd.concat((mapped, pd.DataFrame(manual_rows)), ignore_index=True)

    patient_meta = (
        mapped.groupby("patient", dropna=False)
        .agg(
            ucsfnio_candidates=("ucsfnio", unique_list),
            date_candidates=("date", unique_list),
        )
        .reset_index()
    )
    patient_meta["ucsfnio"] = patient_meta["ucsfnio_candidates"].apply(
        select_single_candidate
    )
    patient_meta["date"] = patient_meta["date_candidates"].apply(
        lambda candidates: (
            normalize_date(candidates[0], fallback_date)
            if len(candidates) == 1
            else fallback_date
        )
    )

    return patient_meta[["patient", "ucsfnio", "date"]].sort_values(
        ["date", "patient"],
        ignore_index=True,
    )


def expand_data_with_patient_meta(
    data: pd.DataFrame,
    patient_meta: pd.DataFrame,
) -> pd.DataFrame:
    return data.merge(patient_meta, on="patient", how="left").sort_values(
        ["date", "patient", "mosaic"], ignore_index=True
    )


def split_last_patient_groups_over_slide_count(
    data: pd.DataFrame,
    min_slide_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_counts = data.groupby("patient", sort=False).size()

    selected_patients = []
    total_slides = 0
    for patient, slide_count in reversed(list(patient_counts.items())):
        selected_patients.append(patient)
        total_slides += slide_count
        if total_slides > min_slide_count:
            break

    selected_patients = list(reversed(selected_patients))
    prospective = data.loc[data["patient"].isin(selected_patients)].reset_index(
        drop=True
    )
    retrospective = data.loc[~data["patient"].isin(selected_patients)].reset_index(
        drop=True
    )

    return retrospective, prospective


def get_split_group(row: pd.Series) -> str:
    if pd.notna(row["ucsfnio"]) and row["ucsfnio"] != "NONE":
        return str(row["ucsfnio"])

    return str(row["patient"])


def split_by_group(
    data: pd.DataFrame,
    train_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data["split_group"] = data.apply(get_split_group, axis=1)
    group_counts = data.groupby("split_group").size()
    target_train_slides = len(data) * train_fraction
    shuffled_groups = group_counts.sample(frac=1, random_state=random_state)

    train_groups = []
    train_slides = 0
    for group, slide_count in shuffled_groups.items():
        train_groups.append(group)
        train_slides += slide_count
        if train_slides >= target_train_slides:
            break

    train = data.loc[data["split_group"].isin(train_groups)].drop(
        columns=["split_group"]
    )
    val = data.loc[~data["split_group"].isin(train_groups)].drop(
        columns=["split_group"]
    )

    return train.reset_index(drop=True), val.reset_index(drop=True)


def main():
    date_corrections = {
        ("2", "3/8/3022"): "3/8/2022",
        ("187", "12/13/122"): "12/13/2022",
        ("280", "6/13/123"): "6/13/2023",
    }
    patient_ucsfnio_overrides = {
        "NIO_UCSF_275": ["275"],
        "NIO_UCSF_326": ["326"],
        "NIO_UCSF_359": ["359"],
        "NIO_UCSF_367": ["367"],
        "NIO_UCSF_368": ["368"],
        "NIO_UCSF_369": ["369"],
        "NIO_UCSF_374": ["374"],
        "NIO_UCSF_384": ["384"],
        "NIO_UCSF_385": ["385"],
        "NIO_UCSF_405": ["405"],
        "NIO_UCSF_422": ["422"],
        "NIO_UCSF_56199": ["56"],
        "NIO_UCSF_577": ["577"],
    }

    ucsf_meta = load_ucsf_meta(
        "/nfs/turbo/umms-tocho/data/chengjia/silica_fg_data_splits/raw/ucsf_meta.csv",
        ucsf_meta_column_map={
            "UCSFNIO #": "ucsfnio",
            "Date of surgery": "date",
            "Barcode": "barcode",
        },
        date_corrections=date_corrections,
    )

    data = load_data(
        data_filenames=[
            "/nfs/turbo/umms-tocho/data/chengjia/silica_fg_data_splits/120425_split/db_srhdg_ucsf_train_fold0.csv",
            "/nfs/turbo/umms-tocho/data/chengjia/silica_fg_data_splits/120425_split/db_srhdg_ucsf_val_fold0.csv",
            "/nfs/turbo/umms-tocho/data/chengjia/silica_fg_data_splits/120425_split/db_srhdg_ucsf_prosp_fold0.csv",
        ],
        slide_root="/nfs/turbo/umms-tocho/root_srh_db/UCSF",
        container_identifier_tag=(0x0040, 0x0512),
    )

    patient_meta = build_patient_meta_map(
        data,
        ucsf_meta,
        patient_ucsfnio_overrides,
        fallback_date="1970-01-01",
    )
    patient_meta = expand_data_with_patient_meta(data, patient_meta)
    patient_meta.to_csv("ucsf_all_sorted.csv", index=False)
    retrospective, prospective = split_last_patient_groups_over_slide_count(
        patient_meta,
        min_slide_count=500,
    )
    retrospective_train, retrospective_val = split_by_group(
        retrospective,
        train_fraction=0.8,
        random_state=0,
    )
    retrospective_train.to_csv("ucsf_retrospective_train.csv", index=False)
    retrospective_val.to_csv("ucsf_retrospective_val.csv", index=False)
    prospective.to_csv("ucsf_prospective.csv", index=False)


if __name__ == "__main__":
    tqdm.pandas()
    main()
