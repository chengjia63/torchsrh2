import pandas as pd

data = pd.read_csv(
    "/nfs/turbo/umms-tocho/data/chengjia/silica_databank/instances/srhum_all_instances_t48.csv",
    dtype=str,
)
ttype_cheng = pd.read_csv(
    "/nfs/turbo/umms-tocho/data/chengjia/silica_databank/srh_ttype_cheng.csv"
)


disallowed_df = pd.DataFrame(
    [("NIO_UM_937b", "4"), ("NIO_UM_950", "3")], columns=["patient", "mosaic"]
)
filtered = data.merge(
    disallowed_df, on=["patient", "mosaic"], how="left", indicator=True
)
filtered = filtered[filtered["_merge"] == "left_only"].drop(columns="_merge")
filtered = filtered[~filtered["patch"].str.contains("INV")]

filtered = filtered.rename({"patient": "nion"}, axis=1)
filtered["patient"] = filtered["nion"].str.extract("(NIO_UM_[0-9]+)")

filtered.merge(
    ttype_cheng, left_on="patient", right_on="type_institution_number", how="left"
)

filtered_with_label = filtered.merge(
    ttype_cheng, left_on="patient", right_on="type_institution_number", how="left"
).drop("type_institution_number", axis=1)
filtered_with_label["ttype_cheng"] = filtered_with_label["ttype_cheng"].fillna("other")
filtered = filtered_with_label[
    filtered_with_label["ttype_cheng"].isin(["glioma", "normal"])
]


sampled = (
    filtered[filtered["patch_type"].isin({"tumor", "normal"})]
    .groupby("patch_type")
    .sample(1000000, random_state=1000)
)
sampled["label"] = sampled["patch_type"]

output_columns = [
    "patch",
    "proposal",
    "patch_flat",
    "institution",
    "nion",
    "mosaic",
    "mmap_idx",
    "patch_type",
    "tensor_shape",
    "patient",
    "label",
    "ttype_cheng",
]
missing_columns = set(output_columns).difference(sampled.columns)
assert (
    not missing_columns
), f"Missing expected output columns: {sorted(missing_columns)}"

sampled[output_columns].to_csv("srhum_glioma_2m_.csv", index=False)
