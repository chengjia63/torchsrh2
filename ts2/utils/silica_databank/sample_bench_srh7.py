import pandas as pd

data = pd.read_csv(
    "/nfs/turbo/umms-tocho/data/data_splits/chengjia/silica_databank/all_instances.csv",
    dtype=str,
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

ttype_cheng = pd.read_csv(
    "/nfs/turbo/umms-tocho/data/data_splits/chengjia/silica_databank/srh_ttype_cheng.csv"
)
filtered.merge(
    ttype_cheng, left_on="patient", right_on="type_institution_number", how="left"
)

filtered_with_label = filtered.merge(
    ttype_cheng, left_on="patient", right_on="type_institution_number", how="left"
).drop("type_institution_number", axis=1)
filtered_with_label["ttype_cheng"] = filtered_with_label["ttype_cheng"].fillna("other")


# filtered = filtered_with_label[
#    filtered_with_label["ttype_cheng"].isin(["glioma", "normal"])
# ]

target_n = 500000
sampled = (
    filtered_with_label[filtered_with_label["patch_type"].isin({"tumor", "normal"})]
    .groupby("ttype_cheng", group_keys=False)
    .apply(lambda df: df.sample(n=min(len(df), target_n), random_state=1000))
)


sampled.to_csv("srhum_4m_.csv", index=False)
