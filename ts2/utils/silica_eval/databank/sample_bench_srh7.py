import pandas as pd
import re


def sample_train_db():
    data = pd.read_csv(
        "/nfs/turbo/umms-tocho/data/chengjia/silica_databank/test_srh7v1_diagnostic_all_instances.csv",
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

    ttype_cheng = (
        pd.read_csv(
            "/nfs/turbo/umms-tocho/data/data_splits/srh7v1/diagnostic/srh7v1_diagnostic_train.csv",
            dtype=str,
        )
        .drop(["institution", "patch_code"], axis=1)
        .rename({"patient": "patient_y", "mosaic": "mosaic_y"}, axis=1)
    )

    # nion_slides = filtered["patch"].apply(lambda x: re.findall("(NIO_UM_[0-9]+[a-zA-Z]*)-([a-zA-Z0-9]*)", x)[0])
    # filtered["nion"] =  nion_slides.apply(lambda x: x[0])
    # filtered["mosaic"] =  nion_slides.apply(lambda x: x[1])

    filtered_with_label = filtered.merge(
        ttype_cheng,
        left_on=["nion", "mosaic"],
        right_on=["patient_y", "mosaic_y"],
        how="left",
    ).drop(["patient_y", "mosaic_y"], axis=1)
    assert not filtered_with_label["label"].isna().any()

    #import pdb; pdb.set_trace()
    # filtered_with_label["ttype_cheng"] = filtered_with_label["ttype_cheng"].fillna("other")

    # filtered = filtered_with_label[
    #    filtered_with_label["ttype_cheng"].isin(["glioma", "normal"])
    # ]

    target_n = 200000
    sampled = (
        filtered_with_label[filtered_with_label["patch_type"].isin({"tumor", "normal"})]
        .groupby("label", group_keys=False)
        .apply(lambda df: df.sample(n=min(len(df), target_n), random_state=1000))
    )

    sampled.to_csv("srh7_1dot4m_.csv", index=False)


def nosample_test():
    data = pd.read_csv(
        "/nfs/turbo/umms-tocho/data/chengjia/silica_databank/instances/srh7v1_diagnostic_test_all_instances.csv",
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

    ttype_cheng = (
        pd.read_csv(
            "/nfs/turbo/umms-tocho/data/data_splits/srh7v1/diagnostic/srh7v1_diagnostic_test.csv",
            dtype=str,
        )
        .drop(["institution", "patch_code"], axis=1)
        .rename({"patient": "patient_y", "mosaic": "mosaic_y"}, axis=1)
    )

    # nion_slides = filtered["patch"].apply(lambda x: re.findall("(NIO_UM_[0-9]+[a-zA-Z]*)-([a-zA-Z0-9]*)", x)[0])
    # filtered["nion"] =  nion_slides.apply(lambda x: x[0])
    # filtered["mosaic"] =  nion_slides.apply(lambda x: x[1])

    filtered_with_label = filtered.merge(
        ttype_cheng,
        left_on=["nion", "mosaic"],
        right_on=["patient_y", "mosaic_y"],
        how="left",
    ).drop(["patient_y", "mosaic_y"], axis=1)
    assert not filtered_with_label["label"].isna().any()

    all_cells = filtered_with_label[
        filtered_with_label["patch_type"].isin({"tumor", "normal"})
    ]

    all_cells.to_csv("srh7_test_.csv", index=False)


if __name__ == "__main__":
    nosample_test()
