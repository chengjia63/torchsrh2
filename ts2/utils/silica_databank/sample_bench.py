import pandas as pd
data = pd.read_csv("all_instances.csv", dtype=str)


disallowed_df = pd.DataFrame([("NIO_UM_937b", "4"), ("NIO_UM_950", "3")], columns=["patient", "mosaic"])
filtered = data.merge(disallowed_df, on=["patient", "mosaic"], how='left', indicator=True)
filtered = filtered[filtered['_merge'] == 'left_only'].drop(columns='_merge')
filtered = filtered[~ filtered["patch"].str.contains("INV")]

filtered = filtered.rename({"patient": "nion"}, axis=1) 
filtered["patient"] = filtered["nion"].str.extract("(NIO_UM_[0-9]+)")

ttype_cheng = pd.read_csv("../playgrounds/srh_ttype_cheng.csv")
filtered.merge(ttype_cheng, left_on="patient", right_on="type_institution_number", how="left")

filtered_with_label = filtered.merge(ttype_cheng, left_on="patient", right_on="type_institution_number", how="left").drop("type_institution_number", axis=1)
filtered_with_label["ttype_cheng"] = filtered_with_label["ttype_cheng"].fillna("other")
filtered = filtered_with_label[filtered_with_label["ttype_cheng"].isin(["glioma", "normal"])]

filtered = filtered.drop("patient", axis=1).rename({"nion": "patient"}, axis=1)
sampled = filtered[filtered["patch_type"].isin({"tumor", "normal"})].groupby("patch_type").sample(1000000, random_state=1000)
sampled.to_csv("srhum_glioma_2m_.csv", index=False)
