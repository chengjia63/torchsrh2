import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

data = pd.read_csv("/home/chengjia/dbta_all.csv")
data["patient"] = data["patient"].str.replace("-", "_")
data.loc[data["tumor"]=="anaplastic_meningioma", "tumor"] = "atypical_meningioma"
data.loc[data["tumor_fine"]=="meningeal_melanocytoma", "tumor"] = "metastasis"

patients = data[["patient", "tumor"]].drop_duplicates().reset_index()

trainvaltest_pt = patients[~(patients["tumor"]=="training")].reset_index(drop=True)
trainingonly_pt = patients[patients["tumor"]=="training"].reset_index(drop=True)
trainvaltest = data[data["patient"].isin(set(trainvaltest_pt["patient"]))].reset_index(drop=True)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=1000)
train_idx, _ = list(sss.split(trainvaltest_pt, trainvaltest_pt["tumor"]))[0]

train_pt = set(trainvaltest_pt.iloc[train_idx].reset_index(drop=True)["patient"])
valtest_pt = trainvaltest_pt[~trainvaltest_pt["patient"].isin(train_pt)]

train = trainvaltest[trainvaltest["patient"].isin(train_pt)]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=1000)
val_idx, _ = list(sss.split(valtest_pt, valtest_pt["tumor"]))[0]

val_pt = set(valtest_pt.iloc[val_idx].reset_index(drop=True)["patient"])
test_pt = set(valtest_pt[~valtest_pt["patient"].isin(val_pt)].reset_index(drop=True)["patient"])

val = trainvaltest[trainvaltest["patient"].isin(val_pt)]
test = trainvaltest[trainvaltest["patient"].isin(test_pt)]

assert len(set(val["patient"]).intersection(set(test["patient"]))) == 0
assert len(set(train["patient"]).intersection(set(test["patient"]))) == 0
assert len(set(train["patient"]).intersection(set(val["patient"]))) == 0


train["patch_code"] = 8
val["patch_code"] = 8
test["patch_code"] = 8

train = train[["institution", "patient", "mosaic", "patch_code", "tumor"]]
val = val[["institution", "patient", "mosaic", "patch_code", "tumor"]]
test = test[["institution", "patient", "mosaic", "patch_code",  "tumor"]]

count = pd.DataFrame({
"train_slide": train.groupby("tumor").count()["mosaic"],
"val_slide": val.groupby("tumor").count()["mosaic"],
"test_slide": test.groupby("tumor").count()["mosaic"],
"train_pt": train[["tumor", "patient"]].drop_duplicates().groupby("tumor").count()["patient"],
"val_pt": val[["tumor", "patient"]].drop_duplicates().groupby("tumor").count()["patient"],
"test_pt": test[["tumor", "patient"]].drop_duplicates().groupby("tumor").count()["patient"]
}).reset_index()

print(count)

train.to_csv("dbta28_train.csv", index=False)
val.to_csv("dbta28_val.csv", index=False)
test.to_csv("dbta28_test.csv", index=False)



