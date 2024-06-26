from platonic_rep.metrics import AlignmentMetrics
import torch
import numpy as np
from tqdm import tqdm


def make_slide_emb(uni_feat):
    #train_slide = [i.split("@")[0] for i in uni_feat["train"]["path"]]
    #train_slide_set = np.array(sorted(set(train_slide)))
    #train_embs = torch.tensor((np.expand_dims(train_slide, 1) == train_slide_set).astype(float))
    #train_slide_mask = torch.tensor((np.expand_dims(train_slide, 1) == train_slide).astype(float))

    val_slide = [i.split("@")[0] for i in uni_feat["val"]["path"]]
    val_slide_set = np.array(sorted(set(val_slide)))
    val_embs = torch.tensor((np.expand_dims(val_slide,
                                            1) == val_slide_set).astype(float))
    val_slide_mask = torch.tensor(
        (np.expand_dims(val_slide, 1) == val_slide).astype(float))

    return {
        #"train": {
        #    "embeddings": train_embs,
        #    "mask": train_slide_mask
        #},
        "val": {
            "embeddings": val_embs,
            "mask": val_slide_mask
        }
    }







rn34_feat = torch.load("rn34_feature.pt")
#rn50_feat = torch.load("rn50_feature.pt")
#rnext50_feat = torch.load("rnext50_feature.pt")
#rnext101_feat = torch.load("rnext101_feature.pt")

dinov2_feat = torch.load("dinov2_feature.pt")
clip_feat = torch.load("clip_feature.pt")

id_simclr = torch.load("indomain_simclr_feature.pt")
id_hidisc = torch.load("indomain_hidisc_feature.pt")

hipt_feat = torch.load("hipt_feature.pt")
ctp_feat = torch.load("ctp_feature.pt")
conch_feat = torch.load("conch_feature.pt")
plip_feat = torch.load("plip_feature.pt")
uni_feat = torch.load("uni_feature.pt")
virchow_feat = torch.load("virchow_feature.pt")
gigapath_feat = torch.load("gigapath_feature.pt")
slide_feat = make_slide_emb(uni_feat)

feat_lists = [slide_feat, rn34_feat, dinov2_feat, clip_feat, id_simclr, id_hidisc, ctp_feat, hipt_feat, plip_feat, conch_feat, uni_feat, virchow_feat, gigapath_feat]

feat = [[
    AlignmentMetrics.mutual_knn(f1["val"]["embeddings"],
    #AlignmentMetrics.masked_mutual_knn(f1["val"]["embeddings"],
                                f2["val"]["embeddings"],
    #                           same_slide_mask=slide_feat["val"]["mask"].to(bool),
                                topk=40)
    for f2 in feat_lists] for f1 in tqdm(feat_lists)]

print(feat)
print(np.array(feat))
import pdb

pdb.set_trace()
