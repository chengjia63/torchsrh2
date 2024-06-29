from platonic_rep.metrics import AlignmentMetrics
import torch
import numpy as np
from tqdm import tqdm
import einops

def proc_feat(feat): # ASSUMES FEATURES ARE IN SLIDE ORDER
    
    train_emb = einops.rearrange(feat["train"]["embeddings"], "(ns pps) d -> ns pps d", pps=40).mean(dim=1)
    val_emb = einops.rearrange(feat["val"]["embeddings"], "(ns pps) d -> ns pps d", pps=40).mean(dim=1)
    return torch.cat((train_emb, val_emb))

indomain_simclr_feature = proc_feat(torch.load("indomain_simclr_feature.pt"))
indomain_hidisc_feature = proc_feat(torch.load("indomain_hidisc_feature.pt"))

rn34_feat = proc_feat(torch.load("rn34_feature.pt"))
dinov2_feat = proc_feat(torch.load("dinov2_feature.pt"))
clip_feat = proc_feat(torch.load("clip_feature.pt"))
hipt_feat = proc_feat(torch.load("hipt_feature.pt"))
ctp_feat = proc_feat(torch.load("ctp_feature.pt"))

conch_feat = proc_feat(torch.load("conch_feature.pt"))
plip_feat = proc_feat(torch.load("plip_feature.pt"))
uni_feat = proc_feat(torch.load("uni_feature.pt"))
virchow_feat = proc_feat(torch.load("virchow_feature.pt"))
gigapath_feat = proc_feat(torch.load("gigapath_feature.pt"))

features = [rn34_feat, dinov2_feat, clip_feat, indomain_simclr_feature, indomain_hidisc_feature, ctp_feat, hipt_feat, plip_feat, conch_feat, uni_feat, virchow_feat, gigapath_feat]
feat = [[
    AlignmentMetrics.mutual_knn(f1,
                                f2,
                                topk=10)
    for f2 in features
] for f1 in tqdm(features)]

print(feat)
print(np.array(feat))
import pdb
pdb.set_trace()
