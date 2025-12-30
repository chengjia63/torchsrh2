import torch
import pandas as pd
import sklearn
import altair as alt
import numpy as np
import yaml
from omegaconf import OmegaConf
from ts2.train.main_cell_inference import SingleCellTempInferenceDataset
from tqdm import tqdm
import gzip
import pydicom
from PIL import Image
import gc
import matplotlib.pyplot as plt

from collections import defaultdict


import io
import base64
import einops
from torchvision.transforms.functional import adjust_contrast, adjust_brightness
from PIL import Image


from ts2.data.db_improc import MemmapTileReader
from ts2.data.transforms import HistologyTransform
from ts2.train.main_cell_inference import SingleCellListInferenceDataset
from ts2.playgrounds.tailwind import TC

def merge_list_of_dicts(dict_list):
    merged = defaultdict(list)
    for d in dict_list:
        for k, v in d.items():
            merged[k].extend(v)
    return dict(merged)



def im_to_bytestr(image) -> str:
        output = io.BytesIO()
        Image.fromarray(image).save(output, format='JPEG')
        return "data:image/jpeg;base64," + base64.b64encode(
            output.getvalue()).decode()


def sample_idx(group, n=8192, random_state=1000):
    return group.sample(n=min(n, len(group)), random_state=random_state).index


def load_data(inf_path):
    return pd.DataFrame(torch.load(inf_path))

@torch.no_grad()
def get_knn_logits(cf, train_embs, val_embs, train_labels):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    batch_size = cf["testing"]["knn"]["knn_params"]["batch_size"]
    all_scores = []
    nc = len(set(train_labels.tolist()))
    temb = train_embs.T.to(device)
    tl = train_labels.to(device)
    for k in tqdm(range(val_embs.shape[0] // batch_size + 1)):
        # find current minibatch
        start_coeff = batch_size * k
        end_coeff = min(batch_size * (k + 1),
                        val_embs.shape[0])  # leftover
        val_embs_k = val_embs[start_coeff:end_coeff].to(
            device)  # 1536 x 2048

        # knn predict on the minibatch
        _, pred_scores = knn_predict(
            val_embs_k,
            temb,
            tl,
            nc,
            knn_k=cf["testing"]["knn"]["knn_params"]["k"],
            knn_t=cf["testing"]["knn"]["knn_params"]["t"])

        # add to list
        all_scores.append(
            torch.nn.functional.normalize(pred_scores, p=1, dim=1).cpu())
        torch.cuda.empty_cache()

    return torch.vstack(all_scores)


# code for kNN prediction from here:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
def knn_predict(feature, feature_bank, feature_labels, classes: int,
                knn_k: int, knn_t: float):
    """Helper method to run kNN predictions on features based on a feature bank
    Args:
        feature: Tensor of shape [B, D] consisting of N D-dimensional features
        feature_bank: Tensor of shape [N, D], a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t: Temperature in kNN, low temperature leads to more weighted kNN.
    """
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.shape[0], -1),
                              dim=-1,
                              index=sim_indices)
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(feature.shape[0] * knn_k,
                                classes,
                                device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1,
                                          index=sim_labels.view(-1, 1),
                                          value=1.0)
    
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) *
                            sim_weight.unsqueeze(dim=-1),
                            dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels, pred_scores


def main(db_path, inf_path, out_path):
    db_data = load_data(db_path)
    db_embs = torch.stack(db_data["embeddings"].tolist())
    db_label = torch.tensor(db_data["label"].apply(lambda x: {"tumor": 1, "normal":0}[x]))

    inf_data = load_data(inf_path)
    inf_embs = torch.stack(inf_data["embeddings"].tolist())

    db_mean = db_embs.mean(dim=0)
    inf_embs = inf_embs - db_mean
    db_embs = db_embs - db_mean

    inf_embs_norm = torch.nn.functional.normalize(inf_embs, dim=1)
    db_embs_norm = torch.nn.functional.normalize(db_embs, dim=1)

    knn_conf = OmegaConf.create({
        "testing":{"knn":{"knn_params":{
            "batch_size":128,
            "k": 200,
            "t": 0.07
    }}}})

    val_predictions = get_knn_logits(knn_conf, db_embs_norm, inf_embs_norm, db_label)

    inf_data["prediction"] = val_predictions[:,0]
    inf_data["slide"] = inf_data["path"].str.extract("(NIO_[a-zA-Z]+_[0-9]+[a-zA-Z]*-[0-9a-zA-Z]*)")
    
    inf_data[["path", "slide", "label","prediction"]].to_csv(f"{out_path}.csv", index=False)

if __name__=="__main__":


    # db
    db_um2m_ours = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/2510a2cb_May30-19-10-14_sd1000_INFDB_dev/predictions/pred.pt"
    db_um2m_dv2  = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/89d3ad98_May23-13-58-49_sd1000_dev_tune0/models/eval/training_124999/17b1bb09_May30-19-44-14_sd1000_INFDB_dev/predictions/pred.pt"
    db_g2m_ours = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/e2e43218_Jun03-18-51-29_sd1000_INFDB_dev/predictions/pred.pt"
    db_g2m_dv2 = "nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/89d3ad98_May23-13-58-49_sd1000_dev_tune0/models/eval/training_124999/6b91de8d_Jun03-19-13-20_sd1000_INFDB_dev/predictions/pred.pt"

    # inf
    inf_mouse_ours = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/4100a9ab_Jun01-02-43-45_sd1000_INF_dev/predictions/pred.pt"
    inf_ucsf_ours = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/5678c19d_Jun01-19-43-34_sd1000_INF_dev/predictions/pred.pt"
    inf_mouse_dv2 = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/89d3ad98_May23-13-58-49_sd1000_dev_tune0/models/eval/training_124999/a668638d_Jun01-03-14-14_sd1000_INF_dev/predictions/pred.pt"
    inf_ucsf_dv2 = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/89d3ad98_May23-13-58-49_sd1000_dev_tune0/models/eval/training_124999/f5385048_Jun01-17-04-04_sd1000_INF_dev/predictions/pred.pt"

    main(db_g2m_ours, inf_ucsf_ours,  "out/ucsf_g2m_ours")
    main(db_g2m_ours, inf_mouse_ours, "out/mouse_g2m_ours")
    main(db_g2m_dv2, inf_ucsf_dv2,    "out/ucsf_g2m_dv2")
    main(db_g2m_dv2, inf_mouse_dv2,   "out/mouse_g2m_dv2")