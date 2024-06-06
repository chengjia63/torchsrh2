import os
import gzip
import logging
from os.path import join as opjoin
from typing import Dict, List, Union, Any
from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

import torch
import pytorch_lightning as pl
from torchmetrics import AveragePrecision, Accuracy, AUROC, Recall, Specificity

from torchsrh.utils.open_color import OpenColor
from torchsrh.lightning_modules import (CESystem, FinetuneSystem, SimCLRSystem,
                                        SupConSystem, VICRegSystem,
                                        VICRegSystemWithMask, SimSiamSystem,
                                        BYOLSystem, HiDiscSystem,
                                        ExemplarLearningSystem, EvalSystem)
from torchsrh.lightning_modules.mil_systems import (MILSystem, MILEvalSystem,
                                                    MILRegionPoolingEvalSystem,
                                                    MILPoolingEvalSystem,
                                                    CLAMSystem)
from torchsrh.lightning_modules.plip_systems import PLIPSystem


def load_prediction(pred_fname):
    """Loading in a prediction file."""
    if os.path.exists(pred_fname):  # compressed version
        with gzip.open(pred_fname) as fd:
            predictions = torch.load(fd)
    else:  # uncompressed version
        unzipped_fname = pred_fname.removesuffix('.gz')
        if os.path.exists(unzipped_fname):
            predictions = torch.load(unzipped_fname, map_location="cpu")
        else:
            logging.critical("can not find predictions %s" % pred_fname)
            exit(1)
    return predictions


def get_knn_logits(cf, train_predictions, val_predictions):
    if "logits" in val_predictions:
        logging.warning("found existing logits. deleting for re-knn eval")
        del val_predictions["logits"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    with torch.no_grad():

        train_embs = torch.nn.functional.normalize(
            train_predictions["embeddings"], p=2, dim=1).T.to(device)
        val_embs = torch.nn.functional.normalize(val_predictions["embeddings"],
                                                 p=2,
                                                 dim=1)
        train_labels = train_predictions["label"].to(device)
        batch_size = cf["testing"]["knn"]["knn_params"]["batch_size"]
        all_scores = []

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
                train_embs,
                train_labels,
                len(cf["data"]["test_dataset"]["classes_reorder"]),
                knn_k=cf["testing"]["knn"]["knn_params"]["k"],
                knn_t=cf["testing"]["knn"]["knn_params"]["t"])

            # add to list
            all_scores.append(
                torch.nn.functional.normalize(pred_scores, p=1, dim=1).cpu())
            torch.cuda.empty_cache()

    # add to predictions dict
    val_predictions["logits"] = torch.vstack(all_scores)
    return val_predictions


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
