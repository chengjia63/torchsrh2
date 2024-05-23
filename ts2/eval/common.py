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


def load_predictions(pred_fname):
    if os.path.exists(pred_fname):  # compressed version
        with gzip.open(pred_fname) as fd:
            predictions = torch.load(fd)
    else:  # uncompressed version
        unzipped_fname = pred_fname.removesuffix('.gz')
        if os.path.exists(unzipped_fname):
            predictions = torch.load(unzipped_fname, map_location="cpu")
        else:
            logging.critical(f"can not find predictions {pred_fname}")
            exit(1)
    return predictions


def get_train_val_embeddings(
    cf: Dict[str, Any], exp_root: str, datamodules: pl.LightningDataModule
) -> Dict[str, Union[torch.Tensor, List[str]]]:

    params = {
        "cf": cf,
        "num_it_per_ep": 0,
        "map_location": torch.device("cpu")
    }

    eval_mode = cf["eval"].get("eval_mode", "patch")
    if eval_mode == "patch":
        pl_module = CESystem
        params["nc"] = len(loaders["valid"].dataset.classes_)
        params["wts"] = None
    elif eval_mode == "patch_knn":
        pl_module = EvalSystem
    elif eval_mode == 'slide_mil':
        pl_module = MILSystem
        params["nc"] = len(loaders["valid"].dataset.classes_)
        params["wts"] = None
    elif eval_mode == 'plip':
        pl_module = PLIPSystem
    elif eval_mode == 'slide_clam':
        pl_module = CLAMSystem
        params["nc"] = len(loaders["valid"].dataset.classes_)
        params["wts"] = None
    elif eval_mode == "slide_mil_knn":
        pl_module = MILEvalSystem
    elif eval_mode == "slide_mil_region_mean":
        pl_module = MILRegionPoolingEvalSystem
    elif eval_mode == "slide_avg_pool":
        pl_module = partial(MILPoolingEvalSystem,
                            pool_func=lambda x: torch.mean(x, dim=0))
    elif eval_mode == "slide_max_pool":
        pl_module = partial(MILPoolingEvalSystem,
                            pool_func=lambda x: torch.max(x, dim=0).values)
    else:
        raise ValueError(
            f"eval_mode expected to be one of " +
            "\{patch, slide_mil, slide_avg_pool, slide_max_pool\}, " +
            f"got {eval_mode}")

    if eval_mode in {
            "slide_avg_pool", "slide_max_pool", "plip", "slide_mil_region_mean"
    }:
        model = pl_module(cf=cf, num_it_per_ep=0)
    elif cf["eval"].get("imagenet_backbone_checkpoint", None):
        ckpt_dict = torch.load(cf["eval"]["imagenet_backbone_checkpoint"],
                               map_location="cpu")
        if "fc.weight" in ckpt_dict and "fc.bias" in ckpt_dict:
            del ckpt_dict["fc.weight"]
            del ckpt_dict["fc.bias"]
        model = pl_module(cf=cf,
                          wts=[],
                          num_it_per_ep=0,
                          nc=len(cf["data"]["camera_ready_classes"]))
        model.model.bb.load_state_dict(ckpt_dict, strict=False)
    else:
        ckpt_path = opjoin(cf["infra"]["log_dir"], cf["infra"]["exp_name"],
                           cf["eval"]["ckpt_path"])
        model = pl_module.load_from_checkpoint(ckpt_path,
                                               strict=False,
                                               **params)
    print(model)
    trainer = pl.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        default_root_dir=exp_root,
        inference_mode=True,
        deterministic=True)

    predictions_s = trainer.predict(model, datamodule=datamodules)

    import pdb
    pdb.set_trace()

    #def process_predictions(predictions):
    #    pred = {}
    #    for k in predictions[0].keys():
    #        if k == "path":
    #            pred[k] = [pk for p in predictions for pk in p[k][0]]
    #        else:
    #            pred[k] = torch.cat([p[k] for p in predictions])
    #    return pred


#
#def pred_one_set(s: str):
#    predictions_s = trainer.predict(model, dataloaders=loaders[s])
#    predictions_s = process_predictions(predictions_s)
#
#    logging.info(
#        f"{s}_embeddings shape {predictions_s['embeddings'].shape}")
#    if "logits" in predictions_s:
#        logging.info(f"{s}_logits shape {predictions_s['logits'].shape}")
#    return predictions_s
#
#if "valid" in loaders:
#    loaders["val"] = loaders.pop("valid")  # TODO: refactor
#return {s: pred_one_set(s) for s in loaders}


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


def make_tsne(cf: Dict[str, Any], exp_root: str,
              predictions: Dict[str, Union[torch.Tensor, List[str]]]):
    raise NotImplementedError()

    classes = cf["data"]["camera_ready_classes"]

    # generate random samples
    embs = predictions["embeddings"].squeeze()
    n_patches = cf['tsne']['num_patches']
    indices = torch.randperm(len(embs))[:n_patches]
    embs = embs[indices]
    labels = predictions["label"][indices]

    nc = len(labels.unique())
    rgb = OpenColor.setup_colors(nc)

    temp = torch.clone(labels)
    for i, j in enumerate(cf["data"]["test_dataset"]["classes_reorder"]):
        if i != j: temp[labels == j] = i
    labels = temp

    colors = rgb[labels]

    tsne = TSNE(n_components=2, **cf["tsne"]["class_params"])
    xy = tsne.fit_transform(embs)

    fig, ax = plt.subplots(nrows=1,
                           ncols=1,
                           num=None,
                           figsize=(4, 4),
                           dpi=300,
                           facecolor='w',
                           edgecolor='k')

    for i in range(nc):
        plt.scatter(x=xy[labels == i, 0],
                    y=xy[labels == i, 1],
                    c=colors[labels == i].squeeze(),
                    s=1,
                    alpha=.4,
                    label=classes[i])

    ax.axis("equal")
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.set_axis_off()
    fig.tight_layout()
    out_fn = opjoin(exp_root, "tsne.svg")
    plt.savefig(out_fn)
    out_fn = opjoin(exp_root, "tsne.png")
    plt.savefig(out_fn)

    ax.legend()
    ax.set_axis_off()
    out_fn = opjoin(exp_root, "tsne_legend.svg")
    plt.savefig(out_fn)


def plot_confusion(cf, confusion, labels, out_file):
    raise NotImplementedError()
    default_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(1, 1)
    labels = np.array(labels)[cf["data"]["test_dataset"]["classes_reorder"]]
    confusion = confusion[
        cf["data"]["classes_reorder"], :][:, cf["data"]["classes_reorder"]]
    confusion_normalized = confusion / np.tile(confusion.sum(axis=1),
                                               (len(labels), 1)).T
    im = ax.imshow(confusion_normalized, cmap=plt.get_cmap("Blues"))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    #ax.axes.xaxis.set_ticklabels([])
    #ax.axes.yaxis.set_ticklabels([])

    im.set_clim(0, 1)
    #fig.colorbar(im)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    thres = np.max(confusion_normalized) / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            if confusion_normalized[i, j] == 0: continue
            color = "k" if confusion_normalized[i, j] < thres else "w"
            ax.text(j,
                    i,
                    confusion[i, j],
                    ha="center",
                    va="center",
                    color=color,
                    size="small")

    fig.tight_layout()
    plt.savefig(out_file)
    plt.savefig(out_file + ".pdf")
    plt.close(fig)

    plt.rcParams.update({'font.size': default_fontsize})


def make_specs_non_hist(
        cf: Dict[str, Any], exp_root: str,
        predictions: Dict[str, Union[torch.Tensor, List[str]]]) -> None:
    """Compute all specs for an experiment"""

    # aggregate prediction into a dataframe
    pred = pd.DataFrame.from_dict({
        "path":
        predictions["path"],
        "labels": [l.item() for l in list(predictions["label"])],
        "logits": [l.tolist() for l in list(predictions["logits"])]
    })

    pred["logits"] = pred["logits"].apply(
        lambda x: torch.nn.functional.softmax(torch.tensor(x), dim=0))
    normalize_f = lambda x: torch.nn.functional.normalize(x, dim=1, p=1)
    patch_logits = normalize_f(torch.tensor(np.vstack(pred["logits"])))
    patch_label = torch.tensor(pred["labels"])

    # generate metrics
    nc = len(cf["data"]["camera_ready_classes"])
    nci = cf["data"]["negative_class_index"]
    all_metrics = get_all_metrics(patch_logits, patch_label, nc,
                                  nci).unsqueeze(0)
    all_metrics = pd.DataFrame(all_metrics,
                               columns=[
                                   "acc", "t2", "t3", "mca", "map",
                                   "tumor_fnr", "auc", "sen", "spec"
                               ],
                               index=["metric"])
    all_metrics.to_csv(opjoin(exp_root, "metrics.csv"), index=False)

    classes = cf["data"]["camera_ready_classes"]
    make_conf_name = lambda x: opjoin(exp_root, f"{x}_conf.svg")

    patch_conf = confusion_matrix(y_true=patch_label,
                                  y_pred=patch_logits.argmax(dim=1))
    plot_confusion(cf, patch_conf, classes, make_conf_name("patch"))

    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    logging.info(f"all metrics\n{str(all_metrics)}")
    print(all_metrics)

    return all_metrics


def get_all_metrics(logits, label, nc: int, neg_class_idx: int):
    if nc == 2:
        auprc = AveragePrecision(task="binary", num_classes=nc)
        acc = Accuracy(task="binary", average="micro")
        mca = Accuracy(task="multiclass", num_classes=nc, average="macro")
        auroc = AUROC(task="binary", average="macro")
        spec = Specificity(task="binary")
        sen = Recall(task="binary")

        acc_val = acc(logits[:, 1], label)
        t2_val = torch.tensor(1.0)
        t3_val = torch.tensor(1.0)
        mca_val = mca(logits, label)
        map_val = auprc(logits[:, 1], label)
        auroc_val = auroc(logits[:, 1], label)
        sen_val = sen(logits[:, 1], label)
        spec_val = spec(logits[:, 1], label)

    else:
        acc = Accuracy(task="multiclass", num_classes=nc, average="micro")
        t2 = Accuracy(task="multiclass",
                      num_classes=nc,
                      top_k=2,
                      average="micro")
        t3 = Accuracy(task="multiclass",
                      num_classes=nc,
                      top_k=3,
                      average="micro")
        mca = Accuracy(task="multiclass", num_classes=nc, average="macro")
        auroc = AUROC(task="multiclass", num_classes=nc, average="macro")
        auprc = AveragePrecision(task="multiclass",
                                 num_classes=nc,
                                 average="macro")
        spec = Specificity(task="multiclass", num_classes=nc, average="micro")
        sen = Recall(task="multiclass", num_classes=nc, average="micro")

        acc_val = acc(logits, label)
        t2_val = t2(logits, label)
        t3_val = t3(logits, label)
        mca_val = mca(logits, label)
        map_val = auprc(logits, label)
        auroc_val = auroc(logits, label)
        sen_val = sen(logits, label)
        spec_val = spec(logits, label)

    fn = (logits.argmax(dim=1) == neg_class_idx) & (label != neg_class_idx)
    tumor_notumor_fnr = fn.sum() / len(fn)

    return torch.stack((acc_val, t2_val, t3_val, mca_val, map_val,
                        tumor_notumor_fnr, auroc_val, sen_val, spec_val))
