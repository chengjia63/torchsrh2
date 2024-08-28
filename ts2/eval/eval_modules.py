import io
import re
import os
import json
import base64
import logging
import argparse
from os.path import join as opjoin
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Any, Optional
from functools import partial
import gzip
from tqdm import tqdm

import imageio
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

import torch
from torchvision.transforms import Compose
import pytorch_lightning as pl
from torchmetrics import AveragePrecision, Accuracy, AUROC, Recall, Specificity, F1Score
import torchmetrics.functional.classification as tmfc

from torchsrh.train.infra import read_cf
from torchsrh.utils.open_color import OpenColor
from torchsrh.utils.rgb_srh import SRHRGBToolbox
from torchsrh.datasets.mnist_dataset import MNISTDataset
from torchsrh.datasets.cifar_dataset import CIFAR10Dataset

from torchsrh.datasets.cell_dataset import scSRHXH
from torchsrh.datasets.db_improc import (get_srh_base_aug, process_read_srh,
                                         process_read_png, ProcessReadNPY,
                                         get_transformations, get_he_base_aug,
                                         get_vision_base_aug)
from torchsrh.eval.infra import setup_eval_module_standalone_infra

tqdm.pandas()


class EvalBaseModule(ABC):

    def __init__(self, cf, out_root, is_knn):
        self.cf_ = cf
        self.out_root_ = out_root
        self.is_knn_ = is_knn
        self.visualize_xmplrs_ = self.cf_.get("nn_viz", {}).get(
            "do_nn_viz", None) and ("exemplars" in self.cf_.get("data", {}))
        self.nc_ = len(
            self.cf_["data"]["test_dataset"]["camera_ready_classes"])
        self.class_names_ = np.array(
            self.cf_["data"]["test_dataset"]["camera_ready_classes"])[
                cf["data"]["test_dataset"]["classes_reorder"]]
        self.classes_ = {}
        logging.info(f"{self.nc_} class eval")

    def proc_pred_one_set__(self, pred):
        if not pred: return None

        if "logits" not in pred:
            pred["logits"] = torch.zeros((len(pred["label"]), self.nc_))

        return pd.DataFrame({
            "logits":
            pred["logits"]
            [:, self.cf_["data"]["test_dataset"]["classes_reorder"]].tolist(),
            "embeddings":
            pred["embeddings"].tolist(),
            "labels":
            self.reorder_classes(pred["label"]).tolist(),
            "pred":
            self.reorder_classes(pred["logits"].argmax(dim=1)).tolist(),
            "path":
            pred["path"]
        }).sort_values(by="path", ignore_index=True)

    def process_predictions(self, preds):
        if "val" in preds:
            self.val_preds_ = self.proc_pred_one_set__(preds["val"])
        else:
            self.val_preds_ = None

        if "train" in preds:
            self.train_preds_ = self.proc_pred_one_set__(preds["train"])
        else:
            self.train_preds_ = None

        if "xmplr" in preds:
            self.xmplr_preds_ = self.proc_pred_one_set__(preds["xmplr"])
        else:
            self.xmplr_preds_ = None

        self.preds_ = {
            "train": self.train_preds_,
            "val": self.val_preds_,
            "xmplr": self.xmplr_preds_
        }

    def sample_predictions(self,
                           which_set=["val"],
                           n_sample=[0]) -> pd.DataFrame:

        def sample_df(curr_pred, num_sample=None):
            if curr_pred is None:
                raise ValueError("Found a empty prediction dataframe. " +
                                 "Check config.")

            num_patches = len(curr_pred)
            if ((num_sample is not None) and (0 < num_sample)
                    and (num_sample < num_patches)):
                return curr_pred.sample(n=num_sample,
                                        replace=False,
                                        random_state=np.random.RandomState(
                                            self.cf_["infra"]["seed"]))
            else:
                return curr_pred

        return {
            k: sample_df(self.preds_[k], num_sample=n)
            for k, n in zip(which_set, n_sample)
        }

    def cat_predictions_dict(self, pred_dict):
        for k in pred_dict:
            pred_dict[k]["is_xmplr"] = (k == "xmplr")
        return pd.concat([pred_dict[k] for k in pred_dict])

    def visualize_feature_rank(self):
        val_embs = torch.tensor(self.val_preds_["embeddings"])
        logging.info(val_embs.shape)

        s = torch.linalg.svdvals(val_embs)
        r = torch.linalg.matrix_rank(val_embs)

        _, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
        ax.plot(np.arange(len(s)), s)
        ax.set_yscale('log')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_label("Singular values (log scale)")
        ax.xaxis.set_label("Singular values order")
        ax.grid(axis='both')
        ax.set_ylim([torch.pow(10, torch.floor(torch.log10(s[-1]))), s[0]])
        ax.set_title(f"rank = {r}")
        plt.tight_layout()
        plt.savefig(opjoin(self.out_root_, "val_emb_rank.png"))
        plt.savefig(opjoin(self.out_root_, "val_emb_rank.svg"))

        #with gzip.open(opjoin(self.out_root_, "val_emb_rank.pt.gz"),
        #               "w") as fd:
        #    torch.save(s, fd)

    @abstractmethod
    def replace_patch_name(self, in_str: str) -> str:
        raise NotImplementedError()

    @abstractmethod
    def reorder_classes(self, array_in: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def get_im(self, im_row):
        raise NotImplementedError()

    @abstractmethod
    def get_patient(self, x):
        raise NotImplementedError()

    @abstractmethod
    def get_slide(self, x):
        raise NotImplementedError()

    @staticmethod
    def im_to_bytestr(image: Image) -> str:
        output = io.BytesIO()
        image.save(output, format='JPEG')
        return "data:image/jpeg;base64," + base64.b64encode(
            output.getvalue()).decode()

    def get_all_metrics(self,
                        logits,
                        label,
                        neg_class_idx=None,
                        one_v_all_idx: List[int] = []) -> List:
        if self.nc_ == 2:
            auprc = AveragePrecision(task="binary", num_classes=self.nc_)
            acc = Accuracy(task="binary", average="micro")
            mca = Accuracy(task="multiclass",
                           num_classes=self.nc_,
                           average="macro")
            auroc = AUROC(task="binary", average="macro")
            spec = Specificity(task="binary")
            sen = Recall(task="binary")
            f1 = F1Score(task="binary")

            acc_val = acc(logits[:, 1], label)
            t2_val = torch.tensor(1.0)
            t3_val = torch.tensor(1.0)
            mca_val = mca(logits, label)
            map_val = auprc(logits[:, 1], label)
            auroc_val = auroc(logits[:, 1], label)
            sen_val = sen(logits[:, 1], label)
            spec_val = spec(logits[:, 1], label)
            f1_val = f1(logits[:, 1], label)

        else:
            acc = Accuracy(task="multiclass",
                           num_classes=self.nc_,
                           top_k=1,
                           average="micro")
            t2 = Accuracy(task="multiclass",
                          num_classes=self.nc_,
                          top_k=2,
                          average="micro")
            t3 = Accuracy(task="multiclass",
                          num_classes=self.nc_,
                          top_k=3,
                          average="micro")
            mca = Accuracy(task="multiclass",
                           num_classes=self.nc_,
                           top_k=1,
                           average="macro")
            auroc = AUROC(task="multiclass",
                          num_classes=self.nc_,
                          average="macro")
            auprc = AveragePrecision(task="multiclass",
                                     num_classes=self.nc_,
                                     average="macro")
            spec = Specificity(task="multiclass",
                               num_classes=self.nc_,
                               top_k=1,
                               average="micro")
            sen = Recall(task="multiclass",
                         num_classes=self.nc_,
                         top_k=1,
                         average="micro")
            f1_macro = F1Score(task="multiclass",
                               num_classes=self.nc_,
                               top_k=1,
                               average="macro")
            f1_micro = F1Score(task="multiclass",
                               num_classes=self.nc_,
                               top_k=1,
                               average="micro")

            acc_val = acc(logits, label)
            t2_val = t2(logits, label)
            t3_val = t3(logits, label)
            mca_val = mca(logits, label)
            map_val = auprc(logits, label)
            auroc_val = auroc(logits, label)
            sen_val = sen(logits, label)
            spec_val = spec(logits, label)

            f1_macro_val = f1_macro(logits, label)
            f1_micro_val = f1_micro(logits, label)

        metrics = [
            acc_val, t2_val, t3_val, mca_val, map_val, auroc_val, sen_val,
            spec_val
        ]
        metric_names = [
            "acc", "t2", "t3", "mca", "map", "auroc", "sen", "sepc"
        ]

        if self.nc_ == 2:
            metrics.append(f1_val)
            metric_names.append("f1")
        else:
            metrics.extend([f1_macro_val, f1_micro_val])
            metric_names.extend(["f1_macro", "f1_micro"])

        if neg_class_idx is not None:
            fn = (logits.argmax(dim=1) == neg_class_idx) & (label
                                                            != neg_class_idx)
            tumor_notumor_fnr = fn.sum() / len(fn)
            metrics.append(tumor_notumor_fnr)
            metric_names.append("tumor_fnr")

        for i in one_v_all_idx:
            pred = logits[:, i]
            gt = (label == i)
            cn = self.cf_["data"]["test_dataset"]["camera_ready_classes"][i]

            ova_acc_i = tmfc.binary_accuracy(pred, gt)
            ova_auroc_i = tmfc.binary_auroc(pred, gt)
            ova_sen_i = tmfc.binary_recall(pred, gt)
            ova_spec_i = tmfc.binary_specificity(pred, gt)
            ova_precision_i = tmfc.binary_precision(pred, gt)
            ova_f1_i = tmfc.binary_f1_score(pred, gt)
            ova_auprc_i = tmfc.binary_average_precision(pred, gt)

            metrics.extend([
                ova_acc_i, ova_auroc_i, ova_sen_i, ova_spec_i, ova_precision_i,
                ova_f1_i, ova_auprc_i
            ])
            metric_names.extend([
                f"{cn}_acc", f"{cn}_auroc", f"{cn}_sen", f"{cn}_spec",
                f"{cn}_precision", f"{cn}_f1", f"{cn}_auprc"
            ])

        return torch.stack(metrics).tolist(), metric_names

    def plot_confusion(self,
                       confusion: np.ndarray,
                       out_file: str,
                       nc: Optional[int] = None,
                       class_names: Optional[List] = None):
        if nc is None: nc = self.nc_
        if class_names is None: class_names = self.class_names_

        default_fontsize = plt.rcParams["font.size"]
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(1, 1)

        confusion = np.array(confusion)
        confusion_normalized = confusion / np.tile(confusion.sum(axis=1),
                                                   (nc, 1)).T
        im = ax.imshow(confusion_normalized, cmap=plt.get_cmap("Blues"))
        ax.set_xticks(np.arange(nc))
        ax.set_yticks(np.arange(nc))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
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

        if np.max(confusion) > 999 or ((nc > 7) and (np.max(confusion) > 99)):
            fontsize = 8
        else:
            fontsize = "small"

        for i in range(nc):
            for j in range(nc):
                if confusion_normalized[i, j] == 0: continue
                color = "k" if confusion_normalized[i, j] < thres else "w"
                ax.text(j,
                        i,
                        confusion[i, j],
                        ha="center",
                        va="center",
                        color=color,
                        fontdict={"size": fontsize})

        fig.tight_layout()
        plt.savefig(out_file)
        plt.savefig(out_file + ".pdf")
        plt.close(fig)

        plt.rcParams.update({'font.size': default_fontsize})

    def make_sample_df(self, smpl_pred_):
        smpl_pred_["patient"] = smpl_pred_["path"].apply(self.get_patient)
        smpl_pred_["slide"] = smpl_pred_["path"].apply(self.get_slide)
        smpl_pred_["patch_name"] = smpl_pred_["path"].apply(
            self.get_patch_name)
        if self.visualize_xmplrs_:
            smpl_pred_["is_xmplr"] = smpl_pred_["is_xmplr"]

        if self.cf_["testing"]["tsne"]["interactive"]:
            smpl_pred_["image"] = smpl_pred_.progress_apply(self.get_im,
                                                            axis=1)

        self.classes_["class_name"] = [
            self.cf_["data"]["test_dataset"]["camera_ready_classes"][i]
            for i in self.cf_["data"]["test_dataset"]["classes_reorder"]
        ]
        smpl_pred_["class_name"] = smpl_pred_["labels"].apply(
            lambda x: self.classes_["class_name"][x])
        return smpl_pred_

    def make_sample_tsne(self, data):
        tsne = TSNE(n_components=2,
                    **self.cf_["testing"]["tsne"]["class_params"])
        xy = tsne.fit_transform(np.array([d for d in data["embeddings"]]))
        data["x"] = xy[:, 0].tolist()
        data["y"] = xy[:, 1].tolist()
        return data

    def make_tsne_matplotlib(self,
                             smpl_data_,
                             color_key="class_name",
                             highlight={}):
        nc = len(self.classes_[color_key])
        rgb = {
            cl: co
            for cl, co in zip(self.classes_[color_key],
                              OpenColor.setup_colors(nc, ind=8))
        }
        colors = np.array([rgb[k] for k in smpl_data_[color_key]])
        if highlight:
            colors[(
                ~smpl_data_[color_key].isin(highlight))] = OpenColor.gray[8]

        self.fig_, self.ax_ = plt.subplots(nrows=1,
                                           ncols=1,
                                           num=None,
                                           figsize=(4, 4),
                                           dpi=300,
                                           facecolor='w',
                                           edgecolor='k')

        for i in self.classes_[color_key]:
            alpha = .03 if ((highlight) and (i not in highlight)) else .4
            x_data = smpl_data_["x"][smpl_data_[color_key] == i]
            y_data = smpl_data_["y"][smpl_data_[color_key] == i]
            c_data = colors[smpl_data_[color_key] == i].squeeze()
            if len(x_data) == 1: c_data = np.expand_dims(c_data, 0)
            plt.scatter(x=x_data,
                        y=y_data,
                        c=c_data,
                        s=1,
                        alpha=alpha,
                        label=i)

        if self.visualize_xmplrs_:
            for i in self.classes_[color_key]:
                alpha = .03 if ((highlight) and (i not in highlight)) else 1
                x_data = smpl_data_["x"][(smpl_data_[color_key] == i)
                                         & (smpl_data_["is_xmplr"])]
                y_data = smpl_data_["y"][(smpl_data_[color_key] == i)
                                         & (smpl_data_["is_xmplr"])]
                c_data = colors[(smpl_data_[color_key] == i)
                                & (smpl_data_["is_xmplr"])].squeeze()
                if len(x_data) == 1: c_data = np.expand_dims(c_data, 0)
                plt.scatter(x=x_data,
                            y=y_data,
                            c=c_data,
                            s=3,
                            alpha=alpha,
                            label=i,
                            marker="D",
                            ec="black",
                            lw=1)

        self.ax_.axis("equal")
        self.ax_.yaxis.set_ticks_position('none')
        self.ax_.xaxis.set_ticks_position('none')
        self.ax_.axes.xaxis.set_ticklabels([])
        self.ax_.axes.yaxis.set_ticklabels([])
        self.ax_.set_axis_off()
        self.fig_.tight_layout()

    def make_tsne_altair(self, smpl_data_, color_key="class_name"):
        alt.data_transformers.disable_max_rows()
        rgb = OpenColor.setup_colors(len(set(smpl_data_[color_key].tolist())),
                                     ind=8)
        if self.visualize_xmplrs_:
            chart_encode_params = {
                "shape":
                alt.Shape('is_xmplr',
                          scale=alt.Scale(domain=[False, True],
                                          range=["circle", "diamond"])),
                "strokeWidth":
                alt.Stroke('is_xmplr',
                           scale=alt.Scale(domain=[False, True], range=[0, 3]))
            }
        else:
            chart_encode_params = {"strokeWidth": alt.value(0)}

        bind_range = alt.binding_range(min=5, max=100, name='Size ')
        param_size = alt.param(bind=bind_range)

        #bind_range_op = alt.binding_range(min=0, max=1, name='Opacity ')
        #param_op = alt.param(bind=bind_range_op)

        patient_list = sorted(list(set(smpl_data_["patient"])))
        patient_bind = alt.binding_select(options=[None] + patient_list,
                                          labels=["All"] + patient_list,
                                          name='Patient ')
        patient_selection = alt.selection_point(fields=['patient'],
                                                bind=patient_bind)
        slide_list = sorted(list(set(smpl_data_["slide"])))
        slide_bind = alt.binding_select(options=[None] + slide_list,
                                        labels=["All"] + slide_list,
                                        name='Slide ')
        slide_selection = alt.selection_point(fields=['slide'],
                                              bind=slide_bind)
        class_selection = alt.selection_point(fields=[color_key],
                                              bind='legend')

        color = alt.condition(
            patient_selection & slide_selection & class_selection,
            alt.Color(f"{color_key}:N",
                      scale=alt.Scale(domain=self.classes_[color_key],
                                      range=rgb.tolist())),
            alt.value('lightgray'))

        op = alt.condition(
            patient_selection & slide_selection & class_selection,
            alt.value(1.0), alt.value(0.1))

        get_range = lambda x: x.max() - x.min()
        get_center = lambda x: (x.max() + x.min()) / 2
        diff = max(get_range(smpl_data_["x"]), get_range(smpl_data_["y"])) / 2
        x_mid = get_center(smpl_data_["x"])
        y_mid = get_center(smpl_data_["y"])

        if self.visualize_xmplrs_:
            smpl_data_ = smpl_data_.sort_values(by="is_xmplr")

        chart = alt.Chart(smpl_data_).mark_point(
                filled=True,
                size=alt.expr(param_size.name),
                stroke="#000000"
            ).encode(
                x=alt.X("x",
                        axis=alt.Axis(tickSize=0),
                        scale=alt.Scale(domain=[x_mid - diff, x_mid + diff])),
                y=alt.Y("y",
                        axis=alt.Axis(tickSize=0),
                        scale=alt.Scale(domain=[y_mid - diff, y_mid + diff])),
                color=color,
                opacity=op,
                tooltip=['image', "patch_name"],
                **chart_encode_params
            ).add_params(
                param_size,
                patient_selection,
                slide_selection,
                class_selection
            ) # yapf: disable

        self.chart_ = chart

    def save_matplotlib_chart(self, out_name="tsne"):
        out_fn = opjoin(self.out_root_, f"{out_name}.svg")
        plt.savefig(out_fn)
        out_fn = opjoin(self.out_root_, f"{out_name}.png")
        plt.savefig(out_fn)

        self.ax_.legend()
        self.ax_.set_axis_off()
        out_fn = opjoin(self.out_root_, f"{out_name}_legend.svg")
        plt.savefig(out_fn)

    def save_altair_chart(self, out_name="interactive_tsne"):
        self.chart_.properties(width=800, height=800).interactive().save(
            opjoin(self.out_root_, f"{out_name}.html"))
        self.chart_.save(opjoin(self.out_root_, f"{out_name}.json"))
        #self.chart_.save(f"{exp_hash}.png")
        #self.chart_.save(f"{exp_hash}.svg")

    def make_interactive_tsne(self):
        smpl_pred_ = self.sample_predictions(
            which_set=self.cf_["testing"]["tsne"]["which_set"],
            n_sample=self.cf_["testing"]["tsne"]["num_patches"])
        smpl_pred_ = self.cat_predictions_dict(smpl_pred_)

        smpl_pred_ = self.make_sample_df(smpl_pred_)
        smpl_pred_ = self.make_sample_tsne(smpl_pred_)

        self.make_tsne_matplotlib(smpl_pred_, color_key="class_name")
        self.save_matplotlib_chart()
        for i in self.cf_["data"].get("one_v_all_index", []):
            cn = self.cf_["data"]["test_dataset"]["camera_ready_classes"][i]
            self.make_tsne_matplotlib(smpl_pred_,
                                      color_key="class_name",
                                      highlight={cn})
            self.save_matplotlib_chart(out_name=f"{cn}_tsne")

        if self.cf_["testing"]["tsne"]["interactive"]:
            self.make_tsne_altair(smpl_pred_)
            self.save_altair_chart()

    @staticmethod
    def get_device_str() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def get_emb_sim(self,
                    query_feat: torch.Tensor,
                    tgt_feat: torch.Tensor,
                    mask: Optional[torch.Tensor] = None):

        device = self.get_device_str()
        query_feat, tgt_feat = query_feat.to(device), tgt_feat.to(device)
        if mask is not None: mask = mask.to(device)

        sim = (torch.nn.functional.normalize(query_feat, p=2, dim=1)
               @ torch.nn.functional.normalize(tgt_feat, p=2, dim=1).T)
        if mask is not None: sim[mask.T] = 0

        logging.info(f"Sim matrix shape {sim.shape}")
        return sim.argsort(
            dim=1, descending=True)[:, :self.cf_["testing"]["nn_viz"]["k"]]

    def make_nn(self):
        query_pred = self.sample_predictions(
            which_set=[self.cf_["testing"]["nn_viz"]["query_set"]],
            n_sample=[
                self.cf_["testing"]["nn_viz"]["n_sample"]
            ])[self.cf_["testing"]["nn_viz"]["query_set"]].sort_index()
        tgt_pred = self.preds_[self.cf_["testing"]["nn_viz"]["target_set"]]
        query_feat = torch.tensor([d for d in query_pred["embeddings"]])
        tgt_feat = torch.tensor([d for d in tgt_pred["embeddings"]])

        if self.cf_["testing"]["nn_viz"]["exclude_same_patient"]:
            query_pt = np.expand_dims(
                query_pred["path"].apply(self.get_patient).to_numpy(), 0)
            tgt_pt = np.expand_dims(
                tgt_pred["path"].apply(self.get_patient).to_numpy(), 1)
            mask = torch.tensor(tgt_pt == query_pt)
        else:
            mask = None

        sim = self.get_emb_sim(query_feat, tgt_feat, mask=mask)

        padding_func = lambda x: np.pad(np.array(x),
                                        pad_width=((2, 2), (2, 2), (0, 0)),
                                        constant_values=255)

        im_xms = np.vstack(
            [padding_func(self.get_im_impl_(x)) for x in query_pred["path"]])
        im_divider = np.ones_like(im_xms) * 255
        im_nns = np.vstack([
            np.hstack([
                padding_func(self.get_im_impl_(tgt_pred["path"].iloc[int(j)]))
                for j in i
            ]) for i in tqdm(sim, desc="load NN image")
        ])
        out = np.hstack([im_xms, im_divider, im_nns])
        imageio.imwrite(opjoin(self.out_root_, "nn.png"), out.astype(np.uint8))

    def fit_clustering(self, smpl_pred, k):
        #dbs = DBSCAN(eps=k, metric="cosine"
        #           ).fit(self.smpl_pred_["embeddings"])

        #dbs = SpectralClustering(n_clusters=k,
        #                         assign_labels='discretize',
        #                         random_state=1000).fit(
        #                             self.smpl_pred_["embeddings"])

        dbs = KMeans(n_clusters=k).fit(smpl_pred)
        if len(set(dbs.labels_.tolist())) == 1:
            silhouette_avg = 0
        else:
            silhouette_avg = silhouette_score(smpl_pred, dbs.labels_).item()

        return {
            "assignment": dbs.labels_.tolist(),
            "silhouette": silhouette_avg
        }

    def make_sample_cluster(self):
        raise NotImplementedError()
        smpl_pred = self.preds_[self.cf_["testing"]["tsne"]["which_set"]]
        smpl_emb = np.array([x for x in smpl_pred["embeddings"]])
        k_vals = np.arange(10, 30).tolist()
        cluster_results = {
            k: self.fit_clustering(smpl_emb, k)
            for k in tqdm(k_vals, desc="make cluster")
        }
        all_cluster_assignments = pd.DataFrame.from_dict(
            {
                k: cluster_results[k]["assignment"]
                for k in cluster_results
            },
            orient="columns").set_index(smpl_pred["path"]).reset_index()
        all_silhouette_scores = {
            k: cluster_results[k]["silhouette"]
            for k in cluster_results
        }
        best_k = max(all_silhouette_scores.items(), key=lambda x: x[1])[0]

        logging.info(f"clustering - all_scores = {all_silhouette_scores}")
        logging.info(f"clustering - best k = {best_k}")

        all_cluster_assignments.to_csv(opjoin(self.out_root_,
                                              "cluster_assignments.csv"),
                                       index=False)
        with open(opjoin(self.out_root_, "cluster_silhouette_scores.csv"),
                  "w") as fd:
            json.dump(all_silhouette_scores, fd)

        smpl_pred = self.make_sample_tsne(smpl_pred)

        for k in tqdm(k_vals, desc="cluster visual"):
            smpl_pred[f"cluster_assignment_{k}"] = all_cluster_assignments[k]
            self.classes_[f"cluster_assignment_{k}"] = np.sort(
                np.unique(all_cluster_assignments[k]))

            im_paths = [
                g["path"].sample(
                    30, replace=True,
                    random_state=np.random.default_rng(1000)).tolist()
                for _, g in smpl_pred.groupby(f"cluster_assignment_{k}")
            ]

            padding_func = lambda x: np.pad(np.array(x),
                                            pad_width=((2, 2), (2, 2), (0, 0)),
                                            constant_values=255)

            im = np.vstack([
                np.hstack([padding_func(self.get_im_impl_(j)) for j in i])
                for i in im_paths
            ])
            imageio.imwrite(opjoin(self.out_root_, f"clusters_{k}.png"),
                            im.astype(np.uint8))

            self.make_tsne_matplotlib(smpl_data_=smpl_pred,
                                      color_key=f"cluster_assignment_{k}")
            self.save_matplotlib_chart(out_name=f"interactive_tsne_{k}")

            self.fig_ = None
            self.ax_ = None
            self.chart_ = None


class MNISTEvalModule(EvalBaseModule):

    def __init__(self, cf, out_root, is_knn, im_required=False):
        super(MNISTEvalModule, self).__init__(cf, out_root, is_knn)
        logging.info(f"Starting {self.__class__.__name__}")

    def replace_patch_name(self, in_str: str) -> str:
        return in_str

    def reorder_classes(self, array_in: np.ndarray) -> np.ndarray:
        return array_in

    def get_dataset(self):
        if self.cf_["data"]["set"].lower() not in ["mnist", "cifar10"]:
            raise NotImplementedError()

        if self.cf_["data"]["set"] in ["cifar10"]:
            base_aug_func = partial(
                get_vision_base_aug,
                imnet_norm=self.cf_["data"]["params"]["imagenet_norm"])
        else:
            base_aug_func = get_vision_base_aug

        mnist_params = {
            "data_root": self.cf_["data"]["params"]["data_dir"],
            "transform": get_transformations({}, base_aug_func)[-1],
            "n_transforms": 1,
            "balance_instance_class": False
        }
        dset_func = {
            "mnist": MNISTDataset,
            "cifar10": CIFAR10Dataset
        }[self.cf_["data"]["set"].lower()]

        self.data_instances_ = {
            i.name: i.im
            for i in dset_func(train_val="train", **mnist_params).instances_ +
            dset_func(train_val="valid", **mnist_params).instances_
        }

    def get_im_impl_(self, im_path: str):
        im = self.data_instances_[im_path].numpy().astype(np.uint8).squeeze()
        if len(im.shape) == 3: im = im.transpose([1, 2, 0])
        return im

    def get_im(self, im_row):
        return self.im_to_bytestr(
            Image.fromarray(self.get_im_impl_(im_row["path"])))

    def get_patient(self, x):
        return x

    def get_slide(self, x):
        return x

    def get_patch_name(self, x):
        return x

    def get_institution(self, x):
        return x

    def compute_metrics(self):
        pred = self.val_preds_
        if not self.is_knn_:
            logging.info("applying softmax for metrics")
            pred["logits"] = pred["logits"].apply(
                lambda x: torch.nn.functional.softmax(torch.tensor(x), dim=0
                                                      ).tolist())
        print(pred)

        one_v_all_idx = self.cf_.data.test_dataset.get("one_v_all_index", [])
        label_l = torch.tensor(pred["labels"])
        logits_l = torch.tensor(pred["logits"])
        metrics_l, metrics_names_l = self.get_all_metrics(
            logits_l, label_l, one_v_all_idx=one_v_all_idx)
        cm_l = confusion_matrix(y_true=label_l,
                                y_pred=logits_l.argmax(dim=1)).tolist()
        all_metrics = {"metrics": metrics_l, "cm": cm_l}

        for i in one_v_all_idx:
            pred_i = logits_l.argmax(dim=1) == i
            gt_i = label_l == i
            cn = self.cf_["data"]["test_dataset"]["camera_ready_classes"][i]
            all_metrics[f"{cn}_cm"] = confusion_matrix(y_true=gt_i,
                                                       y_pred=pred_i).tolist()

        # print out metrics
        all_metrics_df = pd.DataFrame(all_metrics["metrics"],
                                      index=metrics_names_l,
                                      columns=["metric"]).T
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        logging.info(f"all metrics\n{str(all_metrics_df)}")

        # draw conf matrix (images)
        self.plot_confusion(all_metrics["cm"],
                            opjoin(self.out_root_, "conf.svg"))
        logging.info(f"conf matrix\n{str(np.array(all_metrics['cm']))}")

        for i in one_v_all_idx:
            cn = self.cf_["data"]["test_dataset"]["camera_ready_classes"][i]
            self.plot_confusion(all_metrics[f"{cn}_cm"],
                                opjoin(self.out_root_, f"{cn}_conf.svg"),
                                nc=2,
                                class_names=["Others", cn])
            logging.info(
                f"{cn}_conf matrix\n{str(np.array(all_metrics[f'{cn}_cm']))}")

        # save metrics
        all_metrics_df.to_csv(opjoin(self.out_root_, "metrics.csv"),
                              index=False)
        with open(opjoin(self.out_root_, "all_metrics.json"), "w") as fd:
            json.dump(all_metrics, fd)

    def __call__(self, preds):
        self.process_predictions(preds)
        self.get_dataset()
        self.visualize_feature_rank()
        if self.nc_ > 1:
            self.compute_metrics()
        if "tsne" in self.cf_["testing"]:
            self.make_interactive_tsne()
        if "nn_viz" in self.cf_["testing"]:
            self.make_nn()


class SRHEvalModule(EvalBaseModule):

    def __init__(self, cf, out_root, is_knn, im_required=False):
        super(SRHEvalModule, self).__init__(cf, out_root, is_knn=is_knn)

        if self.cf_["data"]["set"] == "he":
            eval_mode = cf["testing"].get("eval_mode", "patch_knn")
            if eval_mode in {
                    "slide_mil", "slide_mil_knn", "slide_avg_pool",
                    "slide_max_pool", "slide_clam", "slide_mil_region_mean"
            }:
                self.pt_str_id_ = -3
                self.inst_str_id_ = -4
                assert not im_required
            else:
                self.pt_str_id_ = -4
                self.inst_str_id_ = -5
                if im_required:
                    self.transform_ = get_transformations({},
                                                          get_he_base_aug)[-1]
                    self.get_im_impl_ = self.get_im_tcga
                    self.get_im_path_impl_ = None
            inst_split_char = "-"
        elif self.cf_["data"]["set"] in {"srh"}:
            eval_mode = cf["testing"].get("eval_mode", "patch_knn")
            if eval_mode in {
                    "slide_mil", "slide_mil_knn", "slide_avg_pool",
                    "slide_max_pool", "slide_clam"
            }:
                self.pt_str_id_ = -3
                self.inst_str_id_ = -4
                assert not im_required
            else:
                self.pt_str_id_ = -4
                self.inst_str_id_ = -5
                if im_required:
                    self.transform_ = get_transformations({},
                                                          get_srh_base_aug)[-1]
                    self.get_im_impl_ = self.get_im_srh
                    self.get_im_path_impl_ = self.get_srh_rgb_cache_path
            inst_split_char = "_"

        elif self.cf_["data"]["set"] == "scsrh":
            raise NotImplementedError()
            self.pt_str_id_ = -5
            self.inst_str_id_ = -6
            if im_required:
                self.transform_ = get_transformations({}, get_srh_base_aug)[-1]
                assert "h5_data_dir" in self.cf_["data"], (
                    "SRH single cell image generation require numpy cache")
                self.get_im_impl_ = self.get_im_scsrh
                self.get_im_path_impl_ = self.get_scsrh_rgb_cache_path
            inst_split_char = "_"
        else:
            raise ValueError("data/set must be in [scsrh, he, srh]")

    def compute_metrics(self):
        pred = self.val_preds_
        if not self.is_knn_:
            logging.info("applying softmax for metrics")
            pred["logits"] = pred["logits"].apply(
                lambda x: torch.nn.functional.softmax(torch.tensor(x), dim=0
                                                      ).tolist())

        # add patient and slide info from patch paths # only NIO images in eval metrics
        eval_mode = self.cf_["testing"].get("eval_mode", "patch_knn")

        if eval_mode in {"patch", "patch_knn"}:
            agg_functions = {
                "patch": None,
                "slide": self.get_slide,
                "patient": self.get_patient
            }
        elif eval_mode in {
                "slide_mil", "slide_avg_pool", "slide_max_pool",
                "slide_mil_knn", "slide_clam", "slide_mil_region_mean"
        }:
            agg_functions = {"slide": None, "patient": self.get_patient}
        else:
            raise ValueError(
                f"eval_mode expected to be one of " +
                "\{patch, slide_mil, slide_avg_pool, slide_max_pool\}, " +
                f"got {eval_mode}")

        for l in agg_functions.keys():
            if agg_functions[l] is not None:
                pred[l] = pred["path"].apply(agg_functions[l])

        print(pred)
        print(pred["patient"])

        # exclude bad patients from evaluation
        patient_exclude = [
            "NIO_NYU_66", "NIO_UM_936", "NIO_UM_936b", "NIO_UM_868",
            "NIO_UM868b"
        ]
        pred = pred[~pred["patient"].isin(patient_exclude)].reset_index()

        # aggregate logits
        get_agged_logits = lambda pred, mode: pd.DataFrame(
            pred.groupby(by=[mode, "labels"])["logits"].apply(
                lambda x: [sum(y) for y in zip(*x)])).reset_index()
        normalize_f = lambda x: torch.nn.functional.normalize(
            torch.tensor(np.vstack(x)), dim=1, p=1)

        neg_class_idx = self.cf_.data.test_dataset.negative_class_index
        one_v_all_idx = self.cf_.data.test_dataset.get("one_v_all_index", [])

        def get_metrics_level(l):

            pred_l = get_agged_logits(pred, l) if agg_functions[l] else pred
            label_l = torch.tensor(pred_l["labels"])
            logits_l = normalize_f(pred_l["logits"])
            metrics_l, metrics_names_l = self.get_all_metrics(
                logits_l,
                label_l,
                neg_class_idx,
                one_v_all_idx=self.cf_.data.test_dataset.get("one_v_all_index", []))
            cm_l = confusion_matrix(y_true=label_l,
                                    y_pred=logits_l.argmax(dim=1)).tolist()

            all_metrics = {
                "metrics": metrics_l,
                "metric_names": metrics_names_l,
                "cm": cm_l
            }
            for i in one_v_all_idx:
                logging.critical("unverified implementation")
                pred_li = logits_l.argmax(dim=1) == i
                gt_li = label_l == i
                cn = self.cf_["data"]["test_dataset"]["camera_ready_classes"][
                    i]
                all_metrics[f"{cn}_cm"] = confusion_matrix(
                    y_true=gt_li, y_pred=pred_li).tolist()

            return all_metrics

        all_metrics = {l: get_metrics_level(l) for l in agg_functions.keys()}
        just_metrics = {l: all_metrics[l]["metrics"] for l in agg_functions}

        metric_names = all_metrics[list(
            agg_functions.keys())[0]]["metric_names"]
        # print out metrics
        all_metrics_df = pd.DataFrame(just_metrics, index=metric_names).T

        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        logging.info(f"all metrics\n{str(all_metrics_df)}")

        # draw conf matrix (images)
        make_conf_name = lambda x: opjoin(self.out_root_, f"{x}_conf.svg")
        for l in agg_functions:
            self.plot_confusion(all_metrics[l]["cm"], make_conf_name(l))
            logging.info(f"{l} conf matrix\n{str(all_metrics[l]['cm'])}")

            for i in one_v_all_idx:
                cn = self.cf_["data"]["test_dataset"]["camera_ready_classes"][
                    i]
                self.plot_confusion(all_metrics[l][f"{cn}_cm"],
                                    opjoin(self.out_root_, f"{cn}_conf.svg"),
                                    nc=2,
                                    class_names=["Others", cn])
                logging.info(
                    f"{l} {cn}_conf matrix\n{str(np.array(all_metrics[f'{cn}_cm']))}"
                )

        # save metrics
        all_metrics_df.to_csv(opjoin(self.out_root_, "metrics.csv"),
                              index=False)
        with open(opjoin(self.out_root_, "all_metrics.json"), "w") as fd:
            json.dump(all_metrics, fd)

    def replace_patch_name(self, in_str: str) -> str:
        drive_path = self.cf_["infra"]["drive_path"]
        in_str = in_str.replace("srh7pv2", "opensrh")

        for c in ["umms-tocho-snr", "umms-tocho-ns", "umms-tocho"]:
            if c in in_str:
                return in_str.replace(f"/nfs/turbo/{c}", drive_path)
        return in_str

    def reorder_classes(self, array_in: np.ndarray) -> np.ndarray:
        temp = torch.clone(array_in)
        for i, j in enumerate(
                self.cf_["data"]["test_dataset"]["classes_reorder"]):
            if i != j: temp[array_in == j] = i
        return temp

    def get_im_srh(self, im_path: str):
        im = self.transform_(process_read_srh(im_path))
        im = SRHRGBToolbox.viz_rescale_hist(im)
        return Image.fromarray(im)

    def get_im_scsrh(self, im_path: str):
        raise NotImplementedError()
        patch_ids = im_path.split("/")[-1].split("+")
        npath = opjoin(self.cf_["data"]["h5_data_dir"],
                       self.get_institution(im_path), self.get_slide(im_path),
                       "cells", self.cf_["data"]["sc_detection_model"],
                       patch_ids[0].removesuffix(".tif"))

        im = self.transform_(
            ProcessReadNPY(cell_crop_dim=self.cf_["data"]["im_size"])(
                npath, int(patch_ids[1])))
        im = SRHRGBToolbox.viz_rescale_hist(im)
        return Image.fromarray(im)

    def get_im_tcga(self, im_path: str):
        im = self.transform_(process_read_png(im_path))
        im = (255 * np.swapaxes(im.numpy(), 0, -1)).astype(np.uint8)
        return Image.fromarray(im)

    def get_srh_rgb_cache_path(self, im_path: str):
        return opjoin(self.cf_["data"]["rgb_data_dir"],
                      im_path.split("/")[-5], self.get_slide(im_path), "rgb",
                      im_path.split("/")[-1].replace('.tif', '.png'))

    def get_scsrh_rgb_cache_path(self, im_path: str):
        im_fname_split = im_path.split("/")[-1].split("+")
        im_name = im_fname_split[0].removesuffix(".tif")
        cell_id = im_fname_split[1]
        return opjoin(self.cf_["data"]["rgb_data_dir"],
                      self.get_institution(im_path), self.get_slide(im_path),
                      "rgb", "+".join([im_name, cell_id]) + ".webp")

    def get_im(self, im_row):
        if self.cf_["testing"]["tsne"].get("embed_images", False):
            return self.im_to_bytestr(self.get_im_impl_(im_row["path"]))
        else:
            return self.get_im_path_impl_(im_row["path"])

    def get_patient(self, x):
        if self.cf_["data"]["set"] == "he":
            return self.get_patient_tcga(x)
        else:
            return self.get_patient_srh(x)

    def get_patient_srh(self, x):
        return re.findall(f"(?:INV|NIO)_[a-zA-Z]+_[0-9]+",x.split("@")[0].split("-")[0])[0]

    def get_patient_tcga(self, x):
        return x.split("@")[0].split("-")[0]
        #return x.split("/")[self.pt_str_id_]

    def get_slide(self, x):
        return x.split("@")[0]
        #return "/".join(
        #    [x.split("/")[self.pt_str_id_],
        #     x.split("/")[self.pt_str_id_ + 1]])

    def get_patch_name(self, x):
        return x
        #return x.split("/")[-1].replace(".tif", "")

    def get_institution(self, x):
        raise NotImplementedError
        return x.split("/")[self.inst_str_id_]

    def __call__(self, preds):

        self.process_predictions(preds)

        #self.visualize_feature_rank()
        if self.nc_ > 1:
            self.compute_metrics()
        if "tsne" in self.cf_.testing:
            self.make_interactive_tsne()
        if "nn_viz" in self.cf_.testing:
            self.make_nn()
        #self.make_sample_cluster()


def do_eval(cf, out_root, predictions, is_knn=True):
    if cf["data"]["set"] in ["mnist", "cifar10"]:
        em = MNISTEvalModule(cf, out_root, is_knn=is_knn, im_required=True)

    elif cf["data"]["set"] in ["scsrh", "he", "srh"]:
        imreq = cf.testing.get("eval_mode",
                               "patch_knn") in {"patch", "patch_knn"}
        em = SRHEvalModule(cf, out_root, is_knn=is_knn, im_required=imreq)
    else:
        raise ValueError("data/set must be in [mnist, he, srh]")

    for k in ["train", "val", "xmplr"]:
        if k not in predictions:
            predictions[k] = None
    em(predictions)


def main():
    config, out_dir, preds = setup_eval_module_standalone_infra(
        get_xmplr_pred=True)
    do_eval(config, out_dir, preds)


if __name__ == "__main__":
    main()
