import io
import re
import os
import json
import base64
import logging
import argparse
from os.path import join as opj
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Any, Optional, Tuple
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

import einops

from torchsrh.train.infra import read_cf
from torchsrh.utils.open_color import OpenColor
from torchsrh.utils.rgb_srh import SRHRGBToolbox
from torchsrh.datasets.mnist_dataset import MNISTDataset
from torchsrh.datasets.cifar_dataset import CIFAR10Dataset

from torchsrh.datasets.db_improc import (get_srh_base_aug, process_read_srh,
                                         process_read_png, ProcessReadNPY,
                                         get_transformations, get_he_base_aug,
                                         get_vision_base_aug)
from torchsrh.eval.infra import setup_eval_module_standalone_infra

from ts2.data.transforms import HistologyTransform
from ts2.data.db_improc import instantiate_process_read
from ts2.data.meta_parser import CachedCSVParser

tqdm.pandas()
plt.rcParams["pdf.fonttype"] = 42


class EvalBaseModule(ABC):

    def __init__(self, cf, out_root, do_softmax):
        self.cf_ = cf
        self.out_root_ = out_root
        self.do_softmax_ = do_softmax
        self.visualize_xmplrs_ = self.cf_.get("nn_viz", {}).get(
            "do_nn_viz", None) and ("exemplars" in self.cf_.get("data", {}))
        self.nc_ = len(
            self.cf_["data"]["test_dataset"]["camera_ready_classes"])
        self.class_names_ = np.array(
            self.cf_["data"]["test_dataset"]["camera_ready_classes"])[
                cf["data"]["test_dataset"]["classes_reorder"]]
        self.classes_ = {}
        logging.info(f"{self.nc_} class eval")

    def process_predictions(self, preds):

        def do_process_one_set(pred):
            if not pred: return None

            if "logits" not in pred:
                pred["logits"] = torch.zeros((len(pred["label"]), self.nc_))

            return pd.DataFrame({
                "logits":
                pred["logits"]
                [:,
                 self.cf_["data"]["test_dataset"]["classes_reorder"]].tolist(),
                "embeddings":
                pred["embeddings"].tolist(),
                "labels":
                self.reorder_classes(pred["label"]).tolist(),
                "pred":
                self.reorder_classes(pred["logits"].argmax(dim=1)).tolist(),
                "path":
                pred["path"]
            }).sort_values(by="path", ignore_index=True)

        for k in preds:
            preds[k] = do_process_one_set(preds[k])

        return preds

    def sample_predictions(
        self,
        preds: Dict[str, pd.DataFrame],
        which_sets: Tuple[str] = ("val", ),
        n_samples: Tuple[int] = (0, )
    ) -> Dict[str, pd.DataFrame]:

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
            k: sample_df(preds[k], num_sample=n)
            for k, n in zip(which_sets, n_samples)
        }

    def visualize_feature_rank(self, preds: pd.DataFrame):
        emb_dim = len(preds.iloc[0]["embeddings"])
        val_embs = torch.tensor(
            self.sample_predictions(
                {"val": preds}, which_sets=("val", ),
                n_samples=(emb_dim, ))["val"]["embeddings"].tolist())

        logging.info(val_embs.shape)

        s = torch.linalg.svdvals(val_embs)
        r = torch.linalg.matrix_rank(val_embs)

        _, ax = plt.subplots(1, 1, figsize=(16 / 3, 4), dpi=300)
        ax.plot(np.arange(len(s)), s, clip_on=False)
        ax.set_yscale("log")
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        ax.set_ylabel("Singular values (log scale)")
        ax.set_xlabel("Singular value order")
        ax.grid(axis="both", c="lightgrey", zorder=0)
        ax.set_ylim([
            torch.pow(10, torch.floor(torch.log10(s[-1]))),
            torch.pow(10, torch.ceil(torch.log10(s[0])))
        ])
        ax.set_xlim([0, len(s)])
        ax.set_xticks(np.arange(0, len(s) + 1, len(s) // 8))
        ax.set_title(f"rank = {r}")

        for spine in ax.spines.values():
            spine.set_edgecolor("lightgrey")
            spine.set_zorder(0)

        plt.tight_layout()
        plt.savefig(opj(self.out_root_, "val_emb_rank.png"))
        plt.savefig(opj(self.out_root_, "val_emb_rank.pdf"))

        #with gzip.open(opj(self.out_root_, "val_emb_rank.pt.gz"),
        #               "w") as fd:
        #    torch.save(s, fd)

    @abstractmethod
    def reorder_classes(self, array_in: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def get_im(self, im_row):
        raise NotImplementedError()

    @abstractmethod
    def get_patient_id(self, x):
        raise NotImplementedError()

    @abstractmethod
    def get_slide_id(self, x):
        raise NotImplementedError()

    @abstractmethod
    def get_patch_id(self, x):
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

            acc_val = acc(logits, label)
            t2_val = t2(logits, label)
            t3_val = t3(logits, label)
            mca_val = mca(logits, label)
            map_val = auprc(logits, label)
            auroc_val = auroc(logits, label)
            sen_val = sen(logits, label)
            spec_val = spec(logits, label)
            f1_val = f1_macro(logits, label)

        metrics = [
            acc_val, t2_val, t3_val, mca_val, map_val, auroc_val, sen_val,
            spec_val, f1_val
        ]
        metric_names = [
            "acc", "t2", "t3", "mca", "map", "auroc", "sen", "sepc", "f1"
        ]

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
        #plt.rcParams.update({'font.size': 18})
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

    def process_sample_df(self, smpl_pred_):

        if self.visualize_xmplrs_:
            for k in smpl_pred_:
                smpl_pred_[k]["is_xmplr"] = (k == "xmplr")

        smpl_pred_ = pd.concat([smpl_pred_[k] for k in smpl_pred_])

        smpl_pred_["patient"] = smpl_pred_["path"].apply(self.get_patient_id)
        smpl_pred_["slide"] = smpl_pred_["path"].apply(self.get_slide_id)
        smpl_pred_["patch"] = smpl_pred_["path"].apply(self.get_patch_id)

        if self.cf_.testing.tsne.interactive:
            smpl_pred_["image"] = smpl_pred_.progress_apply(self.get_im,
                                                            axis=1)

        self.classes_["class_name"] = [
            self.cf_.data.test_dataset.camera_ready_classes[i]
            for i in self.cf_.data.test_dataset.classes_reorder
        ]

        smpl_pred_["class_name"] = smpl_pred_["labels"].apply(
            lambda x: self.class_names_[x])
        return smpl_pred_

    def compute_sample_tsne(self, data):
        tsne = TSNE(n_components=2,
                    **self.cf_["testing"]["tsne"]["class_params"])
        xy = tsne.fit_transform(np.array([d for d in data["embeddings"]]))

        min_vals = xy.min(axis=0)
        max_vals = xy.max(axis=0)
        xy = (xy - min_vals) / (max_vals - min_vals)
        xy = xy * 1.8 - 0.9

        data["x"] = xy[:, 0].tolist()
        data["y"] = xy[:, 1].tolist()
        return data

    def make_tsne_altair(self,
                         smpl_data,
                         color_key="class_name",
                         out_name="interactive_tsne"):
        smpl_data_ = smpl_data.drop(["logits", "embeddings"], axis=1)
        alt.data_transformers.disable_max_rows()
        rgb = OpenColor.setup_colors(len(set(smpl_data_[color_key].tolist())),
                                     ind=8)
        color_scale = alt.Scale(domain=self.classes_[color_key],
                                range=rgb.tolist())
        if self.visualize_xmplrs_:
            smpl_data_ = smpl_data_.sort_values(by="is_xmplr")
            chart_encode_params = {
                "shape":
                alt.Shape(
                    "is_xmplr",
                    scale=alt.Scale(
                        domain=[False, True],
                        range=["circle", "diamond"]
                    )
                ),
                "strokeWidth":
                alt.Stroke(
                    "is_xmplr",
                    scale=alt.Scale(
                        domain=[False, True],
                        range=[0, 3]
                    )
                )
            } # yapf: disable
        else:
            chart_encode_params = {"strokeWidth": alt.value(0)}

        tsne_unit_axis = alt.Axis(tickSize=0,
                                  values=[-1, -0.6, -0.2, 0.2, 0.6, 1],
                                  domain=False,
                                  labels=False,
                                  title="")
        chart = alt.Chart(smpl_data_).mark_point(filled=True).encode(
            x=alt.X("x", scale=alt.Scale(domain=[-1, 1]), axis=tsne_unit_axis),
            y=alt.Y("y", scale=alt.Scale(domain=[-1, 1]), axis=tsne_unit_axis)
        )  # yapf: disable

        chart_static = chart.encode(
            stroke=alt.value("#000000"),
            size=alt.value(16),
            color=alt.Color(f"{color_key}:N", scale=color_scale, legend=None),
            opacity=alt.value(0.9),
            **chart_encode_params
        ).properties(
            width=512,
            height=512
        )  # yapf: disable

        chart_static.save(opj(self.out_root_, f"{out_name}.pdf"))
        chart_static.save(opj(self.out_root_, f"{out_name}.png"),
                          scale_factor=2.0)

        chart_static.encode(
            color=alt.Color(f"{color_key}:N", scale=color_scale)
        ).save(opj(self.out_root_, f"{out_name}_legend.pdf")) # yapf: disable

        if self.cf_.testing.tsne.interactive:
            bind_range = alt.binding_range(min=4, max=256, name='Size ')
            param_size = alt.param(name='point_size', bind=bind_range, value=64)

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

            chart_i = alt.Chart(smpl_data_).mark_point(filled=True,
                size=param_size
            ).encode(
                x=alt.X("x", scale=alt.Scale(domain=[-1, 1]), axis=tsne_unit_axis),
                y=alt.Y("y", scale=alt.Scale(domain=[-1, 1]), axis=tsne_unit_axis),
                tooltip=['image', "path"],
                color=alt.condition(
                    patient_selection & slide_selection & class_selection,
                    alt.Color(f"{color_key}:N", scale=color_scale),
                    alt.value('lightgray')
                ),
                opacity=alt.condition(
                    patient_selection & slide_selection & class_selection,
                    alt.value(1.0),
                    alt.value(0.1)
                ),
                stroke=alt.value("#000000"),
                **chart_encode_params
            ).add_params(
                param_size,
                patient_selection,
                slide_selection,
                class_selection
            ) # yapf: disable

            chart_i.properties(width=768, height=768).interactive().save(
                opj(self.out_root_, f"{out_name}.html"))
            chart_i.save(opj(self.out_root_, f"{out_name}.json"))

    def make_interactive_tsne(self, preds):
        self.make_tsne_altair(
            self.compute_sample_tsne(
                self.process_sample_df(
                    self.sample_predictions(
                        preds,
                        which_sets=self.cf_["testing"]["tsne"]["which_set"],
                        n_samples=self.cf_["testing"]["tsne"]
                        ["num_patches"]))))

    @staticmethod
    def get_device_str() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def compute_emb_sim_top_k(query_feat: torch.Tensor,
                              tgt_feat: torch.Tensor,
                              k: int,
                              mask: Optional[torch.Tensor] = None):

        device = EvalBaseModule.get_device_str()
        query_feat, tgt_feat = query_feat.to(device), tgt_feat.to(device)
        if mask is not None: mask = mask.to(device)

        sim = (torch.nn.functional.normalize(query_feat, p=2, dim=1)
               @ torch.nn.functional.normalize(tgt_feat, p=2, dim=1).T)
        if mask is not None: sim[mask.T] = 0

        logging.info(f"Sim matrix shape {sim.shape}")
        return sim.argsort(dim=1, descending=True)[:, :k]

    def make_nn(self, preds):
        query_set = self.cf_.testing.nn_viz.query_set
        n_query_samples = self.cf_.testing.nn_viz.n_sample
        query_pred = self.sample_predictions(
            preds, which_sets=[query_set],
            n_samples=[n_query_samples])[query_set].sort_index()
        tgt_pred = preds[self.cf_.testing.nn_viz.target_set]

        query_feat = torch.tensor([d for d in query_pred["embeddings"]])
        tgt_feat = torch.tensor([d for d in tgt_pred["embeddings"]])

        if self.cf_.testing.nn_viz.exclude_same_patient:
            query_pt = np.expand_dims(
                query_pred["path"].apply(self.get_patient_id).to_numpy(), 0)
            tgt_pt = np.expand_dims(
                tgt_pred["path"].apply(self.get_patient_id).to_numpy(), 1)
            mask = torch.tensor(tgt_pt == query_pt)
        else:
            mask = None

        sim = self.compute_emb_sim_top_k(query_feat,
                                         tgt_feat,
                                         k=self.cf_.testing.nn_viz.k,
                                         mask=mask)

        def padding_func(x):
            return np.pad(np.array(x),
                          pad_width=((2, 2), (2, 2), (0, 0)),
                          constant_values=255)

        im_xms = np.vstack([
            padding_func(self.get_im_impl_(x))
            for _, x in query_pred.iterrows()
        ])
        im_divider = np.ones_like(im_xms) * 255
        im_nns = np.vstack([
            np.hstack([
                padding_func(self.get_im_impl_(tgt_pred.iloc[int(j)]))
                for j in i
            ]) for i in tqdm(sim, desc="load NN image")
        ])
        out = np.hstack([im_xms, im_divider, im_nns])
        imageio.imwrite(opj(self.out_root_, "nn.png"), out.astype(np.uint8))

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

        all_cluster_assignments.to_csv(opj(self.out_root_,
                                           "cluster_assignments.csv"),
                                       index=False)
        with open(opj(self.out_root_, "cluster_silhouette_scores.csv"),
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
            imageio.imwrite(opj(self.out_root_, f"clusters_{k}.png"),
                            im.astype(np.uint8))

            self.make_tsne_matplotlib(smpl_data_=smpl_pred,
                                      color_key=f"cluster_assignment_{k}")
            self.save_matplotlib_chart(out_name=f"interactive_tsne_{k}")

            self.fig_ = None
            self.ax_ = None


class MNISTEvalModule(EvalBaseModule):

    def __init__(self, **kwargs):
        super(MNISTEvalModule, self).__init__(**kwargs)
        logging.info(f"Starting {self.__class__.__name__}")

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

    def get_patient_id(self, x):
        return x

    def get_slide_id(self, x):
        return x

    def get_patch_id(self, x):
        return x

    def compute_metrics(self):
        pred = self.val_preds_
        if self.do_softmax_:
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
        self.plot_confusion(all_metrics["cm"], opj(self.out_root_, "conf.svg"))
        logging.info(f"conf matrix\n{str(np.array(all_metrics['cm']))}")

        for i in one_v_all_idx:
            cn = self.cf_["data"]["test_dataset"]["camera_ready_classes"][i]
            self.plot_confusion(all_metrics[f"{cn}_cm"],
                                opj(self.out_root_, f"{cn}_conf.svg"),
                                nc=2,
                                class_names=["Others", cn])
            logging.info(
                f"{cn}_conf matrix\n{str(np.array(all_metrics[f'{cn}_cm']))}")

        # save metrics
        all_metrics_df.to_csv(opj(self.out_root_, "metrics.csv"), index=False)
        with open(opj(self.out_root_, "all_metrics.json"), "w") as fd:
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

    def __init__(self, **kwargs):
        super(SRHEvalModule, self).__init__(**kwargs)
        require_pixels = self.cf_.testing.eval_mode in {"patch", "patch_knn"}

        if self.cf_["data"]["set"] == "he":
            eval_mode = self.cf_.testing.get("eval_mode", "patch_knn")
            if eval_mode in {
                    "slide_mil", "slide_mil_knn", "slide_avg_pool",
                    "slide_max_pool", "slide_clam", "slide_mil_region_mean"
            }:
                self.pt_str_id_ = -3
                self.inst_str_id_ = -4
                assert not require_pixels
            else:
                self.pt_str_id_ = -4
                self.inst_str_id_ = -5
                if require_pixels:
                    self.transform_ = get_transformations({},
                                                          get_he_base_aug)[-1]
                    self.get_im_impl_ = self.get_im_tcga
                    self.get_im_path_impl_ = None
        elif self.cf_["data"]["set"] in {"srh"}:
            eval_mode = self.cf_["testing"].get("eval_mode", "patch_knn")
            if eval_mode in {
                    "slide_mil", "slide_mil_knn", "slide_avg_pool",
                    "slide_max_pool", "slide_clam"
            }:
                self.pt_str_id_ = -3
                self.inst_str_id_ = -4
                assert not require_pixels
            else:
                self.pt_str_id_ = -4
                self.inst_str_id_ = -5
                if require_pixels:
                    self.transform_ = get_transformations({},
                                                          get_srh_base_aug)[-1]
                    self.get_im_impl_ = self.get_im_srh
                    self.get_im_path_impl_ = self.get_srh_rgb_cache_path

        else:
            raise ValueError("data/set must be in [scsrh, he, srh]")

    def compute_metrics(self, pred):
        if self.do_softmax_:
            logging.info("applying softmax for metrics")
            pred["logits"] = pred["logits"].apply(
                lambda x: torch.nn.functional.softmax(torch.tensor(x), dim=0
                                                      ).tolist())

        # add patient and slide info from patch paths # only NIO images in eval metrics
        eval_mode = self.cf_["testing"].get("eval_mode", "patch_knn")

        if eval_mode in {"patch", "patch_knn"}:
            agg_functions = {
                "patch": None,
                "slide": self.get_slide_id,
                "patient": self.get_patient_id
            }
        elif eval_mode in {
                "slide_mil", "slide_avg_pool", "slide_max_pool",
                "slide_mil_knn", "slide_clam", "slide_mil_region_mean"
        }:
            agg_functions = {"slide": None, "patient": self.get_patient_id}
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
                one_v_all_idx=self.cf_.data.test_dataset.get(
                    "one_v_all_index", []))
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
        make_conf_name = lambda x: opj(self.out_root_, f"{x}_conf.svg")
        for l in agg_functions:
            self.plot_confusion(all_metrics[l]["cm"], make_conf_name(l))
            logging.info(f"{l} conf matrix\n{str(all_metrics[l]['cm'])}")

            for i in one_v_all_idx:
                cn = self.cf_["data"]["test_dataset"]["camera_ready_classes"][
                    i]
                self.plot_confusion(all_metrics[l][f"{cn}_cm"],
                                    opj(self.out_root_, f"{cn}_conf.svg"),
                                    nc=2,
                                    class_names=["Others", cn])
                logging.info(
                    f"{l} {cn}_conf matrix\n{str(np.array(all_metrics[f'{cn}_cm']))}"
                )

        # save metrics
        all_metrics_df.to_csv(opj(self.out_root_, "metrics.csv"), index=False)
        with open(opj(self.out_root_, "all_metrics.json"), "w") as fd:
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

    def get_im_tcga(self, im_path: str):
        im = self.transform_(process_read_png(im_path))
        im = (255 * np.swapaxes(im.numpy(), 0, -1)).astype(np.uint8)
        return Image.fromarray(im)

    def get_srh_rgb_cache_path(self, im_path: str):
        return opj(self.cf_["data"]["rgb_data_dir"],
                   im_path.split("/")[-5], self.get_slide_id(im_path), "rgb",
                   im_path.split("/")[-1].replace('.tif', '.png'))

    def get_im(self, im_row):
        if self.cf_["testing"]["tsne"].get("embed_images", False):
            return self.im_to_bytestr(self.get_im_impl_(im_row["path"]))
        else:
            return self.get_im_path_impl_(im_row["path"])

    def get_patient_id(self, x):
        if self.cf_["data"]["set"] == "he":
            return self.get_patient_id_tcga(x)
        else:
            return self.get_patient_id_srh(x)

    def get_patient_id_srh(self, x):
        return re.findall(f"(?:INV|NIO)_[a-zA-Z]+_[0-9]+",
                          x.split("@")[0].split("-")[0])[0]

    def get_patient_id_tcga(self, x):
        return x.split("@")[0].split("-")[0]
        #return x.split("/")[self.pt_str_id_]

    def get_slide_id(self, x):
        return x.split("@")[0]
        #return "/".join(
        #    [x.split("/")[self.pt_str_id_],
        #     x.split("/")[self.pt_str_id_ + 1]])

    def get_patch_id(self, x):
        return x
        #return x.split("/")[-1].replace(".tif", "")

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


class SRHCellImageGetter():

    def __init__(self, cf):
        super().__init__()
        curr_base_aug_params = cf.data.transform.test.params.base_aug_params
        curr_base_aug_params["to_uint8"] = True
        curr_base_aug_params["mask_params"]["how_to_process"]="small_patch"
        self.transform_ = HistologyTransform(
            which_set="scsrh",
            base_aug_params=curr_base_aug_params,
            strong_aug_params={
                "aug_list": [],
                "aug_prob": 0
            })

        self.process_read_ = instantiate_process_read(
            which=cf.data.test_dataset.which_process_read,
            which_set=cf.data.set)

        self.all_tsm_ = {}
        for k in cf.data.parser.params.cached_parser_file:
            self.all_tsm_.update(
                CachedCSVParser(
                    cf.data.parser.params.cached_parser_file[k])()[-1])
        self.data_root_ = cf.data.test_dataset.params.data_root

    def make_im_path(self, x):
        return opj(self.data_root_, x)

    def get_im(self, im_row):
        mmap_info = self.all_tsm_[im_row["slide"]]
        mmap_path = self.make_im_path(mmap_info["path"])
        im = self.process_read_(mmap_path, tuple(mmap_info["shape"]),
                                int(im_row["path"].split("@")[-1]))
        return einops.rearrange(self.transform_(im), "c h w -> h w c").numpy()

class SCBenchImageGetter():

    def __init__(self, cf):
        super().__init__()
        curr_base_aug_params = cf.data.transform.test.params.base_aug_params
        curr_base_aug_params["to_uint8"] = True
        curr_base_aug_params["mask_params"]["how_to_process"]="small_patch"
        self.transform_ = HistologyTransform(
            which_set="scsrh",
            base_aug_params=curr_base_aug_params,
            strong_aug_params={
                "aug_list": [],
                "aug_prob": 0
            })

        data_root = cf.data.test_dataset.params.data_root
        slides = pd.read_csv(cf.data.test_dataset.params.slides_file)
        all_meta = []
        all_images = []
        all_paths = []

        for _, s in slides.iterrows():
            meta = pd.read_csv(f"{data_root}/scbench_processed/{s['mosaic']}.csv")
            all_meta.append(meta["annot_labels"])
            all_images.append(
                torch.load(f"{data_root}/scbench_processed/{s['mosaic']}.pt").to(torch.float))
            all_paths.extend([f"scbench.{s['ttype']}.{s['mosaic']}@{i}" for i in range(len(meta))])

        all_meta = pd.concat(all_meta)
        all_images = torch.cat(all_images)

        self.instances_ = {k:j
                           for j,k in zip(all_images, all_paths)}


    def get_im(self, im_row):

        return einops.rearrange(self.transform_(self.instances_[im_row["path"]]),
                                 "c h w -> h w c").numpy()

class SRHCellEvalModule(EvalBaseModule):

    def __init__(self, **kwargs):
        super(SRHCellEvalModule, self).__init__(**kwargs)

        if self.cf_.testing.tsne.interactive:
            if self.cf_.data.set == "scsrh":
                self.im_getter_ = SRHCellImageGetter(self.cf_)
            elif self.cf_.data.set == "scsrh_bench":
                self.im_getter_ = SCBenchImageGetter(self.cf_)
            else:
                raise ValueError()
        else:
            self.im_getter_ = None

    def compute_metrics(self, pred):
        if self.do_softmax_:
            logging.info("applying softmax for metrics")
            pred["logits"] = pred["logits"].apply(
                lambda x: torch.nn.functional.softmax(torch.tensor(x), dim=0
                                                      ).tolist())

        # add patient and slide info from patch paths # only NIO images in eval metrics
        agg_functions = {
            "cell": None,
            "slide": self.get_slide_id,
        }

        for l in agg_functions.keys():
            if agg_functions[l] is not None:
                pred[l] = pred["path"].apply(agg_functions[l])

        pred["patient"] = pred["path"].apply(self.get_patient_id)

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

        neg_class_idx = self.cf_.data.test_dataset.get("negative_class_index")
        one_v_all_idx = self.cf_.data.test_dataset.get("one_v_all_index", [])

        def get_metrics_level(l):

            pred_l = get_agged_logits(pred, l) if agg_functions[l] else pred
            label_l = torch.tensor(pred_l["labels"])
            logits_l = normalize_f(pred_l["logits"])
            metrics_l, metrics_names_l = self.get_all_metrics(
                logits_l, label_l, neg_class_idx, one_v_all_idx)
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
                cn = self.cf_.data.test_dataset.camera_ready_classes[i]
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
        make_conf_name = lambda x: opj(self.out_root_, f"{x}_conf.svg")
        for l in agg_functions:
            self.plot_confusion(all_metrics[l]["cm"], make_conf_name(l))
            logging.info(f"{l} conf matrix\n{str(all_metrics[l]['cm'])}")

            for i in one_v_all_idx:
                cn = self.cf_["data"]["test_dataset"]["camera_ready_classes"][
                    i]
                self.plot_confusion(all_metrics[l][f"{cn}_cm"],
                                    opj(self.out_root_, f"{cn}_conf.svg"),
                                    nc=2,
                                    class_names=["Others", cn])
                logging.info(
                    f"{l} {cn}_conf matrix\n{str(np.array(all_metrics[f'{cn}_cm']))}"
                )

        # save metrics
        all_metrics_df.to_csv(opj(self.out_root_, "metrics.csv"), index=False)
        with open(opj(self.out_root_, "all_metrics.json"), "w") as fd:
            json.dump(all_metrics, fd)

    def reorder_classes(self, array_in: np.ndarray) -> np.ndarray:
        temp = torch.clone(array_in)
        for i, j in enumerate(
                self.cf_["data"]["test_dataset"]["classes_reorder"]):
            if i != j: temp[array_in == j] = i
        return temp

    def get_im_impl_(self, im_row):
        return Image.fromarray(self.im_getter_.get_im(im_row))

    def get_im(self, im_row):
        return self.im_to_bytestr(
            Image.fromarray(self.im_getter_.get_im(im_row)))

    def get_patient_id(self, x):
        return re.findall(f"(?:INV|NIO)_[a-zA-Z]+_[0-9]+",
                          x.split("@")[0].split("-")[0])[0]

    def get_slide_id(self, x):
        return "-".join(x.split("-")[:2])

    def get_patch_id(self, x):
        return x.split("@")[0]

    def __call__(self, preds):

        preds = self.process_predictions(preds)

        self.visualize_feature_rank(preds["val"])
        if self.nc_ > 1:
            self.compute_metrics(preds["val"])

        if "tsne" in self.cf_.testing:
            self.make_interactive_tsne(preds)

        if "nn_viz" in self.cf_.testing:
            self.make_nn(preds)

        #self.make_sample_cluster()


def do_eval(cf, out_root, predictions, do_softmax=False):

    for k in ["train", "val", "xmplr"]:
        if k not in predictions:
            predictions[k] = None

    if cf["data"]["set"] in ["mnist", "cifar10"]:
        MNISTEvalModule(cf=cf, out_root=out_root,
                        do_softmax=do_softmax)(predictions)

    elif cf["data"]["set"] in ["he", "srh"]:
        SRHEvalModule(cf=cf, out_root=out_root,
                      do_softmax=do_softmax)(predictions)

    elif cf["data"]["set"] in {"scsrh_bench", "scsrh"}:
        SRHCellEvalModule(cf=cf, out_root=out_root,
                          do_softmax=do_softmax)(predictions)
    else:
        raise ValueError(
            "data/set must be in [mnist, cifar10, he, srh, scsrh]")


def main():
    config, out_dir, preds = setup_eval_module_standalone_infra(
        get_xmplr_pred=True)
    do_eval(config, out_dir, preds)


if __name__ == "__main__":
    main()
