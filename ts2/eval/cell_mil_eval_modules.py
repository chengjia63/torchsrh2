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
import torchmetrics
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
from ts2.eval.infra import setup_eval_module_standalone_infra

from ts2.data.transforms import HistologyTransform
from ts2.data.db_improc import instantiate_process_read
from ts2.data.meta_parser import CachedCSVParser


def plot_confusion(confusion: np.ndarray, out_file: str,
                   class_names: List[str]):
    assert confusion.shape[0] == confusion.shape[1]
    nc = confusion.shape[0]

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


def compute_sample_tsne(data, k="embs", cf={}):
    tsne = TSNE(n_components=2, **cf)
    xy = tsne.fit_transform(np.array([d for d in data[k]]))

    min_vals = xy.min(axis=0)
    max_vals = xy.max(axis=0)
    xy = (xy - min_vals) / (max_vals - min_vals)
    xy = xy * 0.9 + 0.05

    data["tsne_x"] = xy[:, 0].tolist()
    data["tsne_y"] = xy[:, 1].tolist()
    return data


from ts2.data.transforms import SRHBaseTransform
from ts2.data.db_improc import process_read_srh
from torchvision.transforms.functional import adjust_contrast, adjust_brightness
from PIL import Image


class SRHImageGetter():

    def __init__(self, dset_root: str, return_str: bool = True):
        self.dset_root = dset_root
        self.return_str = return_str
        self.sbt = SRHBaseTransform()

    @staticmethod
    def get_nio_num_components(image_str: str):
        pattern = r'(?P<accession>(?P<nioinv>NIO|INV)_(?P<instution>[A-Z]+)_(?P<id>\d+[a-z]*))-(?P<mosaic>[a-zA-Z0-9_]+)-'
        match = re.match(pattern, image_str)
        assert match

        return match.groupdict()

    @staticmethod
    def im_to_bytestr(im):
        image = Image.fromarray(
            (255 * im.permute(1, 2, 0).numpy()).astype(np.uint8))
        output = io.BytesIO()
        image.save(output, format='JPEG')
        return "data:image/jpeg;base64," + base64.b64encode(
            output.getvalue()).decode()

    def get_image(self, row):
        nionc = self.get_nio_num_components(row["path"])
        try:
            im_path = opj(self.dset_root, nionc["instution"], nionc["accession"],
                          nionc["mosaic"], "patches", f'{row["path"]}.tif')
            im = process_read_srh(im_path)
            im = adjust_brightness(adjust_contrast(self.sbt(im), 2), 3)
        except:
            print(row["path"])
            print(nionc)
            im = torch.zeros(3,300,300)

        if self.return_str:
            return self.im_to_bytestr(im)
        else:
            return im

    def __call__(self, row):
        return self.get_image(row)


class CellMILEvalModule():

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

    def compute_metrics(self, pred):
        if self.do_softmax_:
            logging.info("applying softmax for metrics")
            pred["logits"] = torch.nn.functional.softmax(pred["logits"], dim=1)

        cm = confusion_matrix(y_true=pred["label"],
                              y_pred=pred["logits"].argmax(axis=1))

        metrics = {
            "acc":
            torchmetrics.functional.accuracy(pred["logits"],
                                             pred["label"],
                                             num_classes=2,
                                             average="micro").item(),
            "mca":
            torchmetrics.functional.accuracy(pred["logits"],
                                             pred["label"],
                                             num_classes=2,
                                             average="macro").item(),
            "auroc":
            torchmetrics.functional.auroc(pred["logits"][:, 1],
                                          pred["label"],
                                          task="binary",
                                          average='macro').item(),
            "f1":
            torchmetrics.functional.f1_score(pred["logits"][:, 1],
                                             pred["label"],
                                             task="binary",
                                             average='macro').item(),
            "precision":
            torchmetrics.functional.precision(pred["logits"][:, 1],
                                              pred["label"],
                                              task="binary",
                                              average='macro').item(),
            "recall":
            torchmetrics.functional.recall(pred["logits"][:, 1],
                                           pred["label"],
                                           task="binary",
                                           average='macro').item()
        }

        all_metrics_df = pd.DataFrame.from_dict(metrics,
                                                orient="index",
                                                columns=["patch"]).T
        all_metrics = {"metrics": metrics, "cm": cm.tolist()}

        plot_confusion(cm, "test.svg", self.class_names_)
        # exclude bad patients from evaluation
        #patient_exclude = [
        #    "NIO_NYU_66", "NIO_UM_936", "NIO_UM_936b", "NIO_UM_868",
        #    "NIO_UM868b"
        #]
        #pred = pred[~pred["patient"].isin(patient_exclude)].reset_index()

        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        logging.info(f"all metrics\n{str(all_metrics_df)}")

        # save metrics
        all_metrics_df.to_csv(opj(self.out_root_, "metrics.csv"), index=False)
        with open(opj(self.out_root_, "all_metrics.json"), "w") as fd:
            json.dump(all_metrics, fd)

    def make_interactive_tsne(self, predictions, do_im_tooltip=True):
        pred_smpl = pd.DataFrame({
            "embs": [i.squeeze() for i in predictions["embs"]],
            "path":
            predictions["path"],
            "label":
            predictions["label"]
        }).sample(512).reset_index(drop=True)
        pred_smpl = compute_sample_tsne(pred_smpl)
        pred_smpl = pred_smpl.drop("embs", axis=1)

        if do_im_tooltip:
            get_image = SRHImageGetter(
                dset_root="/nfs/turbo/umms-tocho/root_srh_db/")
            pred_smpl["image"] = pred_smpl.apply(get_image, axis=1)
            tooltip = ['image', "path"]
        else:
            tooltip = ["path"]

        pred_smpl["label"] = self.class_names_[pred_smpl["label"]]
        color_key = "label"
        alt.data_transformers.disable_max_rows()
        rgb = OpenColor.setup_colors(len(set(pred_smpl[color_key].tolist())),
                                     ind=7)
        chart_encode_params = {"strokeWidth": alt.value(0)}

        bind_range = alt.binding_range(min=4, max=256, name='Size ')
        param_size = alt.param(bind=bind_range)

        class_selection = alt.selection_point(fields=[color_key],
                                              bind='legend')
        selected_color = alt.Color(f"{color_key}:N",
                                   scale=alt.Scale(domain=self.class_names_,
                                                   range=rgb.tolist()))
        color = alt.condition(class_selection, selected_color,
                              alt.value('lightgray'))

        op = alt.condition(class_selection, alt.value(1.0), alt.value(0.1))

        chart = alt.Chart(pred_smpl).mark_point(
                filled=True,
                size=alt.expr(param_size.name),
            ).encode(
                x=alt.X("tsne_x",
                        axis=alt.Axis(tickSize=0),
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("tsne_y",
                        axis=alt.Axis(tickSize=0),
                        scale=alt.Scale(domain=[0, 1])),
                color=color,
                opacity=op,
                tooltip=tooltip,
                **chart_encode_params
            ).add_params(
                param_size,
                class_selection
            ) # yapf: disable

        chart.properties(width=800, height=800).interactive().save(
            opj(self.out_root_, f"itsne.html"))

        chart_static = alt.Chart(pred_smpl).mark_point(
                filled=True,
                size=32
            ).encode(
                x=alt.X("tsne_x",
                        axis=alt.Axis(tickSize=0, tickCount=5, title=None),
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("tsne_y",
                        axis=alt.Axis(tickSize=0, tickCount=5, title=None),
                        scale=alt.Scale(domain=[0, 1])),
                color=selected_color,
                **chart_encode_params
            ).configure_axis(
                domain=False,
                ticks=False,
                labels=False
            )# yapf: disable

        chart_static.properties(width=512, height=512).interactive().save(
            opj(self.out_root_, f"itsne.pdf"))
        chart_static.properties(width=512, height=512).interactive().save(
            opj(self.out_root_, f"itsne.png"))

    def reorder_classes(self, array_in: np.ndarray) -> np.ndarray:
        temp = torch.clone(array_in)
        for i, j in enumerate(
                self.cf_["data"]["test_dataset"]["classes_reorder"]):
            if i != j: temp[array_in == j] = i
        return temp

    def visualize_attention_one_patch(self, row):
        cell_meta_root = "/nfs/turbo/umms-tocho-snr/exp/chengjia/scsrh_repl_root2"
        cell_meta_fname = "_".join(row["path"].split("-")[:2] + ["meta.json"])

        with open(opj(cell_meta_root, cell_meta_fname)) as fd:
            cells = pd.DataFrame(json.load(fd)["cells"])

        cells = cells[cells["patch"] == row["path"]]

        inc_mask = (~cells["is_edge"]) & (cells["score"] > 0.8) & (cells["celltype"] == "nuclei")
        exc_mask = ~inc_mask

        included_cells = cells[inc_mask]
        excluded_cells = cells[exc_mask]

        included_cells["attn_score"] = row["attn"][0]
        excluded_cells["attn_score"] = 0
        cells = pd.concat([excluded_cells, included_cells])

        colors = generate_colors_from_confidence(cells["attn_score"]*len(included_cells))
        labels = (cells["attn_score"]).map("{:.2f}".format)

        labels.iloc[:len(excluded_cells)] = "X"
        new_im = draw_yolo_style_boxes_pil(
            to_pil_image(row["image"]),
            boxes=torch.tensor(cells["bbox"].tolist()),
            labels=labels,
            colors=colors)
        new_im_nonum = draw_yolo_style_boxes_pil(
            to_pil_image(row["image"]),
            boxes=torch.tensor(cells["bbox"].tolist()),
            labels=labels,
            colors=colors, do_text=False)

        all_im = np.hstack([np.array(to_pil_image(row["image"])), np.array(new_im_nonum), np.array(new_im)])
        return all_im

    def visualize_attention(self, predictions):
        pred_smpl = pd.DataFrame({
            "prob":
            predictions["logits"].softmax(dim=1)[:, 1],
            "attn":
            predictions["attn"],
            "path":
            predictions["path"],
            "label":
            predictions["label"]
        })
        test = extract_prediction_extremes(pred_smpl)
        get_image = SRHImageGetter(
            dset_root="/nfs/turbo/umms-tocho/root_srh_db/", return_str=False)
        test["image"] = test.apply(get_image, axis=1)
        test["bbox_image"] = test.apply(self.visualize_attention_one_patch, axis=1)

        for i in test["category"].drop_duplicates().tolist():
            os.makedirs(opj(self.out_root_, "attn_viz", i))

        test[["prob", "path", "label", "pred", "category"]].to_csv(opj(self.out_root_, "attn_viz","images.csv"))
        for _, row in test.iterrows():
            Image.fromarray(row["bbox_image"]).save(opj(self.out_root_, "attn_viz", row["category"], f"{row['path']}.png"))


from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw, ImageFont, ImageColor
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def generate_colors_from_confidence(confidences, cmap_name="viridis"):
    """
    Map confidence scores (N, 1) ∈ [0, 1] to RGB colors using a colormap.

    Args:
        confidences (Tensor): (N, 1) tensor of softmax scores (already normalized).
        cmap_name (str): Name of matplotlib colormap.

    Returns:
        List[Tuple[int, int, int]]: RGB values in 0–255.
    """
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=1)

    rgba = cmap(confidences)  # (N, 4), with alpha
    rgb255 = [(int(r * 255), int(g * 255), int(b * 255))
              for r, g, b, _ in rgba]
    return rgb255


def get_contrast_color(rgb):
    """Return black or white depending on RGB background luminance."""
    r, g, b = rgb if isinstance(rgb, tuple) else ImageColor.getrgb(rgb)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 128 else "white"


def draw_yolo_style_boxes_pil(pil_image, boxes, labels, colors, font_size=10, do_text=True):
    """
    Draw YOLO-style bounding boxes with labels above boxes on a PIL image.

    Args:
        pil_image (PIL.Image): Input image.
        boxes (Tensor): Tensor of shape (N, 4), format (xmin, ymin, xmax, ymax).
        labels (List[str]): List of label strings.
        colors (List[Union[str, Tuple[int, int, int]]]): List of colors (str or RGB tuples).
        font_size (int): Font size for labels.
    Returns:
        PIL.Image: Annotated image.
    """
    image = pil_image.copy()
    draw = ImageDraw.Draw(image)

    if do_text:
        font = ImageFont.load_default()

    for box, label, color in zip(boxes, labels, colors):
        x1, y1, x2, y2 = map(int, box.tolist())

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=1)

        if do_text:
            # Text size and background box
            bbox = font.getbbox(label)  # (x0, y0, x1, y1)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

            text_bg = [x1, y1 - text_height - 4, x1 + text_width + 4, y1]
            draw.rectangle(text_bg, fill=color)

            # Draw label text
            draw.text((x1 + 2, y1 - text_height - 2),
                      label,
                      fill=get_contrast_color(color),
                      font=font)

    return image


def extract_prediction_extremes(pred_smpl: pd.DataFrame, k=5) -> pd.DataFrame:
    """
    Extracts:
    - Top 10 confidently correct predictions per class (label = pred, high |prob - 0.5|)
    - Top 10 confidently incorrect predictions per class (label ≠ pred, high |prob - 0.5|)
    - Top 20 most uncertain predictions (prob ≈ 0.5)

    Returns a DataFrame of 60 samples with a 'category' column indicating the group.
    """
    df = pred_smpl.copy()
    df['pred'] = (df['prob'] >= 0.5).astype(int)
    df['confidence'] = np.abs(df['prob'] - 0.5)

    correct_mask = df['label'] == df['pred']
    incorrect_mask = ~correct_mask

    results = []

    for cls in [0, 1]:
        # Confidently correct
        top_correct = df[(df['label'] == cls) & correct_mask].nlargest(
            k, 'confidence')
        top_correct = top_correct.assign(category=f'correct{cls}')

        # Confidently wrong
        top_wrong = df[(df['label'] == cls) & incorrect_mask].nlargest(
            k, 'confidence')
        top_wrong = top_wrong.assign(category=f'wrong{cls}')

        results.extend([top_correct, top_wrong])

    # Most uncertain predictions (closest to 0.5)
    most_uncertain = df.nsmallest(k * 2, 'confidence')
    most_uncertain = most_uncertain.assign(category='uncertain')

    final_df = pd.concat(results + [most_uncertain], ignore_index=True)
    #final_df = pd.concat([most_uncertain], ignore_index=True)
    return final_df


def do_cell_mil_eval(cf, out_root, predictions, do_softmax=False):

    em = CellMILEvalModule(cf=cf, out_root=out_root, do_softmax=do_softmax)
    predictions["val"]["label"] = em.reorder_classes(
        predictions["val"]["label"])
    predictions["val"]["logits"] = predictions["val"][
        "logits"][:, em.cf_["data"]["test_dataset"]["classes_reorder"]]

    em.compute_metrics(predictions["val"])
    em.make_interactive_tsne(predictions["val"])
    em.visualize_attention(predictions["val"])


def main():
    config, out_dir, preds = setup_eval_module_standalone_infra(
        get_train_pred=False)
    do_cell_mil_eval(config, out_dir, preds, do_softmax=True)


if __name__ == "__main__":
    main()
