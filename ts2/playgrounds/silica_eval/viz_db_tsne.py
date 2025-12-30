import torch
import pandas as pd
import sklearn
import altair as alt
import numpy as np
import yaml
from omegaconf import OmegaConf
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
    # ours:
    #with gzip.open("/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/941f49cb_May29-01-43-17_sd1000_smpt_BENCHDB_SE_epoch0-step124999_tune0/predictions/train_predictions.pt.gz") as fd:

    # dv2
    #with gzip.open("/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/89d3ad98_May23-13-58-49_sd1000_dev_tune0/models/eval/training_124999/0e97d769_May26-11-14-43_sd1000_smpt_BENCHDB_SE_epoch0-step124999_tune3/predictions/train_predictions.pt.gz") as fd:
    #    db_data = torch.load(fd)

    #db_data = pd.DataFrame({
    #    "path":db_data["path"],
    #    "label":db_data["label"],
    #    "embeddings": [i for i in db_data["embeddings"]]
    #})
    #db_embs = torch.stack(db_data["embeddings"].tolist())


    #db_data = torch.load("/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/2510a2cb_May30-19-10-14_sd1000_INFDB_dev/predictions/pred.pt")
    #db_data = torch.load("/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/89d3ad98_May23-13-58-49_sd1000_dev_tune0/models/eval/training_124999/17b1bb09_May30-19-44-14_sd1000_INFDB_dev/predictions/pred.pt")

    return pd.DataFrame(torch.load(inf_path))

def compute_process_tsne(all_embs):
    # compute tsne
    tsne = sklearn.manifold.TSNE(n_components=2, perplexity=50, random_state=1000)
    embeddings_2d = tsne.fit_transform(all_embs)

    # process tsne
    min_vals = embeddings_2d.min(axis=0)
    max_vals = embeddings_2d.max(axis=0)
    embeddings_2d = (embeddings_2d - min_vals) / (max_vals - min_vals) * 0.9 + 0.05
    return embeddings_2d

def load_images(cell_samples,
                cell_instances_path,
                default_config_path):
    with open(default_config_path) as fd:
        cf = OmegaConf.create(yaml.safe_load(fd))
    cf.data.test_dataset.params.cell_instances = cell_instances_path

    dataset = SingleCellListInferenceDataset(
        transform=HistologyTransform(**cf.data.xform_params),
        **cf.data.test_dataset.params)
    images = [dataset[i]["image"] for i in cell_samples]
    im_str = [im_to_bytestr(einops.rearrange(
        (adjust_contrast(adjust_brightness(i.squeeze(),2),2) * 255).to(torch.uint8), "c h w -> h w c").numpy() )for i in images]
    return im_str

def main(inf_path, out_name, cell_instances_path, default_config_path):

    # get data
    db_data = load_data(inf_path)

    # sample embeddings
    cell_samples = sorted(db_data.groupby("label").apply(sample_idx).explode().values)
    db_sample = db_data.iloc[cell_samples]
    db_sample_embs = torch.stack(db_sample["embeddings"].tolist()).numpy()
    db_sample_embs = torch.nn.functional.normalize(torch.tensor(db_sample_embs))

    embeddings_2d = compute_process_tsne(db_sample_embs)

    # load image
    im_str = load_images(cell_samples, cell_instances_path, default_config_path)
    combined_data = pd.DataFrame({
        "x": embeddings_2d[:,0],
        "y": embeddings_2d[:,1],
        "path":db_data.iloc[cell_samples]["path"],
        "image": im_str,
        "label": db_data.iloc[cell_samples]["label"],
    })
    combined_data["nio_num"] = combined_data["path"].str.extract("(NIO_UM_[0-9]+)")

    # get tumor type
    ttype_ = pd.read_csv("/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/playgrounds/data/srh_ttype_cheng.csv")
    combined_data = combined_data.merge(
        ttype_,
        left_on="nio_num", right_on="type_institution_number", how="left"
    ).drop("type_institution_number", axis=1
    ).rename({"ttype_cheng": "weak_label"}, axis=1)
    combined_data.loc[combined_data["label"]=="normal", "weak_label"] = "normal"
    combined_data["weak_label"] = combined_data["weak_label"].fillna("other")

    # plot tsne
    weak_labels = combined_data["weak_label"].drop_duplicates().tolist()
    if len(weak_labels) == 2:
        colors = TC()(c="LR")
    elif len(weak_labels) == 9:
        colors = TC()(c="LYTROMQVF",s="656666666")
    else:
        colors = TC()(nc=len(weak_labels))
    alt.data_transformers.disable_max_rows()
    
    tsne_unit_axis = alt.Axis(tickSize=0,
                            values=np.linspace(0,1,6),
                            domain=False,
                            labels=False,
                            title="")

    base_chart = alt.Chart(combined_data).mark_point(
        filled=True
    ).encode(
        x=alt.X("x",
                axis=tsne_unit_axis,
                scale=alt.Scale(domain=[0,1])),
        y=alt.Y("y",
                axis=tsne_unit_axis,
                scale=alt.Scale(domain=[0,1])),
        tooltip=["image", "path"],
        color=alt.Color("weak_label:N",
                        scale=alt.Scale(domain=weak_labels,
                                        range=colors)),
        opacity=alt.value(.8)
    ) # yapf: disable

    base_chart.properties(height=600,width=600).save(f"{out_name}.pdf")
    base_chart.properties(height=600,width=600).save(f"{out_name}.png")
    base_chart.properties(height=600,width=600).interactive().save(f"{out_name}.html")

if __name__=="__main__":
    g2m_instances_path= "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/train/srhum_glioma_2m_.csv"
    um2m_instances_path= "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/train/srhum_2m.csv"
    default_config_path="/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/train/config/chengjia/inference_dinov2_scsrhdb.yaml"

    inf_path = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/e2e43218_Jun03-18-51-29_sd1000_INFDB_dev/predictions/pred.pt"
    main(inf_path, "out/g2m_ours", g2m_instances_path, default_config_path)

    inf_path = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/6778e5d1_May27-15-59-58_sd1000_dev_tune0/models/eval/training_124999/2510a2cb_May30-19-10-14_sd1000_INFDB_dev/predictions/pred.pt"
    main(inf_path, "out/um2m_ours", um2m_instances_path, default_config_path)
    
    inf_path = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/89d3ad98_May23-13-58-49_sd1000_dev_tune0/models/eval/training_124999/17b1bb09_May30-19-44-14_sd1000_INFDB_dev/predictions/pred.pt"
    main(inf_path, "out/um2m_dinov2", um2m_instances_path, default_config_path)

    inf_path = "/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/89d3ad98_May23-13-58-49_sd1000_dev_tune0/models/eval/training_124999/6b91de8d_Jun03-19-13-20_sd1000_INFDB_dev/predictions/pred.pt"
    main(inf_path, "out/g2m_dinov2", um2m_instances_path, default_config_path)

