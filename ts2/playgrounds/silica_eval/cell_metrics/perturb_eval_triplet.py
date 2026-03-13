import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from glob import glob
from os.path import join as opj
import pandas as pd
import gzip
from tqdm import tqdm

import torchmetrics
from torchmetrics.functional.classification import accuracy
import logging

import altair as alt


def load_data(fn, perturb_width):
    with gzip.open(fn) as fd:
        data = torch.load(fd)

    return pd.DataFrame(
        {
            "path": data["path"],
            "embeddings": data["embeddings"].tolist(),
            "perturb_width": perturb_width,
            # "logits": data["logits"].tolist(),
            "label": data["label"],
        }
    )


def load_data_raw(fn, perturb_width):
    with gzip.open(fn) as fd:
        data = torch.load(fd)

    return data


def make_pred_df(data, perturb_width):
    return pd.DataFrame(
        {
            "path": data["path"],
            "embeddings": data["embeddings"].tolist(),
            "perturb_width": perturb_width,
            # "logits": data["logits"].tolist(),
            "label": data["label"],
        }
    )


def nice_chart(chart, chart_size=400):
    return (
        chart.properties(width=chart_size, height=chart_size)
        .configure_axis(tickSize=0, labelFontSize=16, titleFontSize=16)
        .configure_legend(labelFontSize=16, titleFontSize=16)
        .configure_title(
            fontSize=16,
        )
    )


def odd_one_out_index_batch(x: np.ndarray) -> np.ndarray:
    """
    Return odd-one-out indices for a batch of triplets.

    Args:
        x: Array of shape (n, 3, d), where each triplet has 3 embeddings.

    Returns:
        Array of shape (n,) with values in {0, 1, 2}.
    """
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1] != 3:
        raise ValueError(f"Expected shape (n, 3, d), got {x.shape}")

    # Pairwise dot-product similarities: (n, 3, 3)
    S = x @ np.swapaxes(x, -1, -2)

    # Ignore self-similarity
    idx = np.arange(3)
    S[:, idx, idx] = -np.inf

    # Flatten each 3x3 matrix, find max-similarity pair
    flat_argmax = np.argmax(S.reshape(-1, 9), axis=1)
    i, j = np.divmod(flat_argmax, 3)

    # Since indices are {0,1,2}, the missing one is 3 - i - j
    return 3 - i - j


def get_results_one_eval(exp_name, ckpt_iter, eval_regex, band_width_fn):
    preds = pd.DataFrame(
        glob(
            f"/nfs/turbo/umms-tocho-snr/exp/chengjia/ts2/fmi_dinov2_cc_new/{exp_name}/models/eval/{ckpt_iter}/{eval_regex}/predictions/val_predictions.pt.gz"
        ),
        columns=["pred_fn"],
    )
    print(preds)
    preds["band_width"] = preds["pred_fn"].apply(band_width_fn)
    preds = preds.sort_values("band_width").reset_index(drop=True)

    # no_perturb_train_preds = preds.loc[preds["band_width"] == 0, "pred_fn"]
    # assert len(no_perturb_train_preds) == 1

    # with gzip.open(no_perturb_train_preds.iloc[0]) as fd:
    #    train_data_unperturbed = torch.load(fd)

    pred_data = [load_data(i["pred_fn"], i["band_width"]) for _, i in preds.iterrows()]
    oddoneout_ratio = torch.tensor(
        [
            (
                odd_one_out_index_batch(
                    torch.tensor(predd["embeddings"].tolist()).numpy()
                )
                == 2
            ).sum()
            / len(predd)
            for predd in pred_data
        ]
    )
    metrics = {
        "oddoneout_ratio": oddoneout_ratio,
        "band_width": preds["band_width"],
    }
    print(metrics)
    return metrics


def main():
    ckpt_iter = "training_124999"

    eval_configurations = {
        "89d3ad98_May23-13-58-49_sd1000_dev_tune0": [
            # "*BENCHDB_CBSP*_epoch0-step124999_tune*",
            "*BENCHDB_TRIPLET_PATCH*"
            #"*Mar09*_sd1000_smpt_BENCHDB_CCY_CBSP*_epoch0-step124999_sd1k*"
        ],
        # "6778e5d1_May27-15-59-58_sd1000_dev_tune0": [
        #    "*BENCHDB_CBSP*_epoch0-step124999_tune*",
        # ],
        # "a1ceb84e_Jan10-05-16-51_sd1000_newrun_dev_maskobw_tune1":[
        #    "*BENCHDB_CBSP*_epoch0-step124999_sd1k*",
        # ],
        "546d5129_Jan12-20-08-19_sd1000_newrun_dev_nomaskobw_lr54_tune0": [
            "*BENCHDB_TRIPLET_PATCH*"
            #"*Mar10*_sd1000_smpt_BENCHDB_CCY_CBSP*_epoch0-step124999_sd1k*"
            # "*BENCHDB_CBSP*_epoch0-step124999_sd1k*",
        ],
        "c2b59e45_Jan12-20-08-19_sd1000_newrun_dev_maskobw_lr54_tune1": [
            "*BENCHDB_TRIPLET_PATCH*"
            # "*BENCHDB_CBSP*_epoch0-step124999_sd1k*",
            #"*Mar10*_sd1000_smpt_BENCHDB_CCY_CBSP*_epoch0-step124999_sd1k*"
        ],
    }
    exp_name_map = {
        "89d3ad98_May23-13-58-49_sd1000_dev_tune0": "DINOv2",
        "6778e5d1_May27-15-59-58_sd1000_dev_tune0": "Ours",
        "a1ceb84e_Jan10-05-16-51_sd1000_newrun_dev_maskobw_tune1": "Inside iBOT mask",
        "546d5129_Jan12-20-08-19_sd1000_newrun_dev_nomaskobw_lr54_tune0": "Ours_lr54",
        "c2b59e45_Jan12-20-08-19_sd1000_newrun_dev_maskobw_lr54_tune1": "Inside_lr54",
    }

    out_fname = "benchdb_triplet_patch"
    # model_name = exp_name.split("_")[0]

    band_width_fn = lambda x: int(x.split("/")[-3].split("_")[-4].removeprefix("CBSP"))
    exps = list(eval_configurations.keys())
    means = []
    stds = []
    band_widths = None
    for k in exps:
        all_results = []
        for ev in tqdm(eval_configurations[k]):

            all_results.append(get_results_one_eval(k, ckpt_iter, ev, band_width_fn))
            if band_widths == None:
                band_widths = torch.tensor(all_results[-1]["band_width"])
            else:
                assert (
                    band_widths == torch.tensor(all_results[-1]["band_width"])
                ).all()

        means.append(
            torch.mean(torch.stack([x["oddoneout_ratio"] for x in all_results]), axis=0)
        )
        stds.append(
            torch.std(torch.stack([x["oddoneout_ratio"] for x in all_results]), axis=0)
        )

    perct_masked = band_widths * 10

    E = len(exps)
    L = len(perct_masked)

    # (E, L)
    mean_mat = torch.stack(means).cpu().numpy()
    std_mat = torch.stack(stds).cpu().numpy()

    # (E, L)
    lower = mean_mat - std_mat
    upper = mean_mat + std_mat

    # tile and repeat for broadcasted assembly (no loops)
    df = pd.DataFrame(
        {
            "method": np.repeat(exps, L),
            "perct_masked": np.tile(perct_masked, E),
            "mca": mean_mat.reshape(-1),
            "std": std_mat.reshape(-1),
            "lower": lower.reshape(-1),
            "upper": upper.reshape(-1),
        }
    )

    df["method"] = df["method"].map(exp_name_map)

    print(df)

    band = (
        alt.Chart(df)
        .mark_area(opacity=0.15)
        .encode(
            x=alt.X("perct_masked:Q", title="Perct masked"),
            y=alt.Y("lower:Q", title="Odd one out accuracy"),
            y2="upper:Q",
            color="method:N",
        )
    )

    line = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x="perct_masked:Q",
            y="mca:Q",
            color="method:N",
        )
    )

    nice_chart(band + line).interactive().save(f"{out_fname}.svg")
    nice_chart(band + line).interactive().save(f"{out_fname}.png")
    nice_chart(band + line).interactive().save(f"{out_fname}.pdf")


if __name__ == "__main__":
    main()
