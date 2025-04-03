import os
import json
import logging
from shutil import copy2
from functools import partial
from typing import Tuple, Dict, TextIO
import argparse
import gzip
from os.path import join as opjoin

import torch
import pytorch_lightning as pl

from ts2.train.infra import read_process_cf


def setup_eval_module_standalone_infra(get_train_pred=True,
                                       get_xmplr_pred=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')
    parser.add_argument('-e',
                        '--exp_name',
                        type=str,
                        required=False,
                        help='experiment name')
    parser.add_argument('-t',
                        '--train',
                        type=str,
                        required=False,
                        help='train instance')
    parser.add_argument('-v',
                        '--eval',
                        type=str,
                        required=False,
                        help='eval instance')

    args = parser.parse_args()

    config = read_process_cf(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format=
        "[%(levelname)-s|%(asctime)s|%(filename)s:%(lineno)d|%(funcName)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()])
    logging.info("Standalone eval module")

    pl.seed_everything(config["infra"]["seed"])

    if args.exp_name:
        config["infra"]["exp_name"] = args.exp_name

    if args.train:
        config["testing"]["ckpt_path"] = args.train

    if args.eval:
        config["testing"]["eval_predictions"] = args.eval

    logging.info("Loading train predictions")
    if get_train_pred:
        with gzip.open(
                opjoin(config["infra"]["log_dir"], config["infra"]["exp_name"],
                       config["testing"]["ckpt_path"].split("/")[0], "evals",
                       config["testing"]["eval_predictions"], "predictions",
                       "train_predictions.pt.gz")) as fd:
            train_pred = torch.load(fd)
        logging.info("Loading train predictions - done")
    else:
        train_pred = None
        logging.info("Loading train predictions - skipped")

    logging.info("Loading val predictions")
    with gzip.open(
            opjoin(config["infra"]["log_dir"], config["infra"]["exp_name"],
                   config["testing"]["ckpt_path"].split("/")[0], "evals",
                   config["testing"]["eval_predictions"], "predictions",
                   "val_predictions.pt.gz")) as fd:
        val_pred = torch.load(fd)
    logging.info("Loading val predictions - done")

    logging.info("Loading xmplr predictions")
    if get_xmplr_pred:
        with gzip.open(
                opjoin(config["infra"]["log_dir"], config["infra"]["exp_name"],
                       config["testing"]["ckpt_path"].split("/")[0], "evals",
                       config["testing"]["eval_predictions"], "predictions",
                       "xmplr_predictions.pt.gz")) as fd:
            xmplr_pred = torch.load(fd)
        logging.info("Loading xmplr predictions - done")
    else:
        xmplr_pred = None
        logging.info("Loading xmplr predictions - skipped")

    preds = {"train": train_pred, "val": val_pred, "xmplr": xmplr_pred}

    out_dir = config["testing"]["ckpt_path"]  # args.train.split("-")[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logging.info("Standalone eval infra - done")
    return config, out_dir, preds