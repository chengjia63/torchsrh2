import os
import json
import math
import random
import logging
import itertools
from typing import List, Any, Optional, Callable, Union, Tuple, Dict
from abc import ABC, abstractmethod
from enum import Enum, unique, auto
from os.path import join as opj
import pandas as pd
from tqdm import tqdm
from torchsrh.datasets.common import patch_code_to_list
from ts2.data.meta_parser import GTParser, DiscriminationLevel


class CellCSVParser():

    def __init__(self, data_root: str, slide_cell_thres: int,
                 which_cell_filter: str, df: Union[pd.DataFrame, str],
                 primary_label_idx: int):

        if type(df) is pd.DataFrame:
            self.df_name_ = "DataFrame"
            self.df_ = df
        else:
            self.df_name_ = df
            self.df_ = pd.read_csv(df, dtype=str)

        self.gt_parser_params_ = {
            "primary_label_idx": primary_label_idx,
            "use_patch_code_as_label": False
        }
        self.get_gt_ = GTParser(**self.gt_parser_params_)

        slide_list = self.df_.apply(
            lambda x: ".".join([x["patient"], str(x["mosaic"])]),
            axis=1).tolist()

        self.hyper_ = {
            "data_root": data_root,
            "slide_cell_thres": slide_cell_thres,
            "which_cell_filter": which_cell_filter
        }
        self.cmp_ = CellMetaParser(
            get_patch_gt=self.get_gt_,
            **self.hyper_,
        )
        assert len(slide_list) == len(set(slide_list))  # slides are unique

    def __call__(self, cache_dir: str, level: str):

        if level == DiscriminationLevel.CELL:
            inst, mmap_info = self.get_cell_instances()
        elif level == DiscriminationLevel.SLIDE:
            inst, mmap_info = self.get_slide_instances()
        elif level in DiscriminationLevel._value2member_map_:
            raise NotImplementedError()
        else:
            raise ValueError(
                "level must be in {patch, slide, patient, hierarchical}, got %s"
                % level)

        params = self.hyper_
        params.update({"gt_parser": self.gt_parser_params_})
        params["df"] = self.df_name_
        params["instance_len"] = len(inst)
        params["labels"] = sorted({i["label"] for i in inst})

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        with open(opj(cache_dir, "meta.json"), "w", encoding="utf-8") as fd:
            json.dump(params, fd, indent=2)
        with open(opj(cache_dir, "instances.json"), "w",
                  encoding="utf-8") as fd:
            json.dump(inst, fd)
        with open(opj(cache_dir, "mmap_info.json"), "w",
                  encoding="utf-8") as fd:
            json.dump(mmap_info, fd)

    def get_cell_instances(self):
        instances = []
        tensor_shapes = {}

        for _, slide_s in tqdm(self.df_.iterrows(), total=len(self.df_)):
            cells_s, ts_s, id_s = self.cmp_.process_slide(slide_s,
                                                          keep_label=True)

            if cells_s:
                instances.extend(cells_s)
                tensor_shapes.update({id_s: ts_s})

        return instances, tensor_shapes

    def get_slide_instances(
            self) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
        slide_instances = []
        tensor_shapes = {}

        for _, slide_s in tqdm(self.df_.iterrows(), total=len(self.df_)):
            cells_s, ts_s, id_s = self.cmp_.process_slide(slide_s,
                                                          keep_label=False)

            if cells_s:
                slide_instances.append({
                    "name": id_s,
                    "label": self.get_gt_(slide_s, None),
                    "cells": cells_s
                })
                tensor_shapes.update({id_s: ts_s})
        return slide_instances, tensor_shapes


class CellFilters():

    @staticmethod
    def include_all(_: Dict) -> bool:
        return True

    @staticmethod
    def exclude_edge(cell_meta: Dict) -> bool:
        return not cell_meta["is_edge"]

    @staticmethod
    def exclude_lowconfidence(cell_meta: Dict) -> bool:
        return cell_meta["score"] > 0.8

    @staticmethod
    def include_nuclei(cell_meta: Dict) -> bool:
        return cell_meta["celltype"] == "nuclei"

    @staticmethod
    def include_nuclei_exclude_edge_exclude_lowconfidence(cell_meta):
        return (CellFilters.exclude_edge(cell_meta)
                and CellFilters.exclude_lowconfidence(cell_meta)
                and CellFilters.include_nuclei(cell_meta))


class CellMetaParser():
    """Parser to work with MLiNS internal SRH dataset metadata files

    It works with json metadata files for each patient and requires dataframes
    to describe the slides **in a single patient** that we want. The dataframe
    should have the format:
    ```
    institution, patient, mosaic, patch_code, label1, label2, ...
    ```
    It produces list of instances, which are dictionaries in the format
    specified by make_instance_dict function.

    Attributes:
        data_root_: str containing the root path of the dataset
        seg_model_: str specifying the hash of the segmentation model
        p_meta_: Dict read in from the json metadata file
    """

    def __init__(
            self,
            #inst_name: str,
            #patient_id: str,
            data_root: str,
            which_cell_filter: str = "include_all_cells",
            slide_cell_thres: Optional[int] = None,
            get_patch_gt: Optional[Callable] = None):
        """Inits the SRH Metadata parser"""

        self.data_root_ = data_root
        self.slide_cell_thres_ = slide_cell_thres
        self.label_parser_ = get_patch_gt
        #self.patch_path_func_ = self.make_ts2_sc_memmap_path
        self.include_cell = CellFilters.__dict__[which_cell_filter]

    def process_slide(self, slide_s: pd.Series, keep_label: bool):  # one slide

        # Cheng: I made a mistake here - I accidently saved the cell memmap /
        # meta files formatted like slide_id_file, which is wrong and confusing
        # pt and slide ids should be speareted with "-" when saving, since "_"
        # is in pt id. can potentially fix this in the future for better
        # consistency. But need to use both versions here.
        slide_id_file = "_".join([slide_s["patient"], slide_s["mosaic"]])
        slide_id_train = "-".join([slide_s["patient"], slide_s["mosaic"]])

        meta_fname = opj(self.data_root_, f"{slide_id_file}_meta.json")
        mmap_fname = opj(self.data_root_, f"{slide_id_file}_cells.dat")

        if not (os.path.exists(meta_fname) and os.path.exists(mmap_fname)):
            logging.warning("Slide %s DNE", slide_id_train)
            exit(1)
            #return [], {}, ""

        with open(meta_fname) as fd:
            meta_s = json.load(fd)

        # construct instance list
        if keep_label and self.label_parser_:
            label_c = {"label": self.label_parser_.get_gt(slide_s, None)}
        elif keep_label:
            label_c = {"label": None}
        else:
            label_c = {}

        slide_instances = [{
            "slide_id":
            slide_id_train,
            "patch_name":
            curr_c["patch"].removeprefix(slide_id_train).removeprefix("-"),
            "cell_idx":
            i,
            **label_c,
        } for i, curr_c in enumerate(meta_s["cells"])
                           if self.include_cell(curr_c)]

        # logging status
        if self.label_parser_:
            label_logging = str(self.label_parser_.make_gt_list(slide_s))
        else:
            label_logging = ""

        if len(slide_instances) > 0:
            logging.info(
                "Slide %s OK / labels %s / %d cells ramain / %d cells total",
                slide_id_train, label_logging, len(slide_instances),
                len(meta_s["cells"]))
        else:
            logging.warning(
                "Slide %s EMPTY / labels %s / 0 cells ramain / %d cells total",
                slide_id_train, label_logging, len(meta_s["cells"]))
            return [], {}, ""

        # ceil number of cells per slide
        if self.slide_cell_thres_:
            slide_instances = self.ceil_instance_thres(slide_instances)

        # construct mmap info
        slide_mmap_info = {
            "path": f"{slide_id_file}_cells.dat",
            "shape": meta_s["tensor_shape"]
        }

        return slide_instances, slide_mmap_info, slide_id_train

    def ceil_instance_thres(self, instances: List[Any]):
        """random sample thres number of instances from instances list

        If thres >= len(instances), we randomly sample from instances.
        If thres < len(instances), we shuffle, repeat instances, and then choose
        the first thres items.

        Args:
            instances: a list of instances
            thres: the threshold for number of instances(slides/patches) to be
                sampled / oversampled
            """

        num_repeat = math.ceil(self.slide_cell_thres_ / len(instances))
        random.shuffle(instances)
        instances_repeated = list(
            itertools.chain(*itertools.repeat(instances, num_repeat)))

        return sorted(instances_repeated[:self.slide_cell_thres_],
                      key=lambda x: x["cell_idx"])
