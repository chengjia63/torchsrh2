import os
import json
import math
import random
import logging
import itertools
from typing import List, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
from enum import Enum, unique, auto
from os.path import join as opj

import pandas as pd
from tqdm import tqdm
from torchsrh.datasets.common import patch_code_to_list


@unique
class DiscriminationLevel(str, Enum):
    CELL = "cell"
    PATCH = "patch"
    PATCH_NZ = "patch_nz"
    SLIDE = "slide"
    PATIENT = "patient"
    HIERARCHICAL = "hierarchical"
    NA = "na"
    UNKNOWN = "unknown"


class CachedCSVParser():

    def __call__(self, cache_dir: str):
        with open(opj(cache_dir, "instances.json")) as fd:
            data = json.load(fd)
        return data

    def get_meta(self, cache_dir):
        with open(opj(cache_dir, "meta.json")) as fd:
            data = json.load(fd)
        return data


class GTParser():

    def __init__(self, primary_label_idx, use_patch_code_as_label):
        self.primary_label_func_ = lambda x: x[primary_label_idx]
        if use_patch_code_as_label:
            self.get_gt = self.get_gt_with_patch_code
        else:
            self.get_gt = self.get_gt_without_patch_code

    def __call__(self, *args, **kwargs):
        return self.get_gt(*args, **kwargs)

    @staticmethod
    def make_gt_list(x) -> List[Any]:
        return x.iloc[4:].tolist()

    def get_gt_col_name(self, df: pd.DataFrame) -> str:
        return self.primary_label_func_(self.make_gt_list(pd.Series(
            df.keys())))

    def get_gt_with_patch_code(self, x: pd.Series,
                               patch_code: str) -> str:  # x is a row
        assert patch_code in {
            None, "tumor", "normal", "nondiagnostic", "nonblank"
        }

        if patch_code in {None, "tumor", "nonblank"}:
            return self.primary_label_func_(self.make_gt_list(x))
        else:
            return patch_code

    def get_gt_without_patch_code(self, x: pd.Series,
                                  patch_code: str) -> str:  # x is a row
        assert patch_code in {
            None, "tumor", "normal", "nondiagnostic", "nonblank"
        }

        return self.primary_label_func_(self.make_gt_list(x))


class SRHCSVParser(ABC):

    def __init__(self):
        raise NotImplementedError()

    def __call__(self):
        raise NotImplementedError()

    @abstractmethod
    def get_cell_instances(self):
        raise NotImplementedError()

    @abstractmethod
    def get_patch_instances(self):
        raise NotImplementedError()

    @abstractmethod
    def get_slide_instances(self):
        raise NotImplementedError()

    @abstractmethod
    def get_patient_instances(self):
        raise NotImplementedError()

    @abstractmethod
    def get_patient_instances(self):
        raise NotImplementedError()

    @abstractmethod
    def get_hierarchical_instances(self):
        raise NotImplementedError()


class PatchCSVParser(SRHCSVParser):

    def __init__(self, data_root: str, seg_model: str, slide_patch_thres: int,
                 df: Union[pd.DataFrame, str], use_emb: bool,
                 use_patch_code_as_label: bool, primary_label_idx: int):

        if type(df) is pd.DataFrame:
            self.df_name_ = "DataFrame"
            self.df_ = df
        else:
            self.df_name_ = df
            self.df_ = pd.read_csv(df)

        self.gt_parser_params_ = {
            "primary_label_idx": primary_label_idx,
            "use_patch_code_as_label": use_patch_code_as_label
        }
        self.get_gt_ = GTParser(**self.gt_parser_params_)

        slide_list = self.df_.apply(
            lambda x: ".".join([x["patient"], str(x["mosaic"])]),
            axis=1).tolist()

        self.hyper_ = {
            "data_root": data_root,
            "seg_model": seg_model,
            "slide_patch_thres": slide_patch_thres,
            "use_emb": use_emb
        }

        assert len(slide_list) == len(set(slide_list))  # slides are unique

    def __call__(self, cache_dir: str, level: str):
        if level == DiscriminationLevel.PATCH:
            inst = self.get_patch_instances()
        elif level == DiscriminationLevel.SLIDE:
            inst = self.get_slide_instances()
        elif level == DiscriminationLevel.PATIENT:
            inst = self.get_patient_instances()
        elif level == DiscriminationLevel.HIERARCHICAL:
            inst = self.get_hierarchical_instances()
        else:
            raise ValueError(
                f"level must be in {{patch, slide, patient, hierarchical}}, ",
                f"got {level}")

        params = self.hyper_
        params.update({"gt_parser": self.gt_parser_params_})
        params["df"] = self.df_name_
        params["instance_len"] = len(inst)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        with open(opj(cache_dir, "meta.json"), "w", encoding="utf-8") as fd:
            json.dump(params, fd, indent=2)
        with open(opj(cache_dir, "instances.json"), "w",
                  encoding="utf-8") as fd:
            json.dump(inst, fd)

    def get_cell_instances(self):
        raise ValueError(
            "Patch CSV Parser does not support cell level instances")

    def get_patch_instances(self):
        instances = []

        for (inst_name, patient_id), patient_s in tqdm(
                self.df_.groupby(["institution", "patient"], dropna=False)):
            instances += PatchMetaParser(
                inst_name=inst_name,
                patient_id=patient_id,
                get_patch_gt=self.get_gt_,
                **self.hyper_).process_slides(patient_s)
        return instances

    def get_slide_instances(self):
        instances = []
        df_gb = tqdm(self.df_.groupby(["institution", "patient"],
                                      dropna=False))
        for (inst_name, patient_id), patient_s in df_gb:
            mp = PatchMetaParser(inst_name=inst_name,
                                 patient_id=patient_id,
                                 get_patch_gt=self.get_gt_,
                                 **self.hyper_)
            for _, slide_s in patient_s.iterrows():
                slide_instance = {
                    "name": f"{patient_id}/{slide_s['mosaic']}",
                    "label": self.get_gt_(slide_s, None),
                    "patches": mp.process_slide(slide_s)
                }
                if len(slide_instance):
                    instances.append(slide_instance)
        return instances

    def get_patient_instances(self):
        instances = []
        group_keys = [
            "institution", "patient",
            self.get_gt_.get_gt_col_name(self.df_)
        ]
        df_gb = tqdm(self.df_.groupby(group_keys, dropna=False))
        for ((inst_name, patient_id, prime_label), patient_s) in df_gb:
            if pd.isna(prime_label): prime_label = "None"
            logging.info(f"grouped patient ({inst_name}, {patient_id}, " +
                         f"{prime_label}, {len(patient_s)})")
            mp = PatchMetaParser(inst_name=inst_name,
                                 patient_id=patient_id,
                                 get_patch_gt=self.get_gt_,
                                 **self.hyper_)

            patient_instance = {
                "name": patient_id,
                "label": prime_label,
                "patches": mp.process_slides(patient_s)
            }
            if len(patient_instance):
                instances.append(patient_instance)
        return instances

    def get_hierarchical_instances(self):
        instances = []
        group_keys = [
            "institution", "patient",
            self.get_gt_.get_gt_col_name(self.df_)
        ]
        df_gb = tqdm(self.df_.groupby(group_keys, dropna=False))
        for ((inst_name, patient_id, prime_label), patient_s) in df_gb:

            if pd.isna(prime_label): prime_label = "None"
            logging.info(f"grouped patient ({inst_name}, {patient_id}, " +
                         f"{prime_label}, {len(patient_s)})")
            mp = PatchMetaParser(inst_name=inst_name,
                                 patient_id=patient_id,
                                 get_patch_gt=self.get_gt_,
                                 **self.hyper_)

            slides = [{
                "name": f"{patient_id}/{slide_s['mosaic']}",
                "label": self.get_gt_(slide_s, None),
                "patches": mp.process_slide(slide_s)
            } for _, slide_s in patient_s.iterrows()]
            slides = [s for s in slides if len(s)]

            if len(slides):
                instances.append({
                    "name": patient_id,
                    "label": prime_label,
                    "slides": slides
                })
        return instances


class SRHMetaParser(ABC):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    def make_im_path(self, patch_id: str, slide_s: pd.Series):
        """Parser for patch data.

        Produces path to the image
        """
        path = os.path.join(slide_s["institution"], slide_s["patient"],
                            str(slide_s.mosaic), "patches", patch_id)
        if not (path.endswith(".tif") or path.endswith(".tiff")
                or path.endswith(".png")):
            path += ".tif"

        return path

    def make_emb_path(self, patch_id: str, slide_s: pd.Series):
        """Parser for patch data.

        Produces partial path requiring the comment and .pt suffix to the pt
        file that stores the data for the slides
        """
        return os.path.join(slide_s["institution"], slide_s["patient"],
                            str(slide_s.mosaic),
                            f"{slide_s['patient']}.{slide_s['mosaic']}")

    @abstractmethod
    def process_slides(self):
        raise NotImplementedError()

    @abstractmethod
    def process_slide(self):
        raise NotImplementedError()

    @abstractmethod
    def ceil_instance_thres(self):
        raise NotImplementedError()


class PatchMetaParser(SRHMetaParser):
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

    def __init__(self,
                 inst_name: str,
                 patient_id: str,
                 data_root: str,
                 seg_model: str,
                 slide_patch_thres: Optional[int] = None,
                 use_emb: bool = False,
                 get_patch_gt: Optional[Callable] = None):
        """Inits the SRH Metadata parser"""
        self.data_root_ = data_root
        self.seg_model_ = seg_model
        self.slide_patch_thres_ = slide_patch_thres
        self.label_parser_ = get_patch_gt
        meta_file = os.path.join(self.data_root_, inst_name, patient_id,
                                 f"{patient_id}_meta.json")
        with open(meta_file) as fd:
            self.p_meta_ = json.load(fd)

        if use_emb:
            self.patch_path_func_ = self.make_emb_path
        else:
            self.patch_path_func_ = self.make_im_path

    def process_slides(self, patient_s: pd.Series):
        return list(
            itertools.chain(*[
                self.process_slide(slide_s)
                for _, slide_s in patient_s.iterrows()
            ]))

    def process_slide(self, slide_s: pd.Series):
        slide_name = str(slide_s.mosaic)
        if slide_name not in self.p_meta_["slides"]:
            logging.warning(
                f"Slide {slide_s['patient']}/{slide_s['mosaic']} DNE")
            return []

        all_patches_slide = self.p_meta_["slides"][slide_name]["predictions"][
            self.seg_model_]
        patch_code_decoded = patch_code_to_list(slide_s["patch_code"])
        if patch_code_decoded == ['all']:
            patch_code_decoded = list(all_patches_slide.keys())
        patch_code_diff = set(patch_code_decoded).difference(
            all_patches_slide.keys())
        if patch_code_diff:
            logging.warning(f"Slide {slide_s.patient} - {slide_s.mosaic} " +
                            f"does not have any patches in {patch_code_diff}")

        slide_instances = [{
            "im_path":
            self.patch_path_func_(p_name, slide_s),
            "label": (self.label_parser_.get_gt(slide_s, p_code)
                      if self.label_parser_ else None),
            "patch_name":
            p_name
        } for p_code in patch_code_decoded
                           for p_name in all_patches_slide[p_code]]

        if len(slide_instances) > 0:
            if self.label_parser_:
                label_clause = f"{self.label_parser_.make_gt_list(slide_s)}"
            else:
                label_clause = ""

            logging.debug(
                f"Slide {slide_s['patient']}/{slide_s['mosaic']} OK:" +
                f"{len(slide_instances)} " + label_clause)

        else:
            logging.warning(
                f"Slide {slide_s['patient']}/{slide_s['mosaic']} Empty")

        if self.slide_patch_thres_:
            slide_instances = self.ceil_instance_thres(slide_instances)
        return slide_instances

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

        num_repeat = math.ceil(self.slide_patch_thres_ / len(instances))
        random.shuffle(instances)
        instances_repeated = list(
            itertools.chain(*itertools.repeat(instances, num_repeat)))
        return sorted(instances_repeated[:self.slide_patch_thres_])


if __name__ == "__main__":
    from yaml import safe_load

    parser_params = """
        data_root: /nfs/umms-tocho-mr/dropbox/data/root_he_db
        seg_model: 08dc928c
        slide_patch_thres: null
        df: /nfs/turbo/umms-tocho-ns/data/data_splits/he_all/he_toy.csv
        use_emb: False
        use_patch_code_as_label: True
        primary_label_idx: 0
    """

    instances = PatchCSVParser(**safe_load(parser_params))(level="patch")

    import pdb
    pdb.set_trace()
