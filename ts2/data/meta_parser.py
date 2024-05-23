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

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def __call__(self):
        with open(opj(self.cache_dir, "instances.json")) as fd:
            data = json.load(fd)

        with open(opj(self.cache_dir, "mmap_info.json")) as fd:
            ts = json.load(fd)

        return data, ts

    def get_meta(self):
        with open(opj(self.cache_dir, "meta.json")) as fd:
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
                 df: Union[pd.DataFrame, str], which_patch_path_func: str,
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
            "which_patch_path_func": which_patch_path_func,
        }

        assert len(slide_list) == len(set(slide_list))  # slides are unique

    def __call__(self, cache_dir: str, level: str):
        if level == DiscriminationLevel.PATCH:
            inst, mmap_info = self.get_patch_instances()
        elif level == DiscriminationLevel.SLIDE:
            inst, mmap_info = self.get_slide_instances()
        elif level == DiscriminationLevel.PATIENT:
            inst, mmap_info = self.get_patient_instances()
        elif level == DiscriminationLevel.HIERARCHICAL:
            inst, mmap_info = self.get_hierarchical_instances()
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
        raise ValueError(
            "Patch CSV Parser does not support cell level instances")

    def get_patch_instances(
            self) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
        patch_instances = []
        tensor_shapes = {}

        for (inst_name, patient_id), patient_s in tqdm(
                self.df_.groupby(["institution", "patient"], dropna=False)):
            inst_i, shape_i = PatchMetaParser(
                inst_name=inst_name,
                patient_id=patient_id,
                get_patch_gt=self.get_gt_,
                **self.hyper_).process_all_slides(patient_s, keep_label=True)
            patch_instances.extend(inst_i)
            tensor_shapes.update(shape_i)
        return patch_instances, tensor_shapes

    def get_slide_instances(
            self) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
        slide_instances = []
        tensor_shapes = {}

        df_gb = tqdm(self.df_.groupby(["institution", "patient"],
                                      dropna=False))
        for (inst_name, patient_id), patient_s in df_gb:
            mp_i = PatchMetaParser(inst_name=inst_name,
                                   patient_id=patient_id,
                                   get_patch_gt=self.get_gt_,
                                   **self.hyper_)
            for _, slide_s in patient_s.iterrows():
                patches_ij, slide_shape_ij = mp_i.process_slide(slide_s, keep_label=False)

                if patches_ij:
                    slide_instance_ij = {
                        "name": f"{patient_id}-{slide_s['mosaic']}",
                        "label": self.get_gt_(slide_s, None),
                        "patches": patches_ij
                    }
                    slide_instances.append(slide_instance_ij)
                    tensor_shapes.update(slide_shape_ij)

        return slide_instances, tensor_shapes

    def get_patient_instances(
            self) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
        patient_instances = []
        tensor_shapes = {}

        group_keys = [
            "institution", "patient",
            self.get_gt_.get_gt_col_name(self.df_)
        ]
        df_gb = tqdm(self.df_.groupby(group_keys, dropna=False))
        for ((inst_name, patient_id, prime_label), patient_s) in df_gb:
            if pd.isna(prime_label): prime_label = "None"
            logging.info(f"grouped patient ({inst_name}, {patient_id}, " +
                         f"{prime_label}, {len(patient_s)})")
            mp_i = PatchMetaParser(inst_name=inst_name,
                                   patient_id=patient_id,
                                   get_patch_gt=self.get_gt_,
                                   **self.hyper_)

            patches_i, slides_shape_i = mp_i.process_all_slides(patient_s)

            if patches_i:
                patient_instance_i = {
                    "name": patient_id,
                    "label": prime_label,
                    "patches": patches_i
                }

                patient_instances.append(patient_instance_i)
                tensor_shapes.update(slides_shape_i)

        return patient_instances, tensor_shapes

    def get_hierarchical_instances(
            self) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
        patient_instances = []
        mmap_info = {}

        group_keys = [
            "institution", "patient",
            self.get_gt_.get_gt_col_name(self.df_)
        ]
        df_gb = tqdm(self.df_.groupby(group_keys, dropna=False))
        for ((inst_name, patient_id, prime_label), patient_s) in df_gb:

            if pd.isna(prime_label): prime_label = "None"
            logging.info(f"grouped patient ({inst_name}, {patient_id}, " +
                         f"{prime_label}, {len(patient_s)})")
            mp_i = PatchMetaParser(inst_name=inst_name,
                                   patient_id=patient_id,
                                   get_patch_gt=self.get_gt_,
                                   **self.hyper_)

            slides_i = []
            for _, slide_s in patient_s.iterrows():
                patches_ij, slide_shape = mp_i.process_slide(slide_s)

                if patches_ij:
                    slides_i.append({
                        "name": f"{patient_id}-{slide_s['mosaic']}",
                        "label": self.get_gt_(slide_s, None),
                        "patches": patches_ij
                    })
                    mmap_info.update(slide_shape)

            if slides_i:
                patient_instances.append({
                    "name": patient_id,
                    "label": prime_label,
                    "slides": slides_i
                })

        return patient_instances, mmap_info


class SRHMetaParser(ABC):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    def make_im_path(self, patch_id: str, slide_s: pd.Series):
        """Parser for patch data.

        Produces path to the image
        """
        raise NotImplementedError()
        path = os.path.join(slide_s["institution"], slide_s["patient"],
                            str(slide_s.mosaic), "patches", patch_id)
        if not (path.endswith(".tif") or path.endswith(".tiff")
                or path.endswith(".png")):
            path += ".tif"

        return path

    def make_slide_emb_path(self, slide_s: pd.Series):
        """Parser for patch data.

        Produces partial path requiring the comment and .pt suffix to the pt
        file that stores the data for the slides
        """
        return os.path.join(slide_s["institution"], slide_s["patient"],
                            str(slide_s.mosaic),
                            f"{slide_s['patient']}.{slide_s['mosaic']}")

    def make_slide_memmap_path(self, slide_s: pd.Series):
        """Parser for patch data.

        Memmap for wholeslide
        """
        return os.path.join(
            slide_s["institution"], slide_s["patient"], str(slide_s.mosaic),
            "patches", f"{slide_s['patient']}-{slide_s['mosaic']}-patches.dat")

    @abstractmethod
    def process_all_slides(self, patient_s, keep_label):
        raise NotImplementedError()

    @abstractmethod
    def process_slide(self, slide_s, keep_label):
        raise NotImplementedError()

    @abstractmethod
    def ceil_instance_thres(self, instances: List[Any]):
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
                 which_patch_path_func: str = "make_im_path",
                 get_patch_gt: Optional[Callable] = None):
        """Inits the SRH Metadata parser"""
        self.data_root_ = data_root
        self.seg_model_ = seg_model
        self.slide_patch_thres_ = slide_patch_thres
        self.label_parser_ = get_patch_gt
        meta_file = os.path.join(self.data_root_, inst_name, patient_id,
                                 f"{patient_id}_meta.json")
        logging.info("Opening %s", meta_file)
        with open(meta_file) as fd:
            self.p_meta_ = json.load(fd)

        self.patch_path_func_ = {
            "make_slide_emb_path": self.make_slide_emb_path,
            "make_slide_memmap_path": self.make_slide_memmap_path
        }[which_patch_path_func]

    def process_all_slides(self, patient_s: pd.Series, keep_label=False):

        inst = []
        mmap_info = {}

        for _, slide_s in patient_s.iterrows():
            inst_i, shape_i = self.process_slide(slide_s=slide_s,
                                                 keep_label=keep_label)
            inst.extend(inst_i)
            mmap_info.update(shape_i)

        return inst, mmap_info

    def process_slide(self, slide_s: pd.Series, keep_label: bool):  # one slide
        slide_name = str(slide_s.mosaic)
        if slide_name not in self.p_meta_["slides"]:
            logging.warning("Slide %s/%s DNE" %
                            (slide_s['patient'], slide_s['mosaic']))
            return [], {}

        patient_slide_name = f"{slide_s.patient}-{slide_s.mosaic}"

        if "predictions" not in self.p_meta_["slides"][slide_name]:
            logging.warning("Slide %s/%s DNE" %
                            (slide_s['patient'], slide_s['mosaic']))
            return [], {}

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

        def make_optional_label(p_code):
            if keep_label and self.label_parser_:
                return {"label": self.label_parser_.get_gt(slide_s, p_code)}

            if keep_label:
                return {"label": None}

            return {}

        slide_instances = [{
            "slide_name":
            patient_slide_name,
            "patch_name":
            p_name.removeprefix(patient_slide_name).removeprefix("-"),
            "patch_idx":
            all_patches_slide[p_code][p_name],
            **make_optional_label(p_code),
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
            return [], {}

        if self.slide_patch_thres_:
            slide_instances = self.ceil_instance_thres(slide_instances)

        slide_mmap_info = {
            patient_slide_name: {
                "path": self.patch_path_func_(slide_s),
                "shape": all_patches_slide["tensor_shape"]
            }
        }
        return slide_instances, slide_mmap_info

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

        return sorted(instances_repeated[:self.slide_patch_thres_],
                      key=lambda x: x["patch_idx"])
