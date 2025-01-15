import argparse
import os
from os.path import join as opj
import logging
import uuid
import pandas as pd
import openslide
from tqdm import tqdm
import numpy as np

from madeleine.hest_modules.segmentation import TissueSegmenter
from madeleine.hest_modules.wsi import get_pixel_size, OpenSlideWSI

# File extensions for slide images
EXTENSIONS = ['.svs', '.mrxs', '.tiff', '.tif', '.TIFF', '.ndpi']


def process(csv_fname,
            slide_dir,
            out_dir,
            patch_mag,
            patch_size,
            process_batch_size=1):
    fnames_df = pd.read_csv(csv_fname, dtype=str)
    env_var = dict(os.environ)

    if "SLURM_ARRAY_TASK_ID" in env_var:
        taskid = int(env_var["SLURM_ARRAY_TASK_ID"])
    else:
        taskid = 0

    logging.info(f"GOT TASK ID {taskid}")

    start = process_batch_size * taskid
    end = min(process_batch_size * (taskid + 1), len(fnames_df))

    fnames = fnames_df.iloc[start:end]["svs_fname"].tolist()

    #fnames = [fname for fname in os.listdir(slide_dir) if any(fname.endswith(ext) for ext in EXTENSIONS)]
    logging.info(
        f'Running segmentation, patching, and feature extraction on {len(fnames)} slides.'
    )

    # Create necessary directories
    out_dir = os.path.join(
        out_dir,
        uuid.uuid4().hex[:8] + '_nWSI{}_mag{}x_patchsize{}_taskid{}'.format(
            len(fnames), patch_mag, patch_size, taskid))
    logging.info(f"Exp root {out_dir}")

    seg_path = os.path.join(out_dir, 'segmentation')
    os.makedirs(seg_path, exist_ok=True)

    patch_path = os.path.join(out_dir, 'patches')
    os.makedirs(patch_path, exist_ok=True)

    # create tissue segmenter and tile embedder
    segmenter = TissueSegmenter(save_path=seg_path,
                                batch_size=32,
                                num_workers=4)

    for fn in tqdm(fnames):
        try:

            # 1. read slide
            wsi = OpenSlideWSI(openslide.OpenSlide(os.path.join(slide_dir,
                                                                fn)))
            pixel_size = get_pixel_size(wsi.img)
            fn_no_extension = os.path.splitext(os.path.basename(fn))[0]

            # 2. segment tissue
            gdf_contours = segmenter.segment_tissue(wsi=wsi,
                                                    pixel_size=pixel_size,
                                                    save_bn=fn_no_extension)

            patcher = wsi.create_patcher(patch_size=patch_size,
                                         src_pixel_size=1,
                                         dst_pixel_size=1,
                                         overlap=0,
                                         mask=gdf_contours,
                                         coords_only=True)
            patch_cords = patcher.valid_coords
            np.save(opj(patch_path, f"{fn_no_extension}_coords.npy"),
                    patch_cords)

        except:
            logging.warning(f"failed - {fn}")
    logging.info('Done')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_dir",
                        type=str,
                        help="Directory with slides.",
                        default=None)
    parser.add_argument(
        "--local_dir",
        type=str,
        help=
        "Where to save tissue segmentation, patch coords, and patch embeddings.",
        default='./../data/downstream')
    parser.add_argument(
        "--patch_mag",
        type=int,
        help="Magnification at which patching operates. Default to 10x.",
        default=10)
    parser.add_argument("--patch_size",
                        type=int,
                        help="Patch size. Default to 256.",
                        default=256)
    parser.add_argument("--slide_fnames",
                        type=str,
                        help="CSV fname",
                        default=None)
    parser.add_argument("--process_batch_size",
                        type=int,
                        help="number of slides to process at once",
                        default=10)
    args = parser.parse_args()

    logging_format_str = ("[%(levelname)-s|%(asctime)s|%(name)s|" +
                          "%(filename)s:%(lineno)d|%(funcName)s] %(message)s")
    logging.basicConfig(level=logging.INFO,
                        format=logging_format_str,
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler()],
                        force=True)

    logging.info("Patching log")
    process(args.slide_fnames, args.slide_dir, args.local_dir, args.patch_mag,
            args.patch_size, args.process_batch_size)
