import torch
import numpy as np
from os.path import join as opj
import pandas
import logging
import os
from tqdm import tqdm


def proc_one_file(in_file: str):
    out_mm_name = in_file.split("/")[-1].replace(".pt", ".dat")
    out_coords_name = in_file.split("/")[-1].replace(".pt", ".coords.pt")
    out_dir = "/" + opj(*in_file.split("/")[:-1])

    data = torch.load(in_file)

    fd = np.memmap(opj(out_dir, out_mm_name),
                   dtype="float32",
                   mode="w+",
                   shape=data["embeddings"].shape)
    fd[:] = data["embeddings"].numpy()
    fd.flush()

    torch.save(torch.tensor(data["coords"]), opj(out_dir, out_coords_name))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format=
        "[%(levelname)-s|%(asctime)s|%(filename)s:%(lineno)d|%(funcName)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()])
    logging.info("Whatever Log")

    all_files = pandas.read_csv("/home/chengjia/sptf_slides_todo.txt")

    env_var = dict(os.environ)
    if "SLURM_ARRAY_TASK_ID" in env_var:
        taskid = int(env_var["SLURM_ARRAY_TASK_ID"]) + 3000
        logging.info("using task id %d / %d", taskid, len(all_files))
        logging.info(all_files["file"][taskid])
    else:
        taskid = 0
        logging.warning("using default id 0 / %d", len(all_files))
        logging.warning(all_files["file"][taskid])

    proc_one_file(all_files["file"][taskid])
