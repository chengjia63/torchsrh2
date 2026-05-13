import pandas as pd
from tqdm import tqdm
from ns.utils.histreg.align_pipe import get_sections_annot
import tifffile
import numpy as np
import os

matched_su_mxa_fname: str = "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ns/utils/scanning/out/matched_su_mxa.csv"
sections_annot_root: str = "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ns/utils/sections_annot/sections_annot2"

matched_su_mxa: pd.DataFrame = pd.read_csv(matched_su_mxa_fname, dtype=str)
blocks = (matched_su_mxa["UM Accession"] + "." + matched_su_mxa["Block"]).drop_duplicates()

no_he = []
single_block_only = []
for b in tqdm(blocks.tolist()):
    meta_path = os.path.join(sections_annot_root, "meta", f"{b}_sections_annot_meta.csv")

    if os.path.exists(meta_path):
        data = pd.read_csv(meta_path)

        if "H&E" not in data["Stain"].tolist():
            no_he.append(b)

        elif len(data) == 1:
            mask_annot = tifffile.imread(
               os.path.join(sections_annot_root, "block_annot", f"{b}.tiff")
            )
            if len(mask_annot.shape) == 3:  # 1 physical slide
                mask_annot = np.expand_dims(mask_annot, 0)
            section_mask_both = [get_sections_annot(m) for m in mask_annot]
            section_mask = [smb[0] for smb in section_mask_both]
            lengths = list(map(len, section_mask))

            if lengths[0] == 1:
                single_block_only.append(b)

pd.DataFrame({"block": single_block_only,"which": "one_section_only", "status":"ok"}).to_csv("one_section.csv", index=False)
pd.DataFrame({"block": no_he, "which": "no_he", "status":"abort"}).to_csv("no_he.csv", index=False)
