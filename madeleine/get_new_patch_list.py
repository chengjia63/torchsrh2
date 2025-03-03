import pandas as pd
import os
from os.path import join as opj
from datetime import datetime
from glob import glob

def main():
    scanning_master_meta_fname = "/nfs/mm-isilon/brainscans/dropbox/Slide_Incoming/metadata/master_spreadsheet/matched_su_mxa.csv"
    patched_pkl_root = "/nfs/turbo/umms-tocho-snr/exp/chengjia/madeleine_patch/*/segmentation/pkl/"
    to_patch_data_dir = "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/madeleine/data"
    meta = pd.read_csv(scanning_master_meta_fname)
    to_patch = meta[meta["Stain"] == "H&E"]["path"].dropna().tolist()

    already_patched = glob(opj(patched_pkl_root, "*_tissue_mask.pkl"))
    already_patched = [
        i.split("/")[-1].removesuffix("_tissue_mask.pkl") for i in already_patched
        if i.endswith("_tissue_mask.pkl")
    ]

    to_patch = [
        i for i in to_patch
        if not (i.split("/")[-1].removesuffix(".svs") in already_patched)
    ]

    print(f"num slide to path: {len(to_patch)}")

    out_fname = f"to_patch_{datetime.now().strftime('%y%m%d')}.csv"
    out_full_fname = opj(to_patch_data_dir, out_fname)
    print(out_full_fname)
    assert not os.path.exists(out_full_fname)

    pd.DataFrame(to_patch, columns=["svs_fname"]).to_csv(out_full_fname,
                                                         index=False)


if __name__ == "__main__": main()
