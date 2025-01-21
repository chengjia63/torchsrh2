import os
from datetime import datetime
import pandas as pd
from os.path import join as opj


def main():
    block_annot_path = "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/playgrounds/pixel_alignment/sections_annot2/block_annot/"
    block_align_prev = "/nfs/turbo/umms-tocho-snr/exp/chengjia/block_align_0120_rigid/"
    out_data_dir = "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/histreg/data"

    finished_annot_blocks = os.listdir(block_annot_path)
    finished_annot_blocks = [
        f.removesuffix(".tiff") for f in finished_annot_blocks
        if f.endswith(".tiff")
    ]
    previously_reg_blocks = os.listdir(block_align_prev)
    previously_reg_blocks = [
        f.removesuffix("_align.pkl") for f in previously_reg_blocks
        if f.endswith("_align.pkl")
    ]

    to_reg = sorted(
        set(finished_annot_blocks).difference(previously_reg_blocks))

    print(f"num slide to reg: {len(to_reg)}")
    out_fname = opj(out_data_dir,
                    f"to_reg_{datetime.now().strftime('%y%m%d')}.csv")
    print(out_fname)
    assert not os.path.exists(out_fname)

    pd.DataFrame(to_reg, columns=["block"]).to_csv(out_fname, index=False)


if __name__ == "__main__": main()
