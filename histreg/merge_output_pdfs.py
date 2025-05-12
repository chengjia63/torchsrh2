import fitz  # PyMuPDF
import os
import pandas as pd
from os.path import join as opj
from datetime import datetime
from tqdm import tqdm
import pickle
import numpy as np
import einops

def decompose_affine_matrix_batched(matrices):
    """
    Decomposes a batch of 2x3 or 3x3 affine transformation matrices into:
    rotation (theta in degrees), scale (sx, sy), shear (shear in degrees), translation (tx, ty).

    Args:
        matrices (numpy.ndarray): A batch of affine transformation matrices of shape (B, 2, 3) or (B, 3, 3).

    Returns:
        dict: A dictionary containing arrays of 'rotation', 'scale_x', 'scale_y', 'shear', 'tx', 'ty'.
    """
    matrices = np.asarray(matrices)
    B = matrices.shape[0]  # Batch size

    if matrices.shape[1:] == (2, 3):
        padding = np.tile(np.array([[0, 0, 1]]), (B, 1, 1))
        matrices = np.concatenate([matrices, padding], axis=1)
    elif matrices.shape[1:] != (3, 3):
        raise ValueError("Input matrix must have shape (B, 2, 3) or (B, 3, 3).")

    A = matrices[:, :2, :2]  # Extract the 2x2 transformation submatrices
    tx, ty = matrices[:, 0, 2], matrices[:, 1, 2]  # Extract translations

    # Compute scale factors
    sx = np.linalg.norm(A[:, :, 0], axis=1)  # Length of first column

    theta = np.arctan2(A[:, 1, 0], A[:, 0, 0])
    msy = A[:, 0, 1] * np.cos(theta) + A[:, 1, 1] * np.sin(theta)

    sy = np.where(
        np.sin(theta) == 0,
        (A[:, 1, 1] - msy * np.sin(theta)) / np.cos(theta),
        (msy * np.cos(theta) - A[:, 0, 1]) / np.sin(theta)
    )

    shear = msy / sy

    return {
        "rotation": np.degrees(theta),
        "scale_x": sx,
        "scale_y": sy,
        "shear": np.degrees(shear),
        "tx": tx,
        "ty": ty
    }



def merge_two_pdfs(file1, file2, output_file):
    # Open the two PDF files
    pdf1 = fitz.open(file1)
    pdf2 = fitz.open(file2)

    # Create a new PDF
    new_pdf = fitz.open()

    # Add the first page from the first PDF
    new_pdf.insert_pdf(pdf1, from_page=0, to_page=0)

    # Add the first page from the second PDF
    new_pdf.insert_pdf(pdf2, from_page=0, to_page=0)

    # Save the new PDF to the output file
    new_pdf.save(output_file)

    # Close the files
    pdf1.close()
    pdf2.close()
    new_pdf.close()


def main():

    block_align_out_root = "/nfs/turbo/umms-tocho-snr/exp/chengjia/block_align_crop_rigid_ransac" #"/nfs/turbo/umms-tocho-snr/exp/chengjia/block_align_crop_affine_ransac"
    block_align_out_root = "/nfs/turbo/umms-tocho-snr/exp/chengjia/block_align_crop_affine_ransac"
    meta_root="/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/playgrounds/pixel_alignment/sections_annot2/meta/"
    block_align_viz_root = "./viz_out_crop_affine_ransac_0423"
    block_align_viz_root_bad = "./viz_out_crop_affine_ransac_0423_reject"
    align_finished = os.listdir(block_align_out_root)

    #im = set([i.removesuffix("_align.pkl") for i in align_finished if i.endswith("_align.pkl")])
    #to_review = sorted(im)

    to_review = pd.read_csv("data/to_reg_250320.csv")["block"].tolist()

    #previously_reviewed_root = "./viz_out_crop_affine"
    #if previously_reviewed_root:
    #    previously_reviewed = os.listdir(previously_reviewed_root)
    #    previously_reviewed = [pr.removesuffix("_mask_align.pdf") for pr in previously_reviewed]
    #    to_review = sorted(set(to_review).difference(previously_reviewed))

    #previously_accepted_blocks = pd.read_csv("accepted.csv")["block"] 

    #to_review = sorted(set(to_review).difference(previously_accepted_blocks))
    os.makedirs(block_align_viz_root, exist_ok=True)
    os.makedirs(block_align_viz_root_bad, exist_ok=True)


    out_fname = opj(block_align_viz_root,
                     f"to_review_{datetime.now().strftime('%y%m%d')}.csv") 
    pd.DataFrame(to_review, columns=["block"]).to_csv(out_fname,
                                                         index=False)

    for tr in tqdm(to_review):

        try:
            block_meta = pd.read_csv(opj(meta_root, f"{tr}_sections_annot_meta.csv"))

            stain_list = sorted(set(block_meta[~(block_meta["comment"]=="RM")]["Stain"].tolist()))
            with open(opj(block_align_out_root, f"{tr}_align.pkl"), "rb") as fd:
                align_results = pickle.load(fd)

            decomposed_params = pd.DataFrame(decompose_affine_matrix_batched(einops.rearrange(align_results["matrices"], "he ihc mh mw -> (he ihc) mh mw")))
            
            reject = ((decomposed_params["shear"]>2).any() or
                ((decomposed_params["scale_x"] - 1).abs() > 0.2).any() or
                ((decomposed_params["scale_y"] - 1).abs() > 0.2).any() or
                (align_results["num_matches"] <= 50).any())

            if reject:
                curr_out_dir = block_align_viz_root_bad
            else:
                curr_out_dir = block_align_viz_root

            first =  opj(block_align_out_root, f"{tr}_im_align.pdf")
            second = opj(block_align_out_root, f"{tr}_mask_align.pdf")
            merge_two_pdfs(first, second, opj(curr_out_dir, f"{tr}_mask_align.pdf"))
        except:
            print(f"no viz - {tr}")

if __name__=="__main__": main()
