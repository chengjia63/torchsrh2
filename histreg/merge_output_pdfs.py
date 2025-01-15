import fitz  # PyMuPDF
import os
import pandas as pd
from os.path import join as opj
from datetime import datetime
from tqdm import tqdm
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

    block_align_out_root = "/nfs/turbo/umms-tocho-snr/exp/chengjia/block_align/"
    block_align_viz_root = "./viz_out"
    existing = os.listdir(block_align_out_root)
    im = set([i.removesuffix("_im_align.pdf") for i in existing if i.endswith("_im_align.pdf")])
    mask = set([i.removesuffix("_mask_align.pdf") for i in existing if i.endswith("_mask_align.pdf")])
    to_review = sorted(im.intersection(mask))

    out_fname = opj(block_align_viz_root,
                     f"to_review_{datetime.now().strftime('%y%m%d')}.csv") 
    pd.DataFrame(to_review, columns=["block"]).to_csv(out_fname,
                                                         index=False)

    for tr in tqdm(to_review):
        first =  opj(block_align_out_root, f"{tr}_im_align.pdf")
        second = opj(block_align_out_root, f"{tr}_mask_align.pdf")
        merge_two_pdfs(first, second, opj(block_align_viz_root, f"{tr}_mask_align.pdf"))

if __name__=="__main__": main()
