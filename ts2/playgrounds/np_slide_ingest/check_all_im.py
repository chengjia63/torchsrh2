from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from os.path import join as opj
import openslide

all_meta  = pd.read_csv("/nfs/mm-isilon/brainscans/dropbox/Slide_Incoming/metadata/master_spreadsheet/matched_su_mxa.csv")
svs_root = "/nfs/mm-isilon/brainscans/dropbox/Slide_Incoming/svs"

for i in range(2,9):
    currdf = all_meta.iloc[i*1000:(i+1)*1000]
    print(currdf)

    pdf = matplotlib.backends.backend_pdf.PdfPages(f"all_output_thumb_{i}.pdf")

    for i, s in tqdm(currdf.iterrows()):

        slide = openslide.OpenSlide(opj(svs_root, s["path"]))

        fig, axs = plt.subplots(1,3,  figsize=(12,3), width_ratios=[1, 3,3])
        axs[0].imshow(slide.associated_images["label"])
        axs[0].set_title(s["mxa_code"])
        axs[1].imshow(slide.associated_images["macro"])
        axs[2].imshow(slide.associated_images["thumbnail"])
        axs[1].set_title(s["path"].split("/")[-1]+"\n"+s['UM Accession']+" / "+s['Block']+" / "+s["Stain"])

        for ax in axs:
            ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close() 
