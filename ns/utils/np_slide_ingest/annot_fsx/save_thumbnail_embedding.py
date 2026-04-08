import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from sklearn.manifold import TSNE
import altair as alt
import base64
from io import BytesIO
import pickle
from tqdm import tqdm
import openslide
import matplotlib.pyplot as plt
import numpy as np
import glob
import json
import pandas as pd
from tqdm import tqdm
from typing import List
import yaml
import os
from os.path import join as opj
import re

def image_formatter(im):
    with BytesIO() as buffer:
        Image.fromarray(im).save(buffer, 'png')
        data = base64.encodebytes(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{data}"


all_meta = pd.read_csv("/nfs/mm-isilon/brainscans/dropbox/Slide_Incoming/metadata/master_spreadsheet/matched_su_mxa.csv")

currdf = all_meta.loc[all_meta["Block"].str.contains("FS")]
currdf = currdf.loc[currdf["Stain"]=="H&E"]
#currdf = currdf.sample(20)

svs_root = "/nfs/mm-isilon/brainscans/dropbox/Slide_Incoming/svs"
paths = []
thumbnails = []
for i, s in tqdm(currdf.fillna("").iterrows(), total=len(currdf)):
    slide = openslide.OpenSlide(opj(svs_root, s["path"]))
    paths.append(s["path"])
    
    image = slide.associated_images["macro"]
    width, height = image.size
    thumbnails.append(np.array(image.resize((width // 2, height // 2))))

with open("path.pickle", "wb") as fd:
    pickle.dump(paths, fd)

with open("thumbnails.pickle", "wb") as fd:
    pickle.dump(thumbnails, fd)

model = AutoModel.from_pretrained("/nfs/mm-isilon/brainscans/dropbox/exp/models/dinov2/dinov2-large").cuda()
processor = AutoProcessor.from_pretrained("/nfs/mm-isilon/brainscans/dropbox/exp/models/dinov2/dinov2-large")

embs = []
for i in tqdm(thumbnails):
    inputs = processor(images=Image.fromarray(np.uint8(i)), return_tensors="pt")
    inputs = {key: value.cuda() for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        embs.append(outputs.last_hidden_state[0,0,:].detach().cpu().numpy())

with open("embs.pickle", "wb") as fd:
    pickle.dump(embs, fd)
