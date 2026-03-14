import numpy as np
import pickle
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from histreg.manual_alignment import ManualBlockAligner, decompose_affine_matrix, reconstruct_affine_matrix, downsample_cv2
import uvicorn
import os
import cv2
import base64
from PIL import Image
from io import BytesIO
import base64

app = FastAPI()
cf = {                                                          
    "infra":{                                                           
        "annot_root": "/nfs/turbo/umms-tocho/code/chengjia/torchsrh2/ts2/playgrounds/pixel_alignment/sections_annot2/",
        "svs_root": "/nfs/mm-isilon/brainscans/dropbox/Slide_Incoming/svs"
    },                                                                  
    "prev_exp": {                                                       
        #"out_dir": "/nfs/turbo/umms-tocho-snr/exp/chengjia/block_align_crop_rigid_ransac"
        "out_dir": "/nfs/turbo/umms-tocho-snr/exp/chengjia/block_align_crop_affine_ransac"
    },                                                                  
    "manual": {                                                         
        "out_dir": "/nfs/turbo/umms-tocho-snr/exp/chengjia/block_align_manual"
    }                                                                   
}     

aligner = ManualBlockAligner(cf)



@app.get("/")
def home():
    return HTMLResponse(
        """
        <html>
            <head>
                <style>
                    body {
                        width: 1080px;
                        margin: auto;
                        text-align: center;
                        padding-top: 50px;
                        font-family: Helvetica, Arial, sans-serif;
                    }
                    #statusBar {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 30px;
                        background-color: green;
                        text-align: center;
                        line-height: 30px;
                        font-weight: bold;
                    }
                    #visualizationContainer {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        margin-top: 20px;
                        gap: 20px;
                    }
                    #visualizationContainer img {
                        height: auto;
                        object-fit: contain;
                    }
                </style>
            </head>
            <body>
                <div id="statusBar" style="height: 30px; width: 100%; background-color: green; text-align: center; line-height: 30px; font-weight: bold;">Ready</div>
                <h1>Manual Alignment</h1>
                <label for="blockName">Block Name:</label>
                <input type="text" id="blockName" placeholder="Enter block name" />
                <button onclick="loadBlock()">Load Block</button>
                <br><br>
                <label for="heIndex">H&E Index:</label>
                <select id="heIndex"></select>
                <label for="ihcIndex">IHC Index:</label>
                <select id="ihcIndex"></select>
                <button onclick="set_slide_pair()">Set slide pair</button>
                <br><br>
                <div id="visualizationContainer">
                    <img id="he_im" style="width: 24.5%;"/>
                    <img id="ihc_im" style="width: 24.5%;"/>
                    <img id="he_mask" style="width: 24.5%;"/>
                    <img id="ihc_mask" style="width: 24.5%;"/>
                </div>
                <div id="visualizationContainer">
                    <img id="stackedImage" style="width:49%;" />
                    <img id="overlayImage" style="width:49%;" />
                </div>
                <br>
                <label>Rotation:</label>
                <input type="number" id="rotation" min="-180" max="180" value="0" step="1" oninput="applyTransform()"/>
                <label>Scale X:</label>
                <input type="number" id="scale_x" min="0.5" max="2" step="0.05" value="1" oninput="applyTransform()" />
                <label>Scale Y:</label>
                <input type="number" id="scale_y" min="0.5" max="2" step="0.05" value="1" oninput="applyTransform()" />
                <label>Tx:</label>
                <input type="number" id="tx" min="-5000" max="5000" step="5" value="0" oninput="applyTransform()"/>
                <label>Ty:</label>
                <input type="number" id="ty" min="-5000" max="5000" step="5" value="0" oninput="applyTransform()" />

                <button onclick="applyTransform()">Apply</button>
                <button onclick="commit()">Commit</button>
                <button onclick="saveBlock()">Save Block</button>
                <script>
                    function setStatus(message, color) {
                        let statusBar = document.getElementById("statusBar");
                        statusBar.innerText = message;
                        statusBar.style.backgroundColor = color;
                    }
                    async function loadBlock() {
                        setStatus("Loading...", "red");
                        let blockName = document.getElementById("blockName").value;
                        let response = await fetch(`/load_block/?block=${blockName}`);
                        let data = await response.json();
                        let heDropdown = document.getElementById("heIndex");
                        let ihcDropdown = document.getElementById("ihcIndex");
                        heDropdown.innerHTML = data.he_indices.map(i => `<option value='${i}'>${i}</option>`).join("");
                        ihcDropdown.innerHTML = data.ihc_indices.map(i => `<option value='${i}'>${i}</option>`).join("");
                        setStatus("Ready", "green");
                    }
                    async function set_slide_pair() {
                        setStatus("Setting Slide Pair...", "red");
                        let heIndex = document.getElementById("heIndex").value;
                        let ihcIndex = document.getElementById("ihcIndex").value;
                        let response = await fetch(`/set_slide_pair/?he_i=${heIndex}&ihc_i=${ihcIndex}`);
                        let data = await response.json();
                        document.getElementById("rotation").value = data.rotation;
                        document.getElementById("scale_x").value = data.scale_x;
                        document.getElementById("scale_y").value = data.scale_y;
                        document.getElementById("tx").value = data.tx;
                        document.getElementById("ty").value = data.ty;
                        document.getElementById("he_im").src = "data:image/png;base64," + data.he_im;
                        document.getElementById("ihc_im").src = "data:image/png;base64," + data.ihc_im;
                        document.getElementById("he_mask").src = "data:image/png;base64," + data.he_mask;
                        document.getElementById("ihc_mask").src = "data:image/png;base64," + data.ihc_mask;
                        document.getElementById("stackedImage").src = "data:image/png;base64," + data.stacked;
                        document.getElementById("overlayImage").src = "data:image/png;base64," + data.overlay;
                        setStatus("Ready", "green");
                    }
                    async function applyTransform() {
                        setStatus("Applying Transform...", "red");
                        let heIndex = document.getElementById("heIndex").value;
                        let ihcIndex = document.getElementById("ihcIndex").value;
                        let transform = {
                            rotation: parseFloat(document.getElementById("rotation").value),
                            scale_x: parseFloat(document.getElementById("scale_x").value),
                            scale_y: parseFloat(document.getElementById("scale_y").value),
                            tx: parseFloat(document.getElementById("tx").value),
                            ty: parseFloat(document.getElementById("ty").value)
                        };
                        let response = await fetch(`/apply_transform/?he_i=${heIndex}&ihc_i=${ihcIndex}&rotation=${transform.rotation}&scale_x=${transform.scale_x}&scale_y=${transform.scale_y}&tx=${transform.tx}&ty=${transform.ty}`);
                        let data = await response.json();
                        document.getElementById("overlayImage").src = "data:image/png;base64," + data.stack;
                        document.getElementById("stackedImage").src = "data:image/png;base64," + data.overlay;
                        setStatus("Ready", "green");
                    }
                    async function commit() {
                        setStatus("Committing...", "red");
                        let heIndex = document.getElementById("heIndex").value;
                        let ihcIndex = document.getElementById("ihcIndex").value;
                        let transform = {
                            rotation: parseFloat(document.getElementById("rotation").value),
                            scale_x: parseFloat(document.getElementById("scale_x").value),
                            scale_y: parseFloat(document.getElementById("scale_y").value),
                            tx: parseFloat(document.getElementById("tx").value),
                            ty: parseFloat(document.getElementById("ty").value)
                        };
                        await fetch(`/commit/?he_i=${heIndex}&ihc_i=${ihcIndex}&rotation=${transform.rotation}&scale_x=${transform.scale_x}&scale_y=${transform.scale_y}&tx=${transform.tx}&ty=${transform.ty}`, {method: "POST"});
                        setStatus("Ready", "green");
                    }
                    async function saveBlock() {
                        setStatus("Saving...", "red");
                        await fetch(`/save_block/`, {method: "POST"});
                        setStatus("Ready", "green");
                    }
                </script>
            </body>
        </html>
        """
    )


def encode_image(image):
        image_pil = Image.fromarray(image.astype(np.uint8))
        buffered = BytesIO()
        image_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

@app.get("/load_block/")
def load_block(block: str):
    aligner.init_one_block(block)
    he_indices = list(range(len(aligner.curr_he)))
    ihc_indices = list(range(len(aligner.curr_block)))
    return JSONResponse(content={"he_indices": he_indices, "ihc_indices": ihc_indices})

@app.get("/set_slide_pair/")
def set_slide_pair(he_i: int, ihc_i: int):
    aligner.set_align_pair(he_i, ihc_i, do_viz=False)
   
    he_im_viz = np.array(aligner.he_thumbnail).astype(np.uint8)            
    ihc_im_viz = np.array(aligner.ihc_thumbnail).astype(np.uint8)          
    he_im_viz = downsample_cv2(he_im_viz, 4)
    ihc_im_viz = downsample_cv2(ihc_im_viz, 4)

    he_mask_viz = np.array(aligner.he_proc*255).astype(np.uint8)           
    ihc_mask_viz = np.array(aligner.ihc_proc*255).astype(np.uint8)         
    he_mask_viz = downsample_cv2(he_mask_viz,4)
    ihc_mask_viz = downsample_cv2(ihc_mask_viz,4)

    print(aligner.align_results["matrices"][he_i, ihc_i])
    print(aligner.align_results["matrices"][he_i, ihc_i])

    stacked_im, overlay = aligner.generate_transform_viz(aligner.align_results["matrices"][he_i, ihc_i])
    stacked_im = np.array(stacked_im*255).astype(np.uint8)
    transform = decompose_affine_matrix(aligner.align_results["matrices"][he_i, ihc_i])

    
    return JSONResponse(content={
        "rotation": transform["rotation"],
        "scale_x": transform["scale_x"],
        "scale_y": transform["scale_y"],
        "tx": transform["tx"],
        "ty": transform["ty"],
        "he_im": encode_image(he_im_viz),
        "ihc_im": encode_image(ihc_im_viz),
        "he_mask": encode_image(he_mask_viz),
        "ihc_mask": encode_image(ihc_mask_viz),
        "stacked": encode_image(stacked_im),
        "overlay": encode_image(overlay),
    })

@app.post("/commit/")
def commit(he_i: int, ihc_i: int, rotation: float = Query(...), scale_x: float = Query(...), scale_y: float = Query(...), tx: float = Query(...), ty: float = Query(...)):
    transform = {"rotation": rotation, "scale_x": scale_x, "scale_y": scale_y, "tx": tx, "ty": ty, "shear": 0}
    aligner.commit_slide_pair(guess=transform, do_viz=False)
    return JSONResponse(content={"message": "Transformation committed"})

@app.get("/apply_transform/")
def apply_transform(he_i: int, ihc_i: int, rotation: float, scale_x: float, scale_y: float, tx: float, ty: float):
    transform = {"rotation": rotation, "scale_x": scale_x, "scale_y": scale_y, "tx": tx, "ty": ty, "shear": 0}
    guess = reconstruct_affine_matrix(transform)
    stack, overlay = aligner.generate_transform_viz(guess)
    stack = np.array(stack*255).astype(np.uint8)
    return JSONResponse(content={
        "stack": encode_image(stack),
        "overlay": encode_image(overlay),
        })


@app.post("/save_block/")
def save_block():
    aligner.commit_save_block(do_save=True)
    return JSONResponse(content={"message": "Block saved successfully"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=48109)

