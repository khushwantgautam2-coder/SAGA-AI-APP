"""SAGA-AI -- simplified public deployment (Streamlit Community Cloud).

No sidebar, no batch evaluation, no LAN share card (meaningless in a hosted
container). Weights are fixed and pulled from a private Hugging Face repo via
ensure_weights_from_hf(). For the full-featured local/Mac app, see app.py.
"""

import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from cvms_core import (
    app_header,
    bgr_to_b64,
    compute_cvms_score,
    draw_landmarks,
    ensure_weights_from_hf,
    find_weight_files,
    guess_default,
    inject_theme,
    lightbox_image,
    load_net1,
    load_net2,
    run_inference,
    stat_cards_single,
)

st.set_page_config(page_title="SAGA-AI — Skeletal Age and Growth Assessment", layout="wide")
inject_theme()

ensure_weights_from_hf()
weight_files = find_weight_files()
net1_path = guess_default(weight_files, ["net1", "bbox"])
net2_path = guess_default(weight_files, ["net2", "landmark", "hrnet"])
models_ready = bool(net1_path and net2_path)

app_header(models_ready)
st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

if not models_ready:
    st.error("Model weights are not available. Please check the deployment configuration.")
else:
    st.markdown('<div style="font-size:15px; font-weight:600; margin-bottom: 12px;">Single Image Inference</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a lateral cephalometric X-ray", type=["jpg", "jpeg", "png"])

    if uploaded:
        net1 = load_net1(net1_path)
        net2 = load_net2(net2_path)

        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Running inference..."):
            result = run_inference(net1, net2, img_bgr)

        if result is None:
            st.error("Net1 failed to produce a valid crop for this image.")
        else:
            score, stage = compute_cvms_score(result["pred_pts"])
            vis = draw_landmarks(result["crop"], result["pred_pts"])

            col1, col2 = st.columns(2)
            with col1:
                lightbox_image(bgr_to_b64(img_bgr), "Original X-ray")
            with col2:
                lightbox_image(bgr_to_b64(vis), "Cropped region with predicted landmarks")

            st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
            stat_cards_single(score, stage)

            with st.expander("Raw landmark coordinates (crop-relative)"):
                st.dataframe(pd.DataFrame(result["pred_pts"], columns=["x", "y"]), use_container_width=True)
    else:
        st.info("Upload an image to run the pipeline.")
