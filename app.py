"""SAGA-AI -- full-featured local/Mac app: sidebar with model-weight selection,
Single Image and Batch Evaluation modes, and a LAN share-link card.
For the simplified public deployment, see app_cloud.py."""

import os
import socket

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from cvms_core import (
    DEVICE,
    STAGE_DESC,
    compute_cvms_score,
    draw_landmarks,
    ensure_weights_from_hf,
    find_weight_files,
    guess_default,
    get_gt_points,
    inject_theme,
    app_header,
    bgr_to_b64,
    lightbox_image,
    load_net1,
    load_net2,
    logo_b64,
    parse_test_csv,
    results_table_html,
    run_inference,
    stat_cards_single,
)


def sidebar_header():
    b64 = logo_b64()
    logo_html = (
        f'<img src="data:image/png;base64,{b64}" style="width:100%;height:100%;object-fit:contain;" />'
        if b64 else ""
    )
    st.sidebar.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:22px;">
            <div style="width:34px; height:34px; border-radius:7px; background:white;
                        display:flex; align-items:center; justify-content:center;
                        overflow:hidden; flex-shrink:0;">
                {logo_html}
            </div>
            <div>
                <div style="font-size:14px; font-weight:600; color:var(--text);">SAGA-AI</div>
                <div style="font-size:10.5px; color:var(--text-muted); line-height:1.3;">Skeletal Age and Growth Assessment</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_lan_ip():
    """Best-effort LAN IP: opens a UDP socket toward a public address (sends no
    packets) purely to read back which local interface/IP the OS would route through."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def render_share_card():
    ip = get_lan_ip()
    port = st.get_option("server.port") or 8501
    url = f"http://{ip}:{port}"

    components.html(
        f"""
        <style>
            html, body {{ margin: 0; background: #14171c; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }}
        </style>
        <div style="background:#1b1f26; border:1px solid #313947; border-radius:12px;
                    padding:14px 18px; display:flex; align-items:center; justify-content:space-between;
                    gap:14px; box-sizing:border-box;">
            <div style="min-width:0;">
                <div style="font-size:11px; font-weight:600; letter-spacing:0.6px; text-transform:uppercase; color:#93a0ac;">
                    Share on your network
                </div>
                <a id="cvms-share-url" href="{url}" target="_blank"
                   style="display:block; font-size:13.5px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
                          color:#4fc7d1; text-decoration:none; margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                    {url}
                </a>
                <div style="font-size:11px; color:#93a0ac; margin-top:3px;">
                    Anyone on the same Wi-Fi/LAN can open this link
                </div>
            </div>
            <button id="cvms-copy-btn"
                    style="background:#4fc7d1; color:#0d1013; border:none; border-radius:8px;
                           padding:9px 16px; font-size:13px; font-weight:600; cursor:pointer;
                           white-space:nowrap; flex-shrink:0;">
                Copy link
            </button>
        </div>
        <script>
            const btn = document.getElementById('cvms-copy-btn');
            btn.addEventListener('click', function() {{
                const ta = document.createElement('textarea');
                ta.value = "{url}";
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.focus();
                ta.select();
                try {{
                    document.execCommand('copy');
                    btn.innerText = "Copied!";
                }} catch (e) {{
                    btn.innerText = "Copy failed";
                }}
                document.body.removeChild(ta);
                setTimeout(function() {{ btn.innerText = "Copy link"; }}, 1500);
            }});
        </script>
        """,
        height=100,
        scrolling=False,
    )


# ========= STREAMLIT UI =========
st.set_page_config(page_title="SAGA-AI — Skeletal Age and Growth Assessment", layout="wide")
inject_theme()

with st.sidebar:
    sidebar_header()
    st.markdown('<div class="cvms-card-label" style="margin-bottom: 8px;">Model Weights</div>', unsafe_allow_html=True)
    ensure_weights_from_hf()
    weight_files = find_weight_files()

    if weight_files:
        net1_default = guess_default(weight_files, ["net1", "bbox"])
        net2_default = guess_default(weight_files, ["net2", "landmark", "hrnet"])

        net1_path = st.selectbox(
            "Net1 weights (bbox)",
            weight_files,
            index=weight_files.index(net1_default) if net1_default in weight_files else 0,
        )
        net2_path = st.selectbox(
            "Net2 weights (landmarks)",
            weight_files,
            index=weight_files.index(net2_default) if net2_default in weight_files else 0,
        )
        if st.button("Rescan for weight files"):
            st.rerun()
    else:
        st.warning(f"No .pth files found in {os.getcwd()}")
        st.caption("Place your weight files here, then click rescan.")
        if st.button("Rescan for weight files"):
            st.rerun()
        net1_path = st.text_input("Net1 weights (bbox) — manual path", value="best_bbox_net1.pth")
        net2_path = st.text_input("Net2 weights (landmarks) — manual path", value="best_net2.pth")

    st.divider()
    mode = st.radio("Mode", ["Single Image", "Batch Evaluation"])

models_ready = os.path.exists(net1_path) and os.path.exists(net2_path)
if not models_ready:
    st.sidebar.warning("Weight files not found at the given paths yet.")

app_header(models_ready)
render_share_card()
st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

if mode == "Single Image":
    st.markdown('<div style="font-size:15px; font-weight:600; margin-bottom: 12px;">Single Image Inference</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a lateral cephalometric X-ray", type=["jpg", "jpeg", "png"])

    if uploaded and models_ready:
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
    elif uploaded and not models_ready:
        st.error("Select valid model weight files in the sidebar before running inference.")
    else:
        st.info("Upload an image to run the pipeline.")

else:
    st.markdown('<div style="font-size:15px; font-weight:600;">Batch Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:12.5px; color:var(--text-muted); margin-bottom: 14px;">'
        'Provide a VGG-style annotation CSV (<span class="cvms-mono">region_shape_attributes</span> '
        'with <span class="cvms-mono">cx</span>/<span class="cvms-mono">cy</span>) and the folder containing the corresponding images.'
        '</div>',
        unsafe_allow_html=True,
    )

    csv_file = st.file_uploader("Ground-truth CSV", type=["csv"])
    img_dir = st.text_input("Image directory", value="images")
    run_btn = st.button("Run batch evaluation", disabled=not (csv_file and models_ready))

    if run_btn:
        net1 = load_net1(net1_path)
        net2 = load_net2(net2_path)

        df = parse_test_csv(csv_file)
        filenames = df["filename"].unique()

        results = []
        correct = 0
        total = 0

        progress = st.progress(0.0, text="Starting...")
        for i, fname in enumerate(filenames):
            progress.progress((i + 1) / len(filenames), text=f"Processing {fname}")

            img_path = os.path.join(img_dir, fname)
            if not os.path.exists(img_path):
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            result = run_inference(net1, net2, img_bgr)
            if result is None:
                continue

            x1, y1, _, _ = result["bbox"]
            cw = result["crop"].shape[1]
            ch = result["crop"].shape[0]
            pred_pts = result["pred_pts"]

            gt_full = get_gt_points(df, fname)
            gt_pts = []
            for (gx, gy) in gt_full:
                gx -= x1
                gy -= y1
                if 0 <= gx < cw and 0 <= gy < ch:
                    gt_pts.append((gx, gy))

            if len(gt_pts) != 13:
                continue

            gt_score, gt_stage = compute_cvms_score(gt_pts)
            pred_score, pred_stage = compute_cvms_score(pred_pts)
            is_correct = int(gt_stage == pred_stage)

            correct += is_correct
            total += 1

            row = {
                "filename": fname,
                "gt_score": gt_score,
                "pred_score": pred_score,
                "gt_stage": gt_stage,
                "pred_stage": pred_stage,
                "stage_correct": is_correct,
            }
            for j, (x, y) in enumerate(gt_pts):
                row[f"gt_x{j}"], row[f"gt_y{j}"] = x, y
            for j, (x, y) in enumerate(pred_pts):
                row[f"pred_x{j}"], row[f"pred_y{j}"] = x, y

            results.append(row)

        progress.empty()

        if total == 0:
            st.warning("No valid samples were evaluated. Check the image directory and CSV contents.")
        else:
            out_df = pd.DataFrame(results)
            accuracy = correct / total
            mae = np.mean(np.abs(out_df["gt_score"] - out_df["pred_score"]))

            st.markdown(
                f"""
                <div style="display:flex; gap:20px; margin-bottom: 20px;">
                    <div class="cvms-card" style="flex:1;">
                        <div class="cvms-card-label">Clinical Stage Accuracy</div>
                        <div class="cvms-card-value" style="color:var(--good);">{accuracy * 100:.2f}%</div>
                    </div>
                    <div class="cvms-card" style="flex:1;">
                        <div class="cvms-card-label">Score MAE</div>
                        <div class="cvms-card-value">{mae:.4f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            results_table_html(out_df)

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download full results CSV",
                data=csv_bytes,
                file_name="cvms_results_final_complete.csv",
                mime="text/csv",
            )
