"""Shared model, math, and UI-helper logic for the SAGA-AI app.

Used by both entry points:
- app.py       -- full-featured local/Mac app (sidebar, batch evaluation, LAN share card)
- app_cloud.py -- simplified public deployment (single image only, weights via Hugging Face)
"""

import os
import json
import math
import glob
import base64

import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from torchvision import models, transforms

from model import SpineHRNet

# ========= CONFIG =========
IMG_SIZE = (256, 256)
NET2_SIZE = (512, 512)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Resolved next to this script (not cwd) so it works both in dev (run from the
# app folder) and in the packaged .app (cwd is the external weights/data folder).
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maids-logo.png")

net1_tf = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
net2_tf = transforms.Compose([transforms.Resize(NET2_SIZE), transforms.ToTensor()])

# Clinical dark theme: teal for UI/data, amber for radiograph "light-box" framing,
# and a teal->amber gradient across the 4 CVM stages (cool/early -> warm/mature).
STAGE_COLORS = ["#5fb8d9", "#4fc9a0", "#d9b34f", "#d97a3f"]

STAGE_DESC = {
    1: "Pre-pubertal growth spurt",
    2: "Onset of pubertal growth spurt",
    3: "Peak of pubertal growth spurt",
    4: "Growth spurt decelerating / post-peak",
}


# ========= MODEL LOADING (cached) =========
@st.cache_resource(show_spinner="Loading Net1 (crop detector)...")
def load_net1(path):
    model = models.resnet34(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 4),
        torch.nn.Sigmoid(),
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading Net2 (landmark detector)...")
def load_net2(path):
    model = SpineHRNet(num_landmarks=13, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ========= DARK DECODING =========
def get_dark_coords(heatmaps):
    N, H, W = heatmaps.shape
    coords = np.zeros((N, 2), dtype=np.float32)

    for i in range(N):
        hmap = heatmaps[i]
        idx = np.argmax(hmap)
        y, x = np.unravel_index(idx, (H, W))

        if 1 < x < W - 2 and 1 < y < H - 2:
            dx = 0.5 * (hmap[y, x + 1] - hmap[y, x - 1])
            dy = 0.5 * (hmap[y + 1, x] - hmap[y - 1, x])

            dxx = hmap[y, x + 1] - 2 * hmap[y, x] + hmap[y, x - 1]
            dyy = hmap[y + 1, x] - 2 * hmap[y, x] + hmap[y - 1, x]

            if abs(dxx) > 1e-4:
                x += -dx / dxx
            if abs(dyy) > 1e-4:
                y += -dy / dyy

        coords[i] = [x, y]

    return coords


# ========= CLINICAL MATH & CVMS LOGIC =========
def get_dist_point_to_line(pt, line_pt1, line_pt2):
    x0, y0 = pt
    x1, y1 = line_pt1
    x2, y2 = line_pt2
    numerator = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
    denominator = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if denominator < 1e-6:
        denominator = 1e-6
    return numerator / denominator


def get_angle_at_vertex(p1, vertex, p2):
    v1 = np.array([p1[0] - vertex[0], p1[1] - vertex[1]])
    v2 = np.array([p2[0] - vertex[0], p2[1] - vertex[1]])

    dot_prod = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    if mag_v1 < 1e-6:
        mag_v1 = 1e-6
    if mag_v2 < 1e-6:
        mag_v2 = 1e-6

    cos_theta = np.clip(dot_prod / (mag_v1 * mag_v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def calculate_cvms_score(landmarks, clamp_ratios=True):
    h4 = get_dist_point_to_line(landmarks[11], landmarks[10], landmarks[9])
    w4 = get_dist_point_to_line(landmarks[12], landmarks[8], landmarks[10])

    ah3 = get_dist_point_to_line(landmarks[3], landmarks[6], landmarks[5])
    ph3 = get_dist_point_to_line(landmarks[4], landmarks[6], landmarks[5])

    angle_2 = get_angle_at_vertex(landmarks[2], landmarks[1], landmarks[0])

    ratio_1 = h4 / (w4 + 1e-6)
    ratio_2 = ah3 / (ph3 + 1e-6)

    if clamp_ratios:
        ratio_1 = np.clip(ratio_1, 0.1, 4.0)
        ratio_2 = np.clip(ratio_2, 0.1, 4.0)

    cvms = -4.13 + (3.57 * ratio_1) + (4.07 * ratio_2) + (0.03 * angle_2)
    return cvms


def get_stage(cvms):
    T1, T2, T3 = 1.7404, 2.623, 3.5199
    if cvms < T1:
        return 1
    elif T1 <= cvms < T2:
        return 2
    elif T2 <= cvms < T3:
        return 3
    else:
        return 4


def compute_cvms_score(points):
    score = calculate_cvms_score(points)
    stage = get_stage(score)
    return score, stage


# ========= CSV / GROUND TRUTH PARSING (batch mode) =========
def parse_test_csv(csv_path_or_buffer):
    df = pd.read_csv(csv_path_or_buffer)
    rows = []
    for _, row in df.iterrows():
        try:
            s = json.loads(row["region_shape_attributes"])
            rows.append([row["filename"], s["cx"], s["cy"], row["region_id"]])
        except Exception:
            continue
    return pd.DataFrame(rows, columns=["filename", "cx", "cy", "region_id"])


def get_gt_points(df, fname):
    group = df[df["filename"] == fname].sort_values("region_id")
    return [(int(r["cx"]), int(r["cy"])) for _, r in group.iterrows()]


# ========= CORE INFERENCE (single image) =========
def run_inference(net1, net2, img_bgr):
    h, w = img_bgr.shape[:2]

    # ----- NET1 (crop) -----
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    inp = net1_tf(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = net1(inp)[0].cpu().numpy()

    x1 = max(0, int(pred[0] * w))
    y1 = max(0, int(pred[1] * h))
    x2 = min(w, int(pred[2] * w))
    y2 = min(h, int(pred[3] * h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]

    # ----- NET2 (landmarks) -----
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil2 = Image.fromarray(crop_rgb)
    inp2 = net2_tf(pil2).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = net2(inp2)[0].cpu().numpy()

    hm_coords = get_dark_coords(out)

    pred_pts = []
    for (x, y) in hm_coords:
        px = int(x * (cw / 128))
        py = int(y * (ch / 128))
        pred_pts.append((px, py))

    return {
        "bbox": (x1, y1, x2, y2),
        "crop": crop,
        "pred_pts": pred_pts,
    }


def draw_landmarks(crop_bgr, pts):
    """Amber markers with teal index labels, matching the SAGA-AI light-box theme."""
    vis = crop_bgr.copy()
    amber_bgr = (86, 162, 217)
    teal_bgr = (209, 199, 79)
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (x, y), 4, amber_bgr, -1)
        cv2.circle(vis, (x, y), 4, (20, 20, 20), 1)
        cv2.putText(vis, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, teal_bgr, 1)
    return vis


# ========= WEIGHTS DISCOVERY =========
def find_weight_files():
    """Scan the current working directory (and its immediate subfolders) for .pth files."""
    candidates = sorted(glob.glob("*.pth")) + sorted(glob.glob("*/*.pth"))
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def guess_default(files, keywords):
    for kw in keywords:
        for f in files:
            if kw in os.path.basename(f).lower():
                return f
    return files[0] if files else None


HF_WEIGHT_FILENAMES = ["best_bbox_net1.pth", "best_net2.pth"]


def ensure_weights_from_hf():
    """On a cloud deploy (e.g. Streamlit Community Cloud) the confidential .pth
    files are never committed to git — instead they're downloaded here, once,
    from a private Hugging Face model repo using credentials from st.secrets.
    Local/desktop use (weights already on disk) is unaffected: this no-ops
    whenever weight files are already found, or no HF secrets are configured."""
    if find_weight_files():
        return

    try:
        repo_id = st.secrets.get("HF_REPO_ID")
        hf_token = st.secrets.get("HF_TOKEN")
    except Exception:
        return  # no secrets.toml at all (e.g. plain local run) - nothing to do

    if not repo_id or not hf_token:
        return

    try:
        from huggingface_hub import hf_hub_download

        with st.spinner("Downloading model weights..."):
            for fname in HF_WEIGHT_FILENAMES:
                hf_hub_download(repo_id=repo_id, filename=fname, token=hf_token, local_dir=".")
    except Exception as e:
        st.error(f"Could not download weights from Hugging Face: {e}")


# ========= IMAGE / UI HELPERS =========
def bgr_to_b64(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


@st.cache_data
def logo_b64():
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def inject_theme():
    st.markdown(
        """
        <style>
        :root {
            --bg: #14171c;
            --surface: #1b1f26;
            --surface-2: #232932;
            --border: #313947;
            --text: #eef1f5;
            --text-muted: #93a0ac;
            --teal: #4fc7d1;
            --amber: #d9a256;
            --good: #4fc9a0;
            --bad: #d9705f;
            --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        }
        .stApp { background: var(--bg); }
        section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }

        .stButton > button, .stDownloadButton > button {
            background: var(--teal) !important;
            color: #0d1013 !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover { opacity: 0.9; }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(79,199,209,0.06) !important;
            border: 1.5px dashed #3a6b70 !important;
            border-radius: 12px !important;
        }

        .cvms-status-pill {
            display: inline-flex; align-items: center; gap: 7px;
            background: var(--surface); border: 1px solid var(--border);
            border-radius: 999px; padding: 6px 14px; font-size: 12px; color: var(--text-muted);
        }
        .cvms-status-dot {
            width: 7px; height: 7px; border-radius: 50%; background: var(--teal);
            box-shadow: 0 0 6px rgba(79,199,209,0.5);
        }

        .cvms-card {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: 12px; padding: 20px 22px; height: 100%; box-sizing: border-box;
        }
        .cvms-card-label {
            font-size: 11px; font-weight: 600; letter-spacing: 0.6px;
            text-transform: uppercase; color: var(--text-muted);
        }
        .cvms-card-value {
            font-size: 34px; font-weight: 600; font-family: var(--mono);
            margin-top: 6px; color: var(--text);
        }
        .cvms-stage-track { display: flex; gap: 5px; margin-top: 12px; }
        .cvms-stage-seg { flex: 1; height: 7px; border-radius: 4px; }

        .cvms-lightbox {
            border: 1.5px solid var(--amber); border-radius: 12px; padding: 10px;
            background: #0c0f12; box-shadow: 0 0 24px rgba(217,162,86,0.18) inset;
        }
        .cvms-caption { text-align: center; font-size: 12px; color: var(--text-muted); margin-top: 8px; }

        .cvms-table-wrap { max-height: 460px; overflow-y: auto; border: 1px solid var(--border); border-radius: 12px; }
        .cvms-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
        .cvms-table th {
            position: sticky; top: 0; background: var(--surface-2); text-align: left;
            font-size: 11px; font-weight: 600; letter-spacing: 0.4px; text-transform: uppercase;
            color: var(--text-muted); padding: 10px 14px; border-bottom: 1px solid var(--border);
        }
        .cvms-table td { padding: 8px 14px; border-top: 1px solid #262c35; }
        .cvms-mono { font-family: var(--mono); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def app_header(models_ready):
    dot_color = "var(--teal)" if models_ready else "var(--bad)"
    status_text = f"Models loaded &middot; {DEVICE.type.upper()}" if models_ready else "Weights not found"
    b64 = logo_b64()
    logo_html = (
        f"""<div style="width:40px; height:40px; border-radius:8px; background:white;
                        display:flex; align-items:center; justify-content:center;
                        overflow:hidden; flex-shrink:0;">
                <img src="data:image/png;base64,{b64}" style="width:100%; height:100%; object-fit:contain;" />
            </div>"""
        if b64 else ""
    )
    st.markdown(
        f"""
        <div style="display:flex; align-items:flex-start; justify-content:space-between; margin-bottom: 22px;">
            <div style="display:flex; align-items:center; gap:12px;">
                {logo_html}
                <div>
                    <div style="font-size:26px; font-weight:700; letter-spacing:-0.2px; color:var(--text);">SAGA-AI</div>
                    <div style="font-size:13px; color:var(--text-muted); margin-top:4px;">Skeletal Age and Growth Assessment</div>
                </div>
            </div>
            <div class="cvms-status-pill">
                <div class="cvms-status-dot" style="background:{dot_color}; box-shadow: 0 0 6px {dot_color};"></div>
                <span>{status_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_cards_single(score, stage):
    color = STAGE_COLORS[stage - 1]
    segs = "".join(
        f'<div class="cvms-stage-seg" style="background:{STAGE_COLORS[i] if i <= stage - 1 else "var(--border)"};"></div>'
        for i in range(4)
    )
    st.markdown(
        f"""
        <div style="display:flex; gap:20px;">
            <div class="cvms-card" style="flex:1;">
                <div class="cvms-card-label">CVMS Score</div>
                <div class="cvms-card-value">{score:.4f}</div>
            </div>
            <div class="cvms-card" style="flex:1.4;">
                <div style="display:flex; align-items:center; justify-content:space-between;">
                    <div class="cvms-card-label">Predicted Stage</div>
                    <div style="font-size:15px; font-weight:700; font-family:var(--mono); color:{color};">CS{stage}</div>
                </div>
                <div class="cvms-stage-track">{segs}</div>
                <div style="font-size:12.5px; color:var(--text-muted); margin-top:10px;">{STAGE_DESC[stage]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def lightbox_image(b64, caption, compact=False):
    frame_style = 'max-width:360px; margin:0 auto;' if compact else ''
    image_style = 'max-height:300px; object-fit:contain;' if compact else ''
    st.markdown(
        f"""
        <div class="cvms-lightbox" style="{frame_style}">
            <img src="data:image/png;base64,{b64}" style="width:100%; display:block; border-radius:6px; {image_style}" />
        </div>
        <div class="cvms-caption">{caption}</div>
        """,
        unsafe_allow_html=True,
    )


def results_table_html(df, max_rows=500):
    rows_html = []
    for _, r in df.head(max_rows).iterrows():
        correct = bool(r["stage_correct"])
        icon = (
            '<span style="color:var(--good); font-weight:700;">&#10003;</span>' if correct
            else '<span style="color:var(--bad); font-weight:700;">&#10007;</span>'
        )
        rows_html.append(
            f"""<tr>
                <td class="cvms-mono" style="color:var(--text-muted);">{r['filename']}</td>
                <td class="cvms-mono">{r['gt_score']:.4f}</td>
                <td class="cvms-mono">{r['pred_score']:.4f}</td>
                <td>CS{r['gt_stage']}</td>
                <td>CS{r['pred_stage']}</td>
                <td>{icon}</td>
            </tr>"""
        )
    st.markdown(
        f"""
        <div class="cvms-table-wrap">
            <table class="cvms-table">
                <thead><tr><th>Filename</th><th>GT Score</th><th>Pred Score</th><th>GT Stage</th><th>Pred Stage</th><th>Correct</th></tr></thead>
                <tbody>{''.join(rows_html)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
