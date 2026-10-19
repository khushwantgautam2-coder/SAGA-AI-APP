# SAGA-AI Streamlit update

Both entry points now include **Batch Processing**: upload multiple JPG/PNG images, process them sequentially, and download an Excel workbook. No ground-truth CSV is needed for batch predictions. The local app retains its existing Batch Evaluation mode.

The workbook contains Results (one row for every upload, including failures), Landmarks (13 named points per successful image, in crop and full-image coordinates), and Summary (counts and coordinate/scoring definitions). Failed images have blank scores and stages. Image IDs distinguish duplicate filenames. Results remain available through Streamlit reruns and downloads; selecting different images clears the previous report.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app_cloud.py
```

Use `app.py` for the full local interface. Keep both model weights in the working directory or configure private Hugging Face access through Streamlit secrets.

## Update Streamlit Community Cloud

Replace `app.py` and `app_cloud.py`, add `batch_processing.py`, and update `requirements.txt` in your repository with the versions in this folder. Keep `model.py`, `cvms_core.py`, and `maids-logo.png` alongside them. The Cloud main file remains `app_cloud.py`.

Configure `HF_REPO_ID` and `HF_TOKEN` in Streamlit secrets if weights are downloaded from private Hugging Face storage. Never commit weight files or tokens. Reboot the app after updating the repository so the new dependency is installed.

The model architecture, image preprocessing, landmark decoding, score formula and thresholds are unchanged. This update does not measure model accuracy. Actual inference validation requires the confidential weights and representative cephalograms.
