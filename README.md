# quishguard-c3

QR code phishing (quishing) detection using a Vision Transformer and URL behavioural analysis.
Component C3 of SLIIT project J26-IT-424 (Umaharan N, IT23234802).

## How it works

1. **Safe decode (Module B):** finds, straightens and decodes the QR code, and labels the payload (URL, Wi-Fi, payment, text…). The link is never opened.
2. **ViT anomaly detector (C1):** trained only on normal QR codes; tampered codes (stickers, logos, warps) rebuild badly and get a high score.
3. **URL model (C2):** lexical features + XGBoost, with SHAP reasons.
4. **Fusion + severity (D):** `score = 0.60·URL + 0.40·visual` (weight chosen on val) → Safe / Suspicious / Phishing / Critical.
5. **Explainable alert and API (E–F):** JSON for the group's Final Aggregator.

## Setup on the laptop (Windows)

```powershell
cd D:\uma\quishguard-c3
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
python -m pytest -q
```

## Phase 1: prepare the data

```powershell
# URL table only (fast, works even if the images are not fully unzipped)
python -m quishguard.data.prepare --raw-dir D:\uma --skip-images
```

For the image subset, run `notebooks/00_colab_setup.ipynb` in Colab. It unzips all images and drops broken ones.

Outputs go to `data/processed/` (not committed):

| File | What |
|---|---|
| `urls_clean.csv.gz` | every usable URL with label, host, domain and split |
| `image_subset.csv` | balanced images that decode correctly |
| `prepare_report.json` | counts for the report |

## Dataset notes (CIC-Trap4Phish 2025)

- Every image is named `benign_XXXXXX.png`, even in the malicious folder. The label comes from the folder.
- Images have no white quiet zone; the decoder adds one.
- 114 px and 126 px images are broken (one finder pattern only): about 7% of malicious and 2% of benign. They are removed.
- 91% of benign URLs have no `http(s)://` but only 38% of malicious ones, so URLs are normalised (scheme and `www.` removed) before features.
- Splits are made by registered domain (70/10/20), so no website is in both training and testing.

## Scan app: phone scan page + alert portal

One server on the laptop does both jobs. The phone opens the scan page in its browser (no app to install),
takes a photo of a QR code and gets the tier, score, reasons and heatmap. Every Phishing or Critical scan
appears on the portal as a new alert, with sound.

**1. Get the models (once).** In Colab, run the last cell of `06_fusion_and_real_test.ipynb`. Then download
these files from `MyDrive/quishguard/models/` into `D:\uma\quishguard-c3\models\`:
`url_model_v2.joblib`, `visual_patchcore.pt`, `visual_patchcore_calibrator.joblib`, `versions.txt`.

**2. Install (once).**

```powershell
cd D:\uma\quishguard-c3
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-app.txt
pip install -r models\versions.txt      # same library versions as Colab, so the models load
pip install -e .
```

**3. Run.**

```powershell
python -m quishguard.app --models-dir models
```

It prints two addresses, for example `http://192.168.1.23:8000/` (phone scan page) and
`http://192.168.1.23:8000/portal` (alert portal). Open the portal on the laptop and the scan page on the phone.

- The phone and the laptop must be on the **same Wi-Fi**. On university Wi-Fi devices often cannot see each other:
  turn on the phone's hotspot and connect the laptop to it.
- When Windows asks, allow Python on **private networks**.
- Optional: `--key MYKEY` makes the pages need `?key=MYKEY`, so nobody else on the Wi-Fi can use them.
- Optional: `--webhook URL` sends every Phishing/Critical alert as JSON to the group's Final Aggregator.
- Scans are stored in `data/app/` (scans.db, photos, heatmaps), which is never committed to git.
- The decoded link is shown defanged (`hxxps://example[.]com`) and is never opened.

## Layout

```
src/quishguard/   data/ decode/ url_model/ vit/ tamper/ fusion/ realtest/ app/  pipeline.py
notebooks/        Colab notebooks
tests/            pytest
data/, models/    local only, ignored by git
```
