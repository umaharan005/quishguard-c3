# quishguard-c3

QR code phishing (quishing) detection using a Vision Transformer and URL behavioural analysis.
Component C3 of SLIIT project J26-IT-424 (Umaharan N, IT23234802).

## How it works

1. **Safe decode (Module B):** finds, straightens and decodes the QR code, and labels the payload (URL, Wi-Fi, payment, text…). The link is never opened.
2. **ViT anomaly detector (C1):** trained only on normal QR codes; tampered codes (stickers, logos, warps) rebuild badly and get a high score.
3. **URL model (C2):** lexical features + XGBoost, with SHAP reasons.
4. **Fusion + severity (D):** `score = 0.65·URL + 0.35·ViT` → Safe / Suspicious / Phishing / Critical.
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

## Layout

```
src/quishguard/   data/  decode/  (url_model/ vit/ fusion/ explain/ api/ come next)
notebooks/        Colab notebooks
tests/            pytest
data/, models/    local only, ignored by git
```
