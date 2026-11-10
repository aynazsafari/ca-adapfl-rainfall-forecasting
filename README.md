<div align="center">

# 🌧️ CA-AdapFL: Climate-Aware Adaptive Federated Learning 
### *Monthly Rainfall Forecasting across Heterogeneous Hydroclimatic Regions in India*

<p align="center">
  <img src="https://img.shields.io/badge/Release-v1.2.0-f4a2b8?style=for-the-badge&logo=github&logoColor=white" alt="Release" />
  <img src="https://img.shields.io/badge/Python-3.12.14-c8b6ff?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.14.1-ffc8dd?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-bde0fe?style=for-the-badge" alt="License" />
  <a href="https://doi.org/ZENODO_DOI_HERE"><img src="https://img.shields.io/badge/DOI-ZENODO__DOI__HERE-ffafcc?style=for-the-badge" alt="DOI" /></a>
</p>

<p align="center">
  ✨ <i>Official code, processed datasets, model checkpoints, and reproducibility artifacts published in <b>Scientific Reports</b>.</i> ✨
</p>

---
</div>

## 🌷 Overview

Monthly rainfall forecasts for India must serve regions with contrasting rainfall regimes whose records are held by different providers. This repository provides a controlled, source-aware evaluation framework investigating two core questions:
1. **Information Sharing:** Do one-month-ahead regional rainfall forecasts improve when regions share what their models learn (via Federated Learning) without pooling raw records?
2. **Climate Conditioning:** Does incorporating large-scale climate indices (ENSO, Indian Ocean Dipole, Atlantic Multidecadal Variability) and meteorological reanalysis add forecast skill beyond the training-period monthly climatology (the seasonal cycle)?

We evaluate **CA-AdapFL** (an Attention-BiLSTM Climate Encoder with adaptive proximal regularization and optional climate-aware aggregation) alongside standard parameter-aggregation baselines (`FedAvg`, `FedProx`), personalized models (`Ditto`, climate-similarity graph controls), standalone regional (`Local`) models, and pooled (`Centralized`) neural and classical tree/linear controls across **five fixed hydroclimatic domains in India (1981–2024)**.

---

## 📂 Repository Structure

The repository is organized to ensure complete transparency, modularity, and reproducibility:

```text
CAAdapFL/
├── 📁 benchmarks/           # Performance benchmarking scripts & evaluation metrics
├── 📁 docs/                 # Documentation, supplementary guides & protocol notes
├── 📁 metadata/             # Source manifests, coordinate masks, hashes & area weights
├── 📁 notebooks/            # Jupyter notebooks for exploratory analysis & visualization
├── 📁 publication/          # Supplementary materials, tables & publication assets
├── 📁 scripts/              # Executable Python scripts for training, replay & verification
├── 📁 tests/                # Unit tests & independent numerical checks
├── 📄 CITATION.cff          # Citation metadata template
├── 📄 LICENSE               # Apache-2.0 License terms
├── 📄 NOTICE                # Copyright & institutional notices
├── 📄 README.md             # English documentation (this file)
├── 📄 README_FA.md          # Persian documentation (راهنمای فارسی)
├── 📄 benchmark.py          # Main automated benchmarking entrypoint
├── 📄 requirements.txt      # Core Python environment dependencies
├── 📄 requirements-imd-source.txt       # Dependencies for IMD rainfall integration
└── 📄 requirements-reference-windows.txt # Windows reference environment specifications
```

---

## 💻 Environment Setup & Quickstart

Experiments were developed and tested in a deterministic CPU simulation environment (`Python 3.12.14`, `PyTorch 2.14.1+cpu`).

### 1️⃣ Installation
Clone the repository and install the required dependencies based on your system setup:
```bash
git clone https://github.com/aynazsafari/CAAdapFL.git
cd CAAdapFL

# Install core dependencies
pip install -r requirements.txt

# (Optional) If running IMD gridded dataset integration
pip install -r requirements-imd-source.txt
```

### 2️⃣ Running Verification Scripts
You can independently recompute all reported metrics, paired circular block-bootstrap confidence intervals, and communication payloads directly from the archived predictions without retraining:
```bash
python scripts/verify_primary_results.py
python scripts/verify_capacity_output_study.py
python scripts/verify_information_sharing_study.py
```

### 3️⃣ Executing Benchmarks
To run the automated performance evaluation pipeline:
```bash
python benchmark.py --mode evaluate --seeds 7 42 123 999 2024
```

---

## 📊 Summary of Key Findings (2017–2024 Evaluation)

| Model Identity | Scope | Macro $R^2$ (=NSE) | Macro RMSE (mm) | MAE (mm) | KGE | MSESS vs. Climatology |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🌟 **TRAIN Monthly Climatology** | Reference | **0.8152 ± 0.0000** | **45.64 ± 0.00** | **27.26** | 0.7470 | `0.0000` |
| 🧠 **CA-AdapFL (Primary FULL48)** | Global | 0.7553 ± 0.0079 | 52.87 ± 1.25 | 36.26 | 0.7071 | `-0.36` |
| 🌐 **FedAvg (Primary FULL48)** | Global | 0.7581 ± 0.0060 | 51.87 ± 0.86 | 35.07 | 0.7141 | `-0.30` |
| 🔗 **FedProx (Primary FULL48)** | Global | 0.7561 ± 0.0080 | 52.81 ± 1.25 | 36.27 | 0.7090 | `-0.35` |
| ✨ **CA-AdapFL (Nonnegative FULL12)** | Global | 0.7755 ± 0.0114 | 50.23 ± 1.48 | 32.89 | 0.7390 | `-0.22` |
| 🌷 **CA-AdapFL (Personal Checkpoint)** | Personal | 0.8076 ± 0.0024 | 47.43 ± 0.60 | 30.22 | 0.7741 | `-0.10` |
| 🏛️ **Centralized Neural (Pooled)** | Pooled | 0.8050 ± 0.0080 | 47.22 ± 0.66 | 30.17 | 0.7853 | `-0.10` |
| 🌲 **ExtraTrees (Raw Pooled Latest)** | Classical | 0.8119 ± 0.0018 | 45.60 ± 0.18 | 28.01 | **0.8029** | `-0.02` |

> 💡 **Key Takeaway:** With shared input scaling under a fixed local-update budget, tick-wise federated model exchange (`FedAvg K1`) reduces macro MSE by **20.76%** relative to isolated local training, while synchronizing every 5 or 20 ticks cuts required numeric communication payload by **79.33%** or **94.20%** at the cost of higher error. However, because models predict total monthly rainfall, no tested neural configuration exceeds the training-period monthly climatology at this one-month lead.

---

## 🔐 Data Availability & Source Attribution Policy

* **Included Deposit:** Processed regional monthly series, predictor channels (excluding IMD), cohort calendars, manifests, checksums, raw model predictions, bootstrap draws, and saved weights are archived in this repository and permanently deposited in Zenodo (`v1.2.0`).
* **Original Gridded Sources:** Raw gridded archives remain available from official providers (CHIRPS v2.0, ERA5/ERA5-Land, GLDAS Noah v2.1, NASA POWER v10, and NOAA ERSSTv6).
* ⚠️ **IMD RF25 Redistribution Note:** Because no redistribution permission was obtained for the India Meteorological Department (`IMD RF25`, $0.25^\circ$) gridded product, IMD-derived values are excluded from this public deposit. Users who obtain the daily netCDF/grd files directly from **IMD Pune** can reconstruct the exact matched sensitivity cohort by placing files in `data_raw/imd_rf25/` and running scripts provided in `scripts/`.

---

## 💌 Authors & Citation

* **Aynaz Safari Shateri** — University of Tabriz ([ORCID: 0009-0009-3494-2753](https://orcid.org/0009-0009-3494-2753))
* **Mobin Jamshidi Asl** — University of Tabriz ([ORCID: 0009-0005-5494-4247](https://orcid.org/0009-0005-5494-4247))
* **Amirreza Shirini** — Sahand University of Technology ([ORCID: 0009-0005-3533-0047](https://orcid.org/0009-0005-3533-0047))
* **Mohammad Taghi Sattari** *(Corresponding Author)* — University of Tabriz / Khazar University / Ankara University ([ORCID: 0000-0002-5139-2118](https://orcid.org/0000-0002-5139-2118))
* **Halit Apaydin** — Ankara University ([ORCID: 0000-0002-9875-7321](https://orcid.org/0000-0002-9875-7321))


<div align="center">
  <br/>
  <p>🌸 <i>Released under the <b>Apache-2.0 License</b></i> 🌸</p>
</div>
