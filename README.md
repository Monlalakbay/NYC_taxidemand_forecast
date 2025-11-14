# 🚕 NYC Taxi Demand Forecasting (CRISP-DM Case Study)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![MachineLearning](https://img.shields.io/badge/Machine%20Learning-RF%2FLightGBM%2FLSTM-green)
![DeepLearning](https://img.shields.io/badge/Deep%20Learning-Conv1D%20%2B%20LSTM-purple)
![GeoPandas](https://img.shields.io/badge/GeoSpatial-GeoPandas%2FShapely-lightgrey)

This repository contains an end-to-end forecasting pipeline for **hourly taxi demand** in New York City community districts.  
The project combines:

- Classical regression (Linear, Ridge, Lasso)
- Tree-based ensembles (Random Forest, LightGBM)
- A deep **Conv1D + LSTM** model
- A **Hybrid Ensemble** that blends all three families

The work follows the **CRISP-DM** methodology from business understanding to deployment-style evaluation and mapping.

---

## 1.  📁 Project Structure

```text
NYC_taxidemand_forecast/
├─ data/
│  ├─ raw/                  # Original source data 
│  ├─ cleaned/              # Cleaned CSVs 
│  ├─ processed/            # Datasets with additional features, train/test splits 
│  ├─ runs/                 # Model artifacts and predictions 
│  └─ evaluation/           # Summary metrics, Map outputs
│
├─ src/
│  ├─ data/
│  │  ├─ clean_data.py          # Basic cleaning and geospatial filtering
│  │  ├─ preprocess.py          # Spatial join + hourly aggregation + features
│  │  ├─ split_data.py          # Chronological train/test split + scaling
│  │  ├─ data_prep.py           # Sequence preparation for Conv1D + LSTM
│  │  └─ fetch_data_api.py      # (Optional) Download / load raw data
│  │
│  ├─ models/
│  │  ├─ modeling_districts.py      # Linear models + RF + LightGBM per district
│  │  ├─ modeling_conv_lstm_train.py# Conv1D + LSTM per district
│  │  └─ blend_hybrid.py            # Hybrid blending (RF + LGBM + ConvLSTM)
│  │
│  ├─ evaluation/
│  │  └─ evaluate_test_models.py    # Unified evaluation on final test set
│  │
│  ├─ vizualize/
│  │  └─ next_hour_map.py           # Folium map of next-hour forecast + RMSE
│  │
│  ├─ utils/
│  │  └─ pipeline_logger.py         # Centralized logging + pipeline step helper
│  │
│  └─ run_pipeline.py               # Model pipeline
│
├─ notebooks/
│  └─ figures                     # Result plots
│  └─ 00_analysis.ipynb           # Analysis of model results
│  └─ 01_exploration.ipynb        # Exploratory analysis 
│  └─ 02_modeling_baseline.ipynb  # Analysis of baseline model
│
├─ reports/                       # (Optional) Technical reports for Stakeholders
│
├─ README.md
├─ requirements.txt
└─ .gitignore
```

---
## 2. 💻 Setup
### 2.1. Setup Create and activate a virtual environment (recommended)
```shell
cd NYC_taxidemand_forecast

python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate
```

### 2.2. Install dependencies
```shell
pip install --upgrade pip
pip install -r requirements.txt
```
You will also need system dependencies for GeoPandas (GDAL/GEOS/PROJ).
On Windows this is usually handled automatically when installing via pip or conda.


---
## 3. 🚖 Data
The project specifically uses NYC district boundaries and pickup data from 2014. 

All geospatial joins use NYC community district polygons; these are not included in the repository for size/licensing reasons 
and must be added manually under data/raw/district_nyc_data/.

```shell
data/raw/district_nyc_data
data/raw/pickup_data
```

For the pickup data, the project uses publicly available data from:

**Taxi Pickup Records**  
Kaggle. (2019). *Uber Dataset from April to September 2014.*  
Dataset available at:  
https://www.kaggle.com/datasets/amirmotefaker/uber-dataset-from-april-to-september-2014

Feel free to fork, adapt, or extend the pipeline for other cities, additional features (e.g. weather, events), or alternative model architectures.

---
## 4. ⚙️ Run Pipeline
The entire modeling workflow can be launched with **one single command**:

```shell
python -m run_pipeline
```

This command executes:

🧹 Data preprocessing (src/data/preprocess.py)

🔪 Train/test splitting + scaling (src/data/split_data.py)

⏱️ ConvLSTM sequence preparation (src/data/data_prep.py)

🌳 Tree-based model training (src/models/modeling_districts.py)

🧠 Conv1D + LSTM model training (src/models/modeling_conv_lstm_train.py)

⚖️ Hybrid blending of all predictions (src/models/blend_hybrid.py)

❌ Unified evaluation on the test set (src/evaluation/evaluate_test_models.py)

🌍 Spatial visualization (Folium map)

📊 All artifacts (predictions, logs, summaries, plots) are stored under:
```shell
data/runs/<timestamp>/
data/evaluation/
logs/
pipeline_log.csv
```
