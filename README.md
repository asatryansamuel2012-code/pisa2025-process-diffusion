# Replication package (anonymised for review)

Run the Colab notebooks in this order (Google Drive is used for intermediate files; seeds are fixed, SEED = 20251025):

1. `PISA2025_extract_v2.ipynb` – downloads the PISA 2025 public-use files from the OECD (webfs.oecd.org) and extracts per-country Parquet files (CPU).
2. `PISA2025_stage1_prepare.ipynb` – long-format item data, cleaning, NT10 flags (CPU).
3. `PISA2025_stage2_baselines.ipynb` – analysis sample (ARM all, 1,000 per other system) and baseline fits (GPU).
4. `PISA2025_stage3_models.ipynb` – M1, M3, M4 (GPU).
5. `PISA2025_stage4_proposed.ipynb` – M5, M6a (GPU).
6. `PISA2025_stage4b_fair.ipynb` – common joint target, M6b, M6, final comparison (Table 2, Fig. 3) (GPU).
7. `PISA2025_stage7_indicators.ipynb` – model-free indicators and Fay-BRR estimates (Tables 1 and 3, Fig. 4) (CPU).
8. `simulation/idtest.py` (five-dimensional person structure, 6,000 steps) and `simulation/idtest3.py` (reduced structure, 20,000 steps) – identifiability study (CPU; `python idtest3.py 20000`).
9. `make_figures.py` – Figures 1–4.

Software: Python 3, JAX, NumPyro 0.x, ArviZ 0.x, pandas, pyreadstat, statsmodels, matplotlib. PISA data are not redistributed; they are downloaded from the OECD by step 1.
