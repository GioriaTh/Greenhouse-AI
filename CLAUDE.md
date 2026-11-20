# CLAUDE.md — Greenhouse-AI

## Purpose

Diploma-thesis repository: ML analysis of greenhouse sensor data and LSTM "virtual sensors" that
predict thermistor temperatures at positions without a physical sensor. Data comes from the
ESP32 network in the sibling repo `SMART---GREENHOUSE`.

## Structure

- `Data Analysis/Mutual Info.py` — mutual information per removed NTC
- `Data Analysis/Pca_timezone.py` — PCA per time-of-day period
- `LSTM/trial.py` — Optuna search over input-sensor combinations, final training, 3-D heatmaps
- `sensor_readings3rd.csv` — dataset, 1-minute resolution, 28 Apr – 11 May 2025
- `Διπλωματική_Παρουσίαση.pptx` — thesis presentation (Greek); do not edit

## Conventions

- Python 3.10+, dependencies in `requirements.txt` — keep it updated when imports change.
- Sensor names are fixed: `temp1`–`temp12`, `chtIn_*`, `chtOut_*`, `ir_in`, `ir_out`, `door`, `wind`.
  The 3-D coordinates of each NTC live in `ntc_coords` in `LSTM/trial.py` — keep them in one place.
- Always filter invalid light readings: `ir_in == 65535` (saturation) and `ir_in == 1` (error).
- Data is a time series: split chronologically (no shuffling across train/test).
- Do not commit generated plots, model files (`*.h5`, `*.keras`) or Optuna databases.

## Common tasks

- Load the dataset via `CSV_PATH` (repo-root CSV, overridable with `GREENHOUSE_CSV`) — never hard-code local paths.
- Move shared loading/filtering code into one module (e.g. `src/data.py`).
- Add result summaries (MSE per sensor combination) to the README.
- Keep the README "Dataset" table in sync with the CSV columns.

## Do not

- Do not modify or re-order `sensor_readings3rd.csv`; derive new files instead.
- Do not change the thesis presentation.
- Do not rewrite git history.
