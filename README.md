# Greenhouse-AI

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-hyperparameter%20search-2C5BB4)
![License](https://img.shields.io/badge/license-MIT-green)

Machine-learning analysis of a real greenhouse microclimate, built for my diploma thesis.
The goal is **virtual sensing**: predicting the temperature at positions inside the greenhouse
where a physical sensor has been removed, using the remaining sensors plus environmental data.

The data comes from the ESP32 sensor network in
[SMART---GREENHOUSE](https://github.com/GioriaTh/SMART---GREENHOUSE).

## Highlights

- **Dataset**: ~19,200 one-minute readings (28 Apr – 11 May 2025) from 12 NTC thermistors
  placed on a 3-D grid, inside/outside temperature & humidity, inside/outside light, door state and wind speed.
- **Feature analysis**: mutual information to find which sensors best explain each thermistor;
  PCA per time-of-day period (dawn, morning, afternoon, evening, night).
- **LSTM model**: predicts the 12 thermistor temperatures from a reduced set of input sensors,
  with sensor 3-D coordinates as extra features. Optuna searches sequence length, LSTM units,
  dropout and batch size for every combination of input sensors.
- **Visualisation**: 3-D heatmap of the greenhouse with predicted vs. real temperatures.

## Repository structure

```
Greenhouse-AI/
├── Data Analysis/
│   ├── Mutual Info.py        # mutual information: best predictors for each NTC
│   └── Pca_timezone.py       # PCA per time-of-day period + plots
├── LSTM/
│   └── trial.py              # LSTM + Optuna search, final training, 3-D heatmaps
├── sensor_readings3rd.csv    # dataset (1-minute resolution)
└── Διπλωματική_Παρουσίαση.pptx  # thesis presentation (Greek)
```

## Dataset

| Column | Meaning |
|---|---|
| `timestamp` | Reading time (`M/D/YYYY H:MM`) |
| `temp1` … `temp12` | NTC thermistors (°C) on a 5 m × 3 m × 2 m grid, 3 heights per post |
| `chtIn_temp`, `chtIn_humidity` | CHT8305 inside the greenhouse (°C, %RH) |
| `chtOut_temp`, `chtOut_humidity` | CHT8305 outside the greenhouse (°C, %RH) |
| `ir_in`, `ir_out` | TSL2591 light sensors, inside / outside (raw) |
| `door` | Magnetic door sensor, digital reading (0 / 1) |
| `wind` | Anemometer wind speed (m/s) |

Rows where `ir_in` is `65535` (sensor saturation) or `1` (sensor error) are filtered out by the scripts.

## Getting started

```bash
git clone https://github.com/GioriaTh/Greenhouse-AI.git
cd Greenhouse-AI
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The scripts read `sensor_readings3rd.csv` from the repo root. To use another file, set
`GREENHOUSE_CSV=/path/to/file.csv`.

```bash
python "Data Analysis/Mutual Info.py"
python "Data Analysis/Pca_timezone.py"
python LSTM/trial.py        # long run: tries every input-sensor combination with Optuna
```

## Tech stack

Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · Optuna · Matplotlib · Seaborn

## Related projects

- [SMART---GREENHOUSE](https://github.com/GioriaTh/SMART---GREENHOUSE) — ESP32 firmware and database that collect this data

## Author

**GioriaTh** — Computer Engineer · [github.com/GioriaTh](https://github.com/GioriaTh)

## License

Released under the [MIT License](LICENSE).
