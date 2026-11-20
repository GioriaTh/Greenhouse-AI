import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

import os
from pathlib import Path

# Dataset path: defaults to the CSV in the repo root; override with GREENHOUSE_CSV=/path/to/file.csv
REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = Path(os.environ.get("GREENHOUSE_CSV", REPO_ROOT / "sensor_readings3rd.csv"))

# Load dataset
dataset = pd.read_csv(CSV_PATH)

# Parse timestamps and filter invalid IR data
dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], errors='coerce')
dataset['hour'] = dataset['timestamp'].dt.hour
dataset['minute'] = dataset['timestamp'].dt.minute

# Filter out invalid IR readings
fil_dat = dataset[(dataset['ir_in'] != 65535) & (dataset['ir_in'] != 1)].copy()

# Define NTC and environmental columns
temp_cols = [f'temp{i}' for i in range(1, 13)]
other_cols = [
    'ir_in', 'ir_out', 'wind', 'door',
    'chtOut_temp', 'chtOut_humidity',
    'chtIn_humidity', 'chtIn_temp',
    'hour', 'minute'
]

# Store results
mi_results = {}

for removed_ntc in temp_cols:
    remaining_ntcs = [t for t in temp_cols if t != removed_ntc]
    input_features = remaining_ntcs + other_cols

    X_remain = fil_dat[input_features].values
    y_target = fil_dat[removed_ntc].values

    # Compute mutual information between ALL features and the removed NTC
    mi = mutual_info_regression(X_remain, y_target, discrete_features=False)

    # MI series for all input features
    mi_series = pd.Series(mi, index=input_features).sort_values(ascending=False)
    top_feature = mi_series.idxmax()

    mi_results[removed_ntc] = {
        'most_informative_feature': top_feature,
        'mutual_info_score': mi_series.max(),
        'mi_ranking': mi_series
    }

    # Print top 3 features overall (NTCs + environmental)
    print(f"\n=== Removed NTC: {removed_ntc} ===")
    print("Top 3 most informative input features based on mutual information:")
    print(mi_series.head(6))

# Summary table
summary_df = pd.DataFrame({
    'Removed NTC': list(mi_results.keys()),
    'Most Informative Feature': [v['most_informative_feature'] for v in mi_results.values()],
    'Mutual Info Score': [v['mutual_info_score'] for v in mi_results.values()]
})

print("\n=== Summary: Most Informative Features for Each Removed NTC (Including Environmental Data) ===")
print(summary_df.to_string(index=False))
