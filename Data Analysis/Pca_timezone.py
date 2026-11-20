import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
from pathlib import Path

# Dataset path: defaults to the CSV in the repo root; override with GREENHOUSE_CSV=/path/to/file.csv
REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = Path(os.environ.get("GREENHOUSE_CSV", REPO_ROOT / "sensor_readings3rd.csv"))

# 1. Load dataset
df = pd.read_csv(CSV_PATH)

# 2. Extract time components and create timezone labels
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute

conditions = [
    (df['hour'] >= 5) & (df['hour'] <= 7),
    (df['hour'] >= 8) & (df['hour'] <= 11),
    (df['hour'] >= 12) & (df['hour'] <= 17),
    (df['hour'] >= 18) & (df['hour'] <= 20),
    (df['hour'] >= 21) | (df['hour'] <= 4)
]
choices = ['dawn', 'morning', 'afternoon', 'evening', 'night']
df['timezone'] = np.select(conditions, choices, default='unknown')

print("Timezone distribution:\n", df['timezone'].value_counts())

# 3. Define valid feature columns
temp_cols = [f'temp{i}' for i in range(1, 13)]
other_cols = ['hour', 'minute', 'ir_in', 'ir_out', 'wind', 'door',
              'chtOut_temp', 'chtOut_humidity', 'chtIn_humidity', 'chtIn_temp']
all_features = temp_cols + other_cols

# 4. Prepare containers
pc1_vectors = {}
pc1_projections = []

# 5. Loop through each timezone
for tz in df['timezone'].unique():
    group = df[df['timezone'] == tz].copy()

    # Drop rows with all NaNs
    group = group.dropna(subset=all_features, how='all')

    # Filter valid numeric columns
    valid_cols = []
    for col in all_features:
        if col in group.columns and pd.api.types.is_numeric_dtype(group[col]):
            std = group[col].std(skipna=True)
            if pd.notna(std) and std != 0:
                valid_cols.append(col)

    if not valid_cols or group[valid_cols].dropna().shape[0] < 2:
        print(f"⚠️ Skipping timezone '{tz}' due to insufficient usable data.")
        continue

    # Manual z-score standardization
    X = group[valid_cols].copy()
    for col in valid_cols:
        mean = X[col].mean(skipna=True)
        std = X[col].std(skipna=True)
        X[col] = (X[col] - mean) / std

    # Fill any remaining NaNs (optional but recommended)
    X = X.fillna(0)

    # PCA: eigendecomposition
    cov_matrix = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

# 🔍 Print PCA values
    print(f"\n📌 PCA Results for '{tz}':")
    total_var = np.sum(eigvals)
    for i, val in enumerate(eigvals):
        explained = val / total_var
        print(f"PC{i+1}: Eigenvalue = {val:.4f}, Explained Variance = {explained:.4%}")


# Save PC1 and project data
    pc1_vector = eigvecs[:, 0].reshape(-1, 1)
    pc1_vectors[tz] = pc1_vector

    projected = X.values @ eigvecs
    proj_df = pd.DataFrame(projected, columns=[f'PC{i+1}' for i in range(projected.shape[1])])
    proj_df['timezone'] = tz

    # Plot PC1 vs PC2
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=proj_df, x='PC1', y='PC2', alpha=0.7)
    plt.title(f'{tz.upper()} - PC1 vs PC2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Project only onto PC1 and store
    pc1_scores = X.values @ pc1_vector
    pc1_projections.append(pd.DataFrame({
        'PC1': pc1_scores.flatten(),
        'timezone': tz
    }))

# 6. Combine and plot PC1 projections from all timezones
final_pc1_df = pd.concat(pc1_projections, ignore_index=True)

plt.figure(figsize=(10, 6))
sns.boxplot(data=final_pc1_df, x='timezone', y='PC1')
plt.title("PC1 Projection per Timezone")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=final_pc1_df, x='PC1', y='timezone', hue='timezone', palette='Set2', alpha=0.7, s=60)

plt.title('PC1 vs PC2 Projection by Timezone')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Timezone')
plt.grid(True)
plt.tight_layout()
plt.show()
final_pc1_df.to_csv(REPO_ROOT / "pc1_projections_by_timezone.csv", index=False)
