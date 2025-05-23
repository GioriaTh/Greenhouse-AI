# importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("D:\jojo8\Documents\greenData\sensor_readings3rd.csv")

# 2. Extract time components and create timezone labels
dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], errors='coerce')
dataset['hour'] = dataset['timestamp'].dt.hour
dataset['minute'] = dataset['timestamp'].dt.minute

fil_dat = dataset[(dataset['ir_in'] != 65535) &(dataset['ir_in']!= 1)].copy()

temp_cols = [f'temp{i}' for i in range(1, 13)]
other_cols = [ 'ir_in', 'ir_out', 'wind', 'door',  'chtOut_temp',
               'chtOut_humidity', 'chtIn_humidity', 'chtIn_temp', 'hour','minute']

X = fil_dat[other_cols].values
y = fil_dat[temp_cols].values


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# performing preprocessing part
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
dataset.columns

from sklearn.decomposition import PCA


pca_full = PCA(n_components=None)
pca_full.fit(X_train)

explained_variance_ratio_cumulative = np.cumsum(pca_full.explained_variance_ratio_)


plt.plot(range(1, len(explained_variance_ratio_cumulative) + 1), explained_variance_ratio_cumulative)

feature_names = other_cols
n_components = np.argmax(explained_variance_ratio_cumulative >= 0.90) + 1
print(f"Number of components to explain at least 90% variance: {n_components}")
# 1. Fit PCA for the desired number of components (e.g., 7 for ~90% variance)
pca = PCA(n_components=n_components)
pca.fit(X_train) # X_train should be scaled for PCA to work best

# 2. Get the loadings
loadings = pca.components_

# 3. Create a DataFrame for better readability
# Rows are PCs, columns are original features
loadings_df = pd.DataFrame(loadings,
                           columns=feature_names, # Your original feature names
                           index=[f'PC{i+1}' for i in range(n_components)])

print("Loadings of original features on each Principal Component:")
print(loadings_df)

# 4. Analyze the loadings_df:
# For each PC (PC1 to PC7), identify the original features with high absolute values.
# These are the features that are most heavily "loaded" onto that PC.

# print("\nFeatures with highest absolute loadings for each PC (example for PC1):")
# pc1_loadings = loadings_df.loc['PC1']
# sorted_pc1_loadings = pc1_loadings.abs().sort_values(ascending=False)
# print("PC1 influential features (and their loadings):\n", pc1_loadings[sorted_pc1_loadings.index])

# You can also look at the sum of absolute loadings across these 7 PCs for each feature
# This gives a rough idea of overall importance for this 90% variance subspace
feature_importance_approx = loadings_df.abs().sum(axis=0).sort_values(ascending=False)
print("\nApproximate overall importance of features for these PCs (sum of abs loadings):")
print(feature_importance_approx)



# === Start NTC–replacement importance loop ===
from sklearn.feature_selection import mutual_info_regression

mi_results = {}

for removed_ntc in temp_cols:
    remaining_ntcs = [t for t in temp_cols if t != removed_ntc]

    X_remain = fil_dat[remaining_ntcs].values
    y_target = fil_dat[removed_ntc].values

    # Compute mutual information between each remaining sensor and the removed one
    mi = mutual_info_regression(X_remain, y_target, discrete_features=False)

    mi_series = pd.Series(mi, index=remaining_ntcs).sort_values(ascending=False)
    top_sensor = mi_series.idxmax()

    mi_results[removed_ntc] = {
        'most_informative_sensor': top_sensor,
        'mutual_info_score': mi_series.max(),
        'mi_ranking': mi_series
    }

    # Print the top 3 sensors
    print(f"\n=== Removed: {removed_ntc} ===")
    print("Top 3 remaining NTCs based on mutual information:")
    print(mi_series.head(3))

# Create summary
summary_mi_df = pd.DataFrame({
    'Removed NTC': list(mi_results.keys()),
    'Most Informative Remaining Sensor': [v['most_informative_sensor'] for v in mi_results.values()],
    'Mutual Info Score': [v['mutual_info_score'] for v in mi_results.values()]
})

print("\n=== Summary: Most Informative NTCs Based on Mutual Information ===")
print(summary_mi_df.to_string(index=False))

