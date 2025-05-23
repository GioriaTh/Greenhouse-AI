import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models
import tensorflow as tf
import optuna
from itertools import combinations

# --- Configuration ---
csv_path = "D:/jojo8/Documents/greenData/sensor_readings3rd.csv"

ntc_coords = {
    "temp1": (0, 3, 2), "temp2": (2.5, 3, 2),
#    "temp3": (5, 3, 2),
#    "temp4": (0, 3, 1), "temp5": (2.5, 3, 1),
#    "temp6": (5, 3, 1), "temp7": (0, 3, 0), "temp8": (2.5, 3, 0),
#    "temp9": (5, 3, 0), "temp10": (2.5, 0, 2), "temp11": (2.5, 0, 1),
#    "temp12": (2.5, 0, 0),
}
all_ntcs = list(ntc_coords.keys())

env_cols = ['chtIn_humidity', 'chtIn_temp', 'chtOut_humidity', 'chtOut_temp',
            'ir_in', 'ir_out', 'wind', 'hour', 'minute']


# --- Utility Functions ---
def preprocess_data(csv_path, input_ntcs, target_ntcs):
    dataset = pd.read_csv(csv_path)
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], errors='coerce')
    dataset['hour'] = dataset['timestamp'].dt.hour
    dataset['minute'] = dataset['timestamp'].dt.minute
    fil_dat = dataset[(dataset['ir_in'] != 65535) & (dataset['ir_in'] != 1)].copy()
    input_cols = input_ntcs + env_cols
    X = fil_dat[input_cols].values
    y = fil_dat[target_ntcs].values
    return fil_dat, X, y, input_cols

def compute_pca_and_importance(X_train_scaled, input_cols, variance_threshold=0.90):
    pca_full = PCA(n_components=None)
    pca_full.fit(X_train_scaled)
    explained_variance_ratio_cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_auto = np.argmax(explained_variance_ratio_cumulative >= variance_threshold) + 1
    max_components = X_train_scaled.shape[1]
    return pca_full, explained_variance_ratio_cumulative, n_components_auto, max_components

def get_loadings_and_importance(pca, input_cols):
    loadings = pca.components_
    loadings_df = pd.DataFrame(loadings, columns=input_cols, index=[f'PC{i+1}' for i in range(pca.n_components_)])
    feature_importance_approx = loadings_df.abs().sum(axis=0).sort_values(ascending=False)
    return loadings_df, feature_importance_approx

def prepare_lstm_data(X, y, sensor_names, ntc_coords, sequence_length=10):
    if sensor_names:
        coord_features = np.array([ntc_coords[name] for name in sensor_names]).flatten()
    else:
        coord_features = np.zeros(3)
    coord_features_tiled = np.tile(coord_features, (len(X), 1))
    X_augmented = np.hstack([X, coord_features_tiled])
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X_augmented)):
        X_seq.append(X_augmented[i - sequence_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def build_model(input_shape, output_dim, lstm_units, dropout_rate):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(lstm_units),
        layers.Dropout(dropout_rate),
        layers.Dense(output_dim)
    ])
    return model

def objective(trial, X_train_pca, y_train, sequence_length, input_ntcs):
    X_train_seq, y_train_seq = prepare_lstm_data(X_train_pca, y_train, input_ntcs, ntc_coords, sequence_length)
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_seq, y_train_seq, test_size=0.2, random_state=42)

    lstm_units = trial.suggest_int("lstm_units", 32, 128)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    model = build_model((sequence_length, X_train_final.shape[2]), y_train_final.shape[1], lstm_units, dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    model.fit(X_train_final, y_train_final, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=0)

    y_pred = model.predict(X_val)
    mse = tf.keras.losses.MeanSquaredError()(y_val, y_pred).numpy()
    return mse

results = []
sequence_length = 10

for k in range(0, len(all_ntcs)):
    for combo in combinations(all_ntcs, k):
        input_ntcs = list(combo)
        target_ntcs = [ntc for ntc in all_ntcs if ntc not in input_ntcs]

        try:
            print(f"\n📊 Testing input NTCs: {input_ntcs}")

            # Preprocess and scale
            fil_dat, X, y, input_cols = preprocess_data(csv_path, input_ntcs, target_ntcs)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            sc = StandardScaler()
            X_train_scaled = sc.fit_transform(X_train)

            # PCA and variance
            pca_full, explained_var, n_components_auto, max_components = compute_pca_and_importance(X_train_scaled, input_cols)
            n_components_used = n_components_auto
            pca = PCA(n_components=n_components_used)
            X_train_pca = pca.fit_transform(X_train_scaled)
            loadings_df, feature_importance_approx = get_loadings_and_importance(pca, input_cols)

            # Optimize
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective(trial, X_train_pca, y_train, sequence_length, input_ntcs), n_trials=2)

            best_trial = study.best_trial
            results.append({
                "input_ntcs": input_ntcs,
                "target_ntcs": target_ntcs,
                "num_inputs": len(input_ntcs),
                "n_pca_components": n_components_used,
                "explained_variance": explained_var[n_components_used - 1],
                "best_mse": best_trial.value,
                "best_params": best_trial.params,
                "top_features": feature_importance_approx.head(5).to_dict()
            })

            print(f"✅ MSE: {best_trial.value:.4f} for inputs {input_ntcs}")

        except Exception as e:
            print(f"❌ Skipping combo {input_ntcs} due to error: {e}")


results_df = pd.DataFrame(results)
valid_results = results_df[results_df["explained_variance"] >= 0.90]

if len(valid_results) > 0:
    best = valid_results.sort_values(by=["num_inputs", "best_mse"]).iloc[0]
    print("\n✅ Best configuration across all input combinations:")
    print(f"- Input NTCs: {best['input_ntcs']}")
    print(f"- Target NTCs: {best['target_ntcs']}")
    print(f"- Number of Input Sensors: {best['num_inputs']}")
    print(f"- PCA Components: {best['n_pca_components']}")
    print(f"- Explained Variance: {best['explained_variance']:.2%}")
    print(f"- MSE: {best['best_mse']:.4f}")
    print("\n🔥 Top 5 Most Influential Features (PCA Loadings):")
    for feature, score in best["top_features"].items():
        print(f"  {feature}: {score:.4f}")
else:
    print("\n⚠️ No combination achieved at least 90% explained variance.")

