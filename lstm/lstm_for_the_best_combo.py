import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, models
import optuna

# Sensor coordinates
ntc_coords = {
    "temp1": (0, 3, 2), "temp2": (2.5, 3, 2),
    "temp4": (0, 3, 1), "temp5": (2.5, 3, 1),
    "temp6": (5, 3, 1), "temp7": (0, 3, 0),
    "temp8": (2.5, 3, 0), "temp9": (5, 3, 0),
    "temp10": (2.5, 0, 2), "temp11": (2.5, 0, 1),
    "temp12": (2.5, 0, 0),
}

# Preprocessing
def preprocess_data(csv_path, input_ntcs, target_ntcs):
    dataset = pd.read_csv(csv_path)
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], errors='coerce')
    dataset['hour'] = dataset['timestamp'].dt.hour
    dataset['minute'] = dataset['timestamp'].dt.minute
    fil_dat = dataset[(dataset['ir_in'] != 65535) & (dataset['ir_in'] != 1)].copy()

    env_cols = ['chtIn_humidity', 'chtIn_temp', 'chtOut_humidity', 'chtOut_temp',
                'ir_in', 'ir_out', 'wind', 'hour', 'minute']
    input_cols = input_ntcs + env_cols
    X = fil_dat[input_cols].values
    y = fil_dat[target_ntcs].values
    return fil_dat, X, y, input_cols

# PCA component suggestion and variance analysis
def compute_pca_and_importance(X_train_scaled, input_cols, variance_threshold=0.90):
    pca_full = PCA(n_components=None)
    pca_full.fit(X_train_scaled)
    explained_variance_ratio_cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_auto = np.argmax(explained_variance_ratio_cumulative >= variance_threshold) + 1
    max_components = X_train_scaled.shape[1]
    return pca_full, explained_variance_ratio_cumulative, n_components_auto, max_components

# PCA loadings and feature importance
def get_loadings_and_importance(pca, input_cols):
    loadings = pca.components_
    loadings_df = pd.DataFrame(loadings, columns=input_cols, index=[f'PC{i+1}' for i in range(pca.n_components_)])
    feature_importance_approx = loadings_df.abs().sum(axis=0).sort_values(ascending=False)
    return loadings_df, feature_importance_approx

# Sensor coordinate feature extractor
def get_sensor_coords_features(sensor_names, ntc_coords):
    return np.array([ntc_coords[name] for name in sensor_names])

# Prepare LSTM input sequences with spatial coords
def prepare_lstm_data(X, y, sensor_names, ntc_coords, sequence_length=10):
    coord_features = get_sensor_coords_features(sensor_names, ntc_coords).flatten()
    coord_features_tiled = np.tile(coord_features, (len(X), 1))
    X_augmented = np.hstack([X, coord_features_tiled])

    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X_augmented)):
        X_seq.append(X_augmented[i - sequence_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# Build LSTM model
def build_model(input_shape, output_dim, lstm_units, dropout_rate):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(lstm_units),
        layers.Dropout(dropout_rate),
        layers.Dense(output_dim)
    ])
    return model

def visualize_greenhouse(ntc_coords, values=None, cmap="coolwarm"):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.cm import ScalarMappable
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Greenhouse Sensor Heatmap')

    edges = [
        [(0, 0, 0), (5, 0, 0)], [(5, 0, 0), (5, 3, 0)], [(5, 3, 0), (0, 3, 0)], [(0, 3, 0), (0, 0, 0)],
        [(0, 0, 2), (5, 0, 2)], [(5, 0, 2), (5, 3, 2)], [(5, 3, 2), (0, 3, 2)], [(0, 3, 2), (0, 0, 2)],
        [(0, 0, 0), (0, 0, 2)], [(0, 3, 0), (0, 3, 2)], [(5, 0, 0), (5, 0, 2)], [(5, 3, 0), (5, 3, 2)],
    ]
    for edge in edges:
        ax.plot(*zip(*edge), color='black')
    ax.plot([0, 2.5, 5], [0, 0, 0], [2, 2.5, 2], color='brown')
    ax.plot([0, 2.5, 5], [3, 3, 3], [2, 2.5, 2], color='brown')

    vmin = np.min(list(values.values())) if values else None
    vmax = np.max(list(values.values())) if values else None

    sm = None
    if values:
        cmap_obj = plt.colormaps.get_cmap(cmap)
        norm = plt.Normalize(vmin, vmax)
        sm = ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])

    for name, (x, y, z) in ntc_coords.items():
        color = None
        if values and name in values:
            color = cmap_obj((values[name] - vmin) / (vmax - vmin + 1e-5))
        ax.scatter(x, y, z, color=color if color else 'red', s=80)
        ax.text(x, y, z + 0.1, name, fontsize=9)

    ax.set_xlabel("Length (X)")
    ax.set_ylabel("Width (Y)")
    ax.set_zlabel("Height (Z)")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3)
    ax.set_zlim(0, 2.5)

    if sm:
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label('Temperature (°C)')

    plt.tight_layout()
    plt.show()


# Optuna trial logic
def objective(trial):
    csv_path = "D:/jojo8/Documents/greenData/sensor_readings3rd.csv"
    input_ntcs = ["temp1", "temp2","temp4", "temp6", "temp7"]
    target_ntcs = [ "temp8", "temp10", "temp11", "temp5", "temp9", "temp12"]
    sequence_length = 10

    fil_dat, X, y, input_cols = preprocess_data(csv_path, input_ntcs, target_ntcs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # PCA suggestion
    pca_full, explained_var, n_components_auto, max_components = compute_pca_and_importance(X_train_scaled, input_cols)
    n_components_used = trial.suggest_int("n_components", n_components_auto, max_components)
    pca = PCA(n_components=n_components_used)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Get loadings and importance
    loadings_df, feature_importance_approx = get_loadings_and_importance(pca, input_cols)

    # Log results
    print(f"\nTrial {trial.number} — PCA Components Used: {n_components_used}")
    print("Cumulative Explained Variance:", explained_var[n_components_used - 1])
    print("\nPCA Loadings:")
    print(loadings_df)
    print("\nFeature Importance:")
    print(feature_importance_approx)

    # Prepare data with spatial info
    X_train_seq, y_train_seq = prepare_lstm_data(X_train_pca, y_train, input_ntcs, ntc_coords, sequence_length)
    X_test_seq, y_test_seq = prepare_lstm_data(X_test_pca, y_test, input_ntcs, ntc_coords, sequence_length)
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_seq, y_train_seq, test_size=0.2, random_state=42)

    # Model hyperparams
    lstm_units = trial.suggest_int("lstm_units", 32, 128)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    model = build_model((sequence_length, X_train_final.shape[2]), y_train_final.shape[1], lstm_units, dropout_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    model.fit(X_train_final, y_train_final, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=0)

    y_pred = model.predict(X_val)
    mse = tf.keras.losses.MeanSquaredError()(y_val, y_pred).numpy()

    return mse


# Run optimization
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)


    # === After study.optimize(...) ===
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (MSE): {trial.value}")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # === Re-run best trial logic to animate and visualize ===
    input_ntcs = ["temp1", "temp2"]
    target_ntcs = ["temp4", "temp6", "temp7", "temp8", "temp10", "temp11", "temp5", "temp9", "temp12"]
    sequence_length = 10
    csv_path = "D:/jojo8/Documents/greenData/sensor_readings3rd.csv"

    # Preprocess
    fil_dat, X, y, input_cols = preprocess_data(csv_path, input_ntcs, target_ntcs)
    timestamps = fil_dat['timestamp'].reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    pred_timestamps = timestamps[sequence_length + int(len(timestamps) * 0.8):].reset_index(drop=True)
# Apply best PCA
    pca = PCA(n_components=trial.params["n_components"])
    X_test_pca = pca.fit(X_train_scaled).transform(X_test_scaled)

    # Prepare LSTM sequences
    X_test_seq, y_test_seq = prepare_lstm_data(X_test_pca, y_test, input_ntcs, ntc_coords, sequence_length)

    # Rebuild model with best trial params
    model = build_model(
        input_shape=(sequence_length, X_test_seq.shape[2]),
        output_dim=len(target_ntcs),
        lstm_units=trial.params["lstm_units"],
        dropout_rate=trial.params["dropout_rate"]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=trial.params["learning_rate"]), loss="mse")
    model.fit(X_test_seq, y_test_seq, epochs=20, batch_size=32, verbose=0)

    # Predict
    y_pred = model.predict(X_test_seq)

    from sklearn.metrics import mean_squared_error
    # --- Setup ---
    actual_df = pd.DataFrame(y_test_seq, columns=target_ntcs)
    pred_df = pd.DataFrame(y_pred, columns=target_ntcs)
    actual_df["hour"] = pred_timestamps[:len(actual_df)].dt.hour
    pred_df["hour"] = pred_timestamps[:len(pred_df)].dt.hour

    # --- Compute grouped hourly means ---
    actual_mean_by_hour = actual_df.groupby("hour").mean()
    pred_mean_by_hour = pred_df.groupby("hour").mean()

    # --- Plot 1: Actual vs Predicted per Sensor ---
    for sensor in target_ntcs:
        plt.figure(figsize=(8, 4))
        plt.plot(actual_mean_by_hour.index, actual_mean_by_hour[sensor], label="Actual", linewidth=2)
        plt.plot(pred_mean_by_hour.index, pred_mean_by_hour[sensor], label="Predicted", linestyle="--")
        plt.title(f"[{sensor}] Avg Temperature per Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Temperature (°C)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- Compute MSE per hour ---
    mse_df = (pred_df[target_ntcs] - actual_df[target_ntcs]) ** 2
    mse_df["hour"] = actual_df["hour"]
    mse_by_hour = mse_df.groupby("hour").mean()

    # --- Plot 2: MSE per Sensor ---
    for sensor in target_ntcs:
        plt.figure(figsize=(8, 4))
        plt.plot(mse_by_hour.index, mse_by_hour[sensor], color="tomato", linewidth=2)
        plt.title(f"[{sensor}] MSE per Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("MSE (°C²)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()



# === Timezone-based visualization setup ===
    timezone_labels = ["Dawn", "Morning", "Afternoon", "Evening", "Night"]
    target_hours = [5, 9, 13, 17, 21]
    timestamps = fil_dat['timestamp'].reset_index(drop=True)
    pred_timestamps = timestamps[sequence_length + int(len(timestamps) * 0.8):].reset_index(drop=True)

    timezone_preds = []
    timezone_labels_matched = []

    print("\n🔍 Matching predictions to timezones:")
    for label, hour in zip(timezone_labels, target_hours):
        for i, ts in enumerate(pred_timestamps):
            if ts.hour == hour:
                pred_dict = {name: val for name, val in zip(target_ntcs, y_pred[i])}
                true_dict = {name: val for name, val in zip(target_ntcs, y_test_seq[i])}

                print(f"🕒 Matched {label} ({ts.strftime('%H:%M')}) at index {i}")
                visualize_greenhouse(ntc_coords, pred_dict)
                visualize_greenhouse(ntc_coords, true_dict)

                timezone_preds.append(y_pred[i])
                timezone_labels_matched.append(f"{label} ({ts.strftime('%H:%M')})")
                break

    if len(timezone_preds) < 5:
        print("⚠️ Not all timezones found — try broader hour ranges.")

