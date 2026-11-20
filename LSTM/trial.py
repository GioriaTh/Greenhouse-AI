import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
import tensorflow as tf
import optuna
from itertools import combinations
import gc

import os
from pathlib import Path

# Dataset path: defaults to the CSV in the repo root; override with GREENHOUSE_CSV=/path/to/file.csv
REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = Path(os.environ.get("GREENHOUSE_CSV", REPO_ROOT / "sensor_readings3rd.csv"))

# --- Configuration ---
csv_path = CSV_PATH

ntc_coords = {
    "temp1": (0, 3.0, 2.0), "temp9": (0, 3.0, 1.0), "temp8": (0, 3.0, 0.0),
    "temp12": (5, 3.0, 2.0), "temp10": (5, 3.0, 1.0), "temp11": (5, 3.0, 0.0),
    "temp2": (2.5, 1.5, 2.0), "temp6": (2.5, 1.5, 1.0), "temp7": (2.5, 1.5, 0.0),
    "temp5": (2.5, 0.0, 2.0), "temp4": (2.5, 0.0, 1.0), "temp3": (2.5, 0.0, 0.0),
}
all_ntcs = list(ntc_coords.keys())

env_cols = ['chtIn_humidity', 'chtIn_temp', 'chtOut_humidity', 'chtOut_temp',
            'ir_in', 'ir_out', 'wind', 'hour', 'minute']


def preprocess_data(csv_path, input_ntcs, target_ntcs):
    dataset = pd.read_csv(csv_path)
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], errors='coerce')
    dataset['hour'] = dataset['timestamp'].dt.hour
    dataset['minute'] = dataset['timestamp'].dt.minute

    fil_dat = dataset[(dataset['ir_in'] != 65535) & (dataset['ir_in'] != 1)].copy()
    fil_dat = fil_dat.sort_values(by='timestamp').reset_index(drop=True)

    input_cols = input_ntcs + env_cols
    X = fil_dat[input_cols].values
    y = fil_dat[target_ntcs].values

    return fil_dat, X, y, input_cols

# Sensor coordinate feature extractor
def get_sensor_coords_features(sensor_names, ntc_coords):
    return np.array([ntc_coords[name] for name in sensor_names])

# Prepare LSTM input sequences with spatial coords
def prepare_lstm_data(X, y, sensor_names, ntc_coords, sequence_length):
    if sensor_names:
        coord_features = get_sensor_coords_features(sensor_names, ntc_coords).flatten()
    else:
        coord_features = np.zeros(3)

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

def visualize_3d_greenhouse_heatmap(ntc_coords, values, title="3D Heatmap", cmap="coolwarm"):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.cm import ScalarMappable

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    # Draw greenhouse edges (walls and roof lines)
    edges = [
        [(0, 0, 0), (5, 0, 0)], [(5, 0, 0), (5, 3, 0)], [(5, 3, 0), (0, 3, 0)], [(0, 3, 0), (0, 0, 0)],
        [(0, 0, 2), (5, 0, 2)], [(5, 0, 2), (5, 3, 2)], [(5, 3, 2), (0, 3, 2)], [(0, 3, 2), (0, 0, 2)],
        [(0, 0, 0), (0, 0, 2)], [(0, 3, 0), (0, 3, 2)], [(5, 0, 0), (5, 0, 2)], [(5, 3, 0), (5, 3, 2)],
        [(0, 0, 2), (2.5, 0, 2.5)], [(2.5, 0, 2.5), (5, 0, 2)],
        [(0, 3, 2), (2.5, 3, 2.5)], [(2.5, 3, 2.5), (5, 3, 2)],
        [(2.5, 0, 2.5), (2.5, 3, 2.5)]
    ]
    for edge in edges:
        ax.plot(*zip(*edge), color='gray', linewidth=1)

    # Sensor scatter plot with colormap
    xs, ys, zs, temps = [], [], [], []
    for sensor, (x, y, z) in ntc_coords.items():
        if sensor in values:
            xs.append(x)
            ys.append(y)
            zs.append(z)
            temps.append(values[sensor])

    cmap_obj = plt.colormaps.get_cmap(cmap)
    norm = plt.Normalize(min(temps), max(temps))
    colors = cmap_obj(norm(temps))

    sc = ax.scatter(xs, ys, zs, c=colors, s=100, edgecolor='k', label='Sensors')

    # Label sensors
    for sensor, (x, y, z) in ntc_coords.items():
        if sensor in values:
            ax.text(x, y, z + 0.1, sensor, fontsize=8)

    ax.set_xlabel("X (Length)")
    ax.set_ylabel("Y (Width)")
    ax.set_zlabel("Z (Height)")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3)
    ax.set_zlim(0, 3)

    sm = ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Temperature (°C)")

    plt.tight_layout()
    plt.show()


#optuna
def objective(trial, X_train_processed, y_train, X_val_processed, y_val, current_input_ntcs, ntc_coords_map):
    sequence_length = trial.suggest_int("sequence_length", 10, 30, step=10)
    lstm_units = trial.suggest_int("lstm_units", 32, 128, step=16)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_choice = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])

    X_train_seq, y_train_seq = prepare_lstm_data(X_train_processed, y_train, current_input_ntcs, ntc_coords_map, sequence_length)
    X_val_seq, y_val_seq = prepare_lstm_data(X_val_processed, y_val, current_input_ntcs, ntc_coords_map, sequence_length)

    if X_train_seq.shape[0] < batch_size or X_val_seq.shape[0] < 1:
        raise optuna.exceptions.TrialPruned()

    model = build_model(input_shape=(sequence_length, X_train_seq.shape[2]), output_dim=y_train_seq.shape[1],
                        lstm_units=lstm_units, dropout_rate=dropout_rate)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) if optimizer_choice == "adam" \
        else tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="mse")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=20,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )

    val_loss = model.evaluate(X_val_seq, y_val_seq, verbose=0)
    return val_loss


# --- Global trackers ---
global_best_trial = None
global_best_combo = None
global_best_mse = float("inf")

fixed_inputs = ['temp6', 'temp10', 'temp11', 'temp12']
# --- Main Loop ---
for k in range(0, len(fixed_inputs) + 1):
    for combo in combinations(fixed_inputs, k):
        input_ntcs =list(combo)
        target_ntcs = [ntc for ntc in all_ntcs if ntc not in input_ntcs]

        if len(target_ntcs) == 0:
            continue

        try:
            print(f"\n Testing input NTCs: {input_ntcs}")
            fil_dat, X_full, y_full, _ = preprocess_data(csv_path, input_ntcs, target_ntcs)
            n_total = len(X_full)
            train_end_idx = int(n_total * 0.7)
            val_end_idx = int(n_total * 0.85)

            X_train_raw, y_train = X_full[:train_end_idx], y_full[:train_end_idx]
            X_val_raw, y_val = X_full[train_end_idx:val_end_idx], y_full[train_end_idx:val_end_idx]
            X_test_raw, y_test = X_full[val_end_idx:], y_full[val_end_idx:]

            if len(X_train_raw) == 0 or len(X_val_raw) == 0 or len(X_test_raw) == 0:
                continue

            sc = StandardScaler()
            X_train_scaled = sc.fit_transform(X_train_raw)
            X_val_scaled = sc.transform(X_val_raw)
            X_test_scaled = sc.transform(X_test_raw)

            study = optuna.create_study(direction="minimize")
            study.optimize(lambda t: objective(t, X_train_scaled, y_train, X_val_scaled, y_val, input_ntcs, ntc_coords), n_trials=10)

            best_trial = study.best_trial
            if best_trial.value < global_best_mse:
                global_best_mse = best_trial.value
                global_best_trial = best_trial
                global_best_combo = input_ntcs.copy()

        except Exception as e:
            print(f"Error on combo {input_ntcs}: {e}")
            continue

        tf.keras.backend.clear_session()
        gc.collect()

# --- Retrain Final Model ---
print("\n Finished all combinations.")
print(f" Best global input NTCs: {global_best_combo}")
print(f" Best global MSE: {global_best_mse}")
print(f" Best global parameters: {global_best_trial.params}")

input_ntcs = global_best_combo
target_ntcs = [ntc for ntc in all_ntcs if ntc not in input_ntcs]

fil_dat, X_full, y_full, _ = preprocess_data(csv_path, input_ntcs, target_ntcs)
n_total = len(X_full)
train_end_idx = int(n_total * 0.85)

X_trainval_raw, y_trainval = X_full[:train_end_idx], y_full[:train_end_idx]
X_test_raw, y_test = X_full[train_end_idx:], y_full[train_end_idx:]

sc = StandardScaler()
X_trainval_scaled = sc.fit_transform(X_trainval_raw)
X_test_scaled = sc.transform(X_test_raw)

params = global_best_trial.params
sequence_length = params['sequence_length']
lstm_units = params['lstm_units']
dropout_rate = params['dropout_rate']
learning_rate = params['learning_rate']
batch_size = params['batch_size']
optimizer_choice = params['optimizer']

X_seq_trainval, y_seq_trainval = prepare_lstm_data(X_trainval_scaled, y_trainval, input_ntcs, ntc_coords, sequence_length)
X_seq_test, y_seq_test = prepare_lstm_data(X_test_scaled, y_test, input_ntcs, ntc_coords, sequence_length)

final_model = build_model((sequence_length, X_seq_trainval.shape[2]), y_seq_trainval.shape[1], lstm_units, dropout_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) if optimizer_choice == 'adam' else tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
final_model.compile(optimizer=optimizer, loss='mse')

final_model.fit(X_seq_trainval, y_seq_trainval, epochs=50, batch_size=batch_size,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)],
                verbose=1)
# Predict
y_pred = final_model.predict(X_seq_test)

test_loss = final_model.evaluate(X_seq_test, y_seq_test, verbose=1)
print(f" Final Test MSE (Best Global Model): {test_loss}")


timestamps = fil_dat['timestamp'].reset_index(drop=True)
pred_timestamps = timestamps[sequence_length + int(len(timestamps) * 0.8):].reset_index(drop=True)




from sklearn.metrics import mean_squared_error

actual_df = pd.DataFrame(y_seq_test, columns=target_ntcs)
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

print("\n Matching predictions to timezones:")
for label, hour in zip(timezone_labels, target_hours):
    for i, ts in enumerate(pred_timestamps):
        if ts.hour == hour:
            pred_dict = {name: val for name, val in zip(target_ntcs, y_pred[i])}
            true_dict = {name: val for name, val in zip(target_ntcs, y_seq_test[i])}

            print(f" Matched {label} ({ts.strftime('%H:%M')}) at index {i}")
            visualize_3d_greenhouse_heatmap(ntc_coords, pred_dict, f"{label} - Prediction ({ts.strftime('%H:%M')})")
            visualize_3d_greenhouse_heatmap(ntc_coords, true_dict, f"{label} - Ground Truth ({ts.strftime('%H:%M')})")


            timezone_preds.append(y_pred[i])
            timezone_labels_matched.append(f"{label} ({ts.strftime('%H:%M')})")
            break

    if len(timezone_preds) < 5:
        print("⚠️ Not all timezones found — try broader hour ranges.")

