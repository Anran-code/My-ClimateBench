import os
from pathlib import Path
import sys

LOGS = []
def LOG(s):
    LOGS.append(str(s))
    print(s, flush=True)

LOG("=== week4 nn quickstart (CNN + RNN, tas single target) ===")

EXE = Path(sys.executable)
if sys.platform.startswith("win") and EXE.exists():
    ENV_ROOT = EXE.parent.parent if EXE.parent.name == "Scripts" else EXE.parent
    DLL_DIRS = [
        ENV_ROOT / "Library" / "bin",
        ENV_ROOT / "Library" / "mingw-w64" / "bin",
        ENV_ROOT / "Library" / "usr" / "bin",
        ENV_ROOT / "Scripts",
        ENV_ROOT / "bin",
    ]
    for d in DLL_DIRS:
        if d.is_dir():
            os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(d))

import netCDF4  # noqa: F401

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xarray as xr

from baseline_models.utils import (
    create_predictor_data,
    create_predictdand_data,
    get_test_data,
    get_rmse,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

FIG_DIR = PROJECT_ROOT / "my_code" / "figures"
OUT_DIR = PROJECT_ROOT / "my_code" / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

train_files = ["historical", "ssp585", "ssp126", "ssp370"]
test_file = "ssp245"
target = "tas"

LOG("build X/Y data for tas only ...")
X_train, eof_solvers = create_predictor_data(train_files, n_eofs=5)
Y_train_all = create_predictdand_data(train_files)
X_test = get_test_data(test_file, eof_solvers, n_eofs=5)
Y_test_all = create_predictdand_data([test_file])

X_train_arr = np.asarray(X_train.to_numpy(), dtype=np.float32)
X_test_arr = np.asarray(X_test.to_numpy(), dtype=np.float32)

Y_train_tas = np.asarray(Y_train_all[target].values, dtype=np.float32)   # (N, 96, 144)
Y_test_tas = np.asarray(Y_test_all[target].values, dtype=np.float32)     # (86, 96, 144)

N_LAT = Y_train_tas.shape[1]
N_LON = Y_train_tas.shape[2]
DIM_X = X_train_arr.shape[1]

# precompute lat weights for eval
weights_lat = np.cos(np.deg2rad(Y_train_all[target].lat.to_numpy())).astype(np.float32)
w_2d = weights_lat[:, None]
w_sum = w_2d.sum()

def lat_weighted_mse(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, N_LAT, N_LON))
    y_pred = tf.reshape(y_pred, (-1, N_LAT, N_LON))
    se = tf.square(y_true - y_pred)
    w = tf.constant(w_2d, dtype=tf.float32)
    return tf.reduce_sum(se * w, axis=[1, 2]) / tf.constant(w_sum, dtype=tf.float32)


def build_cnn_model(dim_x=DIM_X, n_lat=N_LAT, n_lon=N_LON, hidden=48):
    inp = L.Input(shape=(dim_x,))
    h1 = n_lat // 4
    w1 = n_lon // 4
    ch = 16
    h = L.Dense(h1 * w1 * ch, activation="silu")(inp)
    h = L.Reshape((h1, w1, ch))(h)
    h = L.Conv2DTranspose(hidden, 3, strides=2, padding="same", activation="silu")(h)
    h = L.Conv2D(hidden, 3, padding="same", activation="silu")(h)
    h = L.Conv2DTranspose(hidden, 3, strides=2, padding="same", activation="silu")(h)
    h = L.Conv2D(16, 3, padding="same", activation="silu")(h)
    out = L.Conv2D(1, 1, padding="same")(h)
    out = L.Reshape((n_lat, n_lon))(out)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=lat_weighted_mse)
    return model


def build_rnn_model(dim_x=DIM_X, n_lat=N_LAT, n_lon=N_LON, lstm_hidden=64, seq_len=11):
    inp = L.Input(shape=(seq_len, dim_x))
    h = L.Dense(128, activation="silu")(inp)
    h = L.LSTM(lstm_hidden, return_sequences=True, activation="tanh", recurrent_activation="sigmoid")(h)
    h = L.LSTM(lstm_hidden, return_sequences=False, activation="tanh", recurrent_activation="sigmoid")(h)
    h1 = n_lat // 4
    w1 = n_lon // 4
    ch = 8
    h = L.Dense(h1 * w1 * ch, activation="silu")(h)
    h = L.Reshape((h1, w1, ch))(h)
    h = L.Conv2DTranspose(32, 3, strides=2, padding="same", activation="silu")(h)
    h = L.Conv2D(32, 3, padding="same", activation="silu")(h)
    h = L.Conv2DTranspose(32, 3, strides=2, padding="same", activation="silu")(h)
    out = L.Conv2D(1, 1, padding="same")(h)
    out = L.Reshape((n_lat, n_lon))(out)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=lat_weighted_mse)
    return model


def build_seq_samples(X, Y, seq_len=11):
    T = X.shape[0]
    xs = []
    ys = []
    for t in range(seq_len - 1, T):
        xs.append(X[t - seq_len + 1 : t + 1])
        ys.append(Y[t])
    return np.stack(xs, 0).astype(np.float32), np.stack(ys, 0).astype(np.float32)


# CNN train / predict
LOG("build CNN")
cnn = build_cnn_model()
LOG(f"CNN params: {cnn.count_params():,}")
LOG("fit CNN (small epochs for quickstart)")
hist_cnn = cnn.fit(X_train_arr, Y_train_tas, epochs=25, batch_size=16,
                   validation_split=0.1, shuffle=True, verbose=2)
pred_cnn = cnn.predict(X_test_arr, verbose=0).astype(np.float32)
LOG(f"CNN test pred shape: {pred_cnn.shape}")

pred_cnn_ds = xr.Dataset({
    target: xr.DataArray(
        pred_cnn,
        dims=("sample", "lat", "lon"),
        coords=dict(lat=Y_test_all[target].lat, lon=Y_test_all[target].lon),
    )
}).assign_coords(time=np.arange(pred_cnn.shape[0]) + 2014)
cnn_nc = OUT_DIR / "outputs_ssp245_prediction_CNN_tas_smoke.nc"
pred_cnn_ds.to_netcdf(str(cnn_nc), "w")
LOG(f"saved CNN pred nc: {cnn_nc}")

rmse_map_cnn = get_rmse(Y_test_all[target][35:], pred_cnn[35:])
rmse_cnn = float(np.mean(np.asarray(rmse_map_cnn).reshape(-1)))
LOG(f"CNN {target} eval rmse(35:) = {rmse_cnn:.6f}")

# RNN prepare sequence data
SEQ_LEN = 11
LOG(f"build RNN seq samples (seq_len={SEQ_LEN})")
Xseq_train, Yseq_train = build_seq_samples(X_train_arr, Y_train_tas, SEQ_LEN)
Xseq_test, Yseq_test = build_seq_samples(X_test_arr, Y_test_tas, SEQ_LEN)
LOG(f"Xseq_train {Xseq_train.shape}, Yseq_train {Yseq_train.shape}")
LOG(f"Xseq_test  {Xseq_test.shape},  Yseq_test  {Yseq_test.shape}")

LOG("build RNN")
rnn = build_rnn_model(seq_len=SEQ_LEN)
LOG(f"RNN params: {rnn.count_params():,}")
LOG("fit RNN")
hist_rnn = rnn.fit(Xseq_train, Yseq_train, epochs=25, batch_size=16,
                   validation_split=0.1, shuffle=True, verbose=2)
pred_rnn_seq = rnn.predict(Xseq_test, verbose=0).astype(np.float32)
# pad first SEQ_LEN-1 predictions with simple truth (conservative placeholder)
pred_rnn = np.zeros_like(Y_test_tas, dtype=np.float32)
if SEQ_LEN - 1 > 0:
    pred_rnn[:SEQ_LEN - 1] = Y_test_tas[:SEQ_LEN - 1]
pred_rnn[SEQ_LEN - 1:] = pred_rnn_seq
LOG(f"RNN test pred shape: {pred_rnn.shape}")

pred_rnn_ds = xr.Dataset({
    target: xr.DataArray(
        pred_rnn,
        dims=("sample", "lat", "lon"),
        coords=dict(lat=Y_test_all[target].lat, lon=Y_test_all[target].lon),
    )
}).assign_coords(time=np.arange(pred_rnn.shape[0]) + 2014)
rnn_nc = OUT_DIR / "outputs_ssp245_prediction_RNN_tas_smoke.nc"
pred_rnn_ds.to_netcdf(str(rnn_nc), "w")
LOG(f"saved RNN pred nc: {rnn_nc}")

rmse_map_rnn = get_rmse(Y_test_all[target][35:], pred_rnn[35:])
rmse_rnn = float(np.mean(np.asarray(rmse_map_rnn).reshape(-1)))
LOG(f"RNN {target} eval rmse(35:) = {rmse_rnn:.6f}")

# write scores
scores_path = OUT_DIR / "cnn_rnn_tas_smoke_scores.csv"
pd.DataFrame([
    {"model": "CNN_tas_smoke", "rmse_late_35plus": rmse_cnn},
    {"model": "RNN_tas_smoke", "rmse_late_35plus": rmse_rnn},
]).to_csv(scores_path, index=False)
LOG(f"saved scores: {scores_path}")

# quick training curves figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
axes[0].plot(hist_cnn.history["loss"], label="CNN train")
axes[0].plot(hist_cnn.history["val_loss"], label="CNN val")
axes[0].set_title("CNN training curve (lat-weighted MSE)")
axes[0].set_xlabel("epoch")
axes[0].legend()
axes[1].plot(hist_rnn.history["loss"], label="RNN train")
axes[1].plot(hist_rnn.history["val_loss"], label="RNN val")
axes[1].set_title("RNN training curve (lat-weighted MSE)")
axes[1].set_xlabel("epoch")
axes[1].legend()
fig.tight_layout()
curves_path = FIG_DIR / "week4_cnn_rnn_tas_training_curves_smoke.png"
fig.savefig(curves_path, bbox_inches="tight")
plt.close(fig)
LOG(f"saved training curves: {curves_path}")

log_path = OUT_DIR / "week4_nn_quickstart.log"
log_path.write_text("\n".join(LOGS) + "\n", encoding="utf-8")
LOG(f"saved log: {log_path}")
LOG("=== week4 nn quickstart done ===")
