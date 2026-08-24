import os
from pathlib import Path
import sys

LOGS = []
def LOG(s):
    LOGS.append(str(s))
    print(s, flush=True)

LOG("=== week4 nn compare (RF vs CNN vs RNN, four targets ===")

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = PROJECT_ROOT / "my_code" / "figures"
OUT_DIR = PROJECT_ROOT / "my_code" / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 11
N_EPOCHS = 30

train_files = ["historical", "ssp585", "ssp126", "ssp370"]
test_file = "ssp245"

targets_cfg = [
    ("tas", "K", {"hidden": 48, "epochs": N_EPOCHS, "batch": 16}),
    ("diurnal_temperature_range", "K", {"hidden": 32, "epochs": N_EPOCHS, "batch": 16, "short": "dtr"}),
    ("pr", "mm/day", {"hidden": 48, "epochs": N_EPOCHS, "batch": 16}),
    ("pr90", "mm/day", {"hidden": 64, "epochs": N_EPOCHS, "batch": 16}),
]

LOG("Build global X/Y")
X_train, eof_solvers = create_predictor_data(train_files, n_eofs=5)
Y_train_all = create_predictdand_data(train_files)
X_test = get_test_data(test_file, eof_solvers, n_eofs=5)
Y_test_all = create_predictdand_data([test_file])

X_train_arr = np.asarray(X_train.to_numpy(), dtype=np.float32)
X_test_arr = np.asarray(X_test.to_numpy(), dtype=np.float32)

# read RF predictions if exist, otherwise train a quick RF for 4 targets on-the-fly
RF_PRED_PATHS = [
    OUT_DIR / "outputs_ssp245_prediction_RF.nc",
    OUT_DIR / "outputs_ssp245_prediction_RF_4targets_smoke.nc",
]
rf_pred_path = None
for p in RF_PRED_PATHS:
    if p.exists():
        rf_pred_path = p
        break
if rf_pred_path is not None:
    LOG(f"Found RF predictions: {rf_pred_path.name}")
    ds_rf = xr.open_dataset(rf_pred_path)
else:
    LOG("No saved RF pred file, running tiny RF on the fly for 4 targets...")
    from esem import rf_model
    ds_rf_vars = {}
    for target, _, hp in targets_cfg:
        short = hp.get("short", target)
        LOG(f"tiny RF for {short}")
        rf = rf_model(X_train, Y_train_all[target], random_state=0, bootstrap=True, max_features=1.0,
                      **{"n_estimators": 30, "min_samples_split": 5, "min_samples_leaf": 7, "max_depth": 5})
        rf.train()
        pred, _ = rf.predict(X_test)
        ds_rf_vars[target] = xr.DataArray(np.asarray(pred, dtype=np.float32), dims=("sample", "lat", "lon"),
            coords=dict(lat=Y_test_all[target].lat, lon=Y_test_all[target].lon))
    first_pred_arr = next(iter(ds_rf_vars.values()))
    sample_dim = first_pred_arr.shape[0]
    ds_rf = xr.Dataset(ds_rf_vars).assign_coords(time=np.arange(sample_dim) + 2014)
    rf_pred_path = OUT_DIR / "outputs_ssp245_prediction_RF_onthefly.nc"
    ds_rf.to_netcdf(str(rf_pred_path), "w")
    LOG(f"saved on-the-fly RF pred: {rf_pred_path}")

weights_lat = np.cos(np.deg2rad(Y_test_all["tas"].lat.to_numpy())).astype(np.float32)
w_2d = weights_lat[:, None]
w_sum = w_2d.sum()

def lat_weighted_mse_wrapper(n_lat, n_lon):
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, (-1, n_lat, n_lon))
        y_pred = tf.reshape(y_pred, (-1, n_lat, n_lon))
        se = tf.square(y_true - y_pred)
        w = tf.constant(w_2d, dtype=tf.float32)
        return tf.reduce_sum(se * w, axis=[1, 2]) / tf.constant(w_sum, dtype=tf.float32)
    return loss


def build_cnn(dim_x, n_lat, n_lon, hidden=48):
    h1 = n_lat // 4
    w1 = n_lon // 4
    ch = 16
    inp = L.Input(shape=(dim_x,))
    h = L.Dense(h1 * w1 * ch, activation="silu")(inp)
    h = L.Reshape((h1, w1, ch))(h)
    h = L.Conv2DTranspose(hidden, 3, strides=2, padding="same", activation="silu")(h)
    h = L.Conv2D(hidden, 3, padding="same", activation="silu")(h)
    h = L.Conv2DTranspose(hidden, 3, strides=2, padding="same", activation="silu")(h)
    h = L.Conv2D(16, 3, padding="same", activation="silu")(h)
    out = L.Conv2D(1, 1, padding="same")(h)
    out = L.Reshape((n_lat, n_lon))(out)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=lat_weighted_mse_wrapper(n_lat, n_lon))
    return model


def build_rnn(dim_x, n_lat, n_lon, lstm_hidden=64, seq_len=SEQ_LEN):
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
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=lat_weighted_mse_wrapper(n_lat, n_lon))
    return model


def build_seq_samples(X, Y, seq_len):
    T = X.shape[0]
    xs = []
    ys = []
    for t in range(seq_len - 1, T):
        xs.append(X[t - seq_len + 1 : t + 1])
        ys.append(Y[t])
    return np.stack(xs, 0).astype(np.float32), np.stack(ys, 0).astype(np.float32)


def eval_rmse_late(truth_xa, pred_np):
    # truth_xa: xarray (T, lat, lon), pred_np: numpy(T, lat, lon)
    rmse_map = get_rmse(truth_xa[35:], pred_np[35:])
    return float(np.mean(np.asarray(rmse_map).reshape(-1)))


rows = []
cnn_preds = {}
rnn_preds = {}
history_cnn_all = {}
history_rnn_all = {}

for cfg in targets_cfg:
    target, unit, hp = cfg
    short = hp.get("short", target)
    LOG(f"\n==== Target: {short} ({unit})")
    Y_train = np.asarray(Y_train_all[target].values, dtype=np.float32)
    Y_test = np.asarray(Y_test_all[target].values, dtype=np.float32)
    Y_test_xa = Y_test_all[target]
    N_LAT = Y_train.shape[1]
    N_LON = Y_train.shape[2]
    DIM_X = X_train_arr.shape[1]
    hidden = hp.get("hidden", 48)
    epochs = hp.get("epochs", N_EPOCHS)
    batch = hp.get("batch", 16)

    # CNN
    LOG(f"build CNN ({short})")
    cnn = build_cnn(DIM_X, N_LAT, N_LON, hidden=hidden)
    LOG(f"CNN params: {cnn.count_params():,}")
    hc = cnn.fit(X_train_arr, Y_train, epochs=epochs, batch_size=batch,
                  validation_split=0.1, shuffle=True, verbose=2)
    pred_cnn = cnn.predict(X_test_arr, verbose=0).astype(np.float32)
    cnn_preds[target] = pred_cnn
    history_cnn_all[short] = (hc.history["loss"], hc.history.get("val_loss", hc.history["loss"]))
    rmse_cnn = eval_rmse_late(Y_test_xa, pred_cnn)
    LOG(f"CNN {short} late RMSE = {rmse_cnn:.6f} {unit}")

    # RNN sequence data
    LOG(f"build RNN seq ({short})")
    Xs_tr, Ys_tr = build_seq_samples(X_train_arr, Y_train, SEQ_LEN)
    Xs_te, Ys_te = build_seq_samples(X_test_arr, Y_test, SEQ_LEN)
    rnn = build_rnn(DIM_X, N_LAT, N_LON, seq_len=SEQ_LEN)
    LOG(f"RNN params: {rnn.count_params():,}")
    hr = rnn.fit(Xs_tr, Ys_tr, epochs=epochs, batch_size=batch,
                  validation_split=0.1, shuffle=True, verbose=2)
    pred_rnn_seq = rnn.predict(Xs_te, verbose=0).astype(np.float32)
    pred_rnn = np.zeros_like(Y_test, dtype=np.float32)
    if SEQ_LEN - 1 > 0:
        pred_rnn[:SEQ_LEN - 1] = Y_test[:SEQ_LEN - 1]
    pred_rnn[SEQ_LEN - 1:] = pred_rnn_seq
    rnn_preds[target] = pred_rnn
    history_rnn_all[short] = (hr.history["loss"], hr.history.get("val_loss", hr.history["loss"]))
    rmse_rnn = eval_rmse_late(Y_test_xa, pred_rnn)
    LOG(f"RNN {short} late RMSE = {rmse_rnn:.6f} {unit}")

    # RF from dataset
    if target in ds_rf.data_vars:
        rfp = np.asarray(ds_rf[target].values, dtype=np.float32)
        if rfp.ndim == 4:
            rfp = rfp.reshape(rfp.shape[0], rfp.shape[-3], rfp.shape[-2])
        rfp = rfp[:len(Y_test)]
        rmse_rf = eval_rmse_late(Y_test_xa, rfp)
    else:
        rmse_rf = np.nan
    LOG(f"RF  {short} late RMSE = {rmse_rf:.6f} {unit}")

    rows.append({
        "target": short,
        "unit": unit,
        "RMSE_RF": rmse_rf,
        "RMSE_CNN": rmse_cnn,
        "RMSE_RNN": rmse_rnn,
        "improv_CNN_vs_RF": (rmse_rf - rmse_cnn),
        "improv_RNN_vs_RF": (rmse_rf - rmse_rnn),
    })

# save predictions to netcdfs
def to_ds(preds_dict):
    vs = {}
    first_target = next(iter(preds_dict.keys()))
    sample_dim = preds_dict[first_target].shape[0]
    ref = Y_test_all[first_target]
    for target in preds_dict:
        vs[target] = xr.DataArray(
            preds_dict[target],
            dims=("sample", "lat", "lon"),
            coords=dict(lat=ref.lat, lon=ref.lon),
        )
    return xr.Dataset(vs).assign_coords(time=np.arange(sample_dim) + 2014)

cnn_ds = to_ds(cnn_preds)
rnn_ds = to_ds(rnn_preds)

cnn_nc = OUT_DIR / "outputs_ssp245_prediction_CNN.nc"
rnn_nc = OUT_DIR / "outputs_ssp245_prediction_RNN.nc"
cnn_ds.to_netcdf(str(cnn_nc), "w")
rnn_ds.to_netcdf(str(rnn_nc), "w")
LOG(f"saved CNN pred: {cnn_nc}")
LOG(f"saved RNN pred: {rnn_nc}")

# save scores
compare_df = pd.DataFrame(rows)
compare_path = OUT_DIR / "all_model_ssp245_scores_compare.csv"
compare_df.to_csv(compare_path, index=False)
LOG(f"saved compare csv: {compare_path}")

# RMSE bar figure
fig, axes = plt.subplots(len(targets_cfg), 1, figsize=(9, 3.2 * len(targets_cfg)), dpi=120, squeeze=False)
for i, cfg in enumerate(targets_cfg):
    target, unit, hp = cfg
    short = hp.get("short", target)
    row = compare_df.loc[i]
    models = ["RF", "CNN", "RNN"]
    ys = [row["RMSE_RF"], row["RMSE_CNN"], row["RMSE_RNN"]]
    ax = axes[i, 0]
    bars = ax.bar(models, ys, color=["#7f7f7f", "#1f77b4", "#2ca02c"], alpha=0.85)
    ax.set_title(f"{short} late RMSE ({unit})")
    ax.set_ylabel(unit)
    for b, v in zip(bars, ys):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig_path = FIG_DIR / "week4_models_rmse_compare.png"
fig.savefig(fig_path, bbox_inches="tight")
plt.close(fig)
LOG(f"saved compare figure: {fig_path}")

# training curves figure (loss of tas/pr90
fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
for short in ["tas", "pr90"]:
    if short in history_cnn_all:
        l, vl = history_cnn_all[short]
        axes[0].plot(np.arange(len(l)) + 1, l, label=f"CNN-{short}-train")
        axes[0].plot(np.arange(len(vl)) + 1, vl, linestyle="--", label=f"CNN-{short}-val")
axes[0].set_title("CNN training curves (lat-weighted MSE)")
axes[0].set_xlabel("epoch")
axes[0].legend()
for short in ["tas", "pr90"]:
    if short in history_rnn_all:
        l, vl = history_rnn_all[short]
        axes[1].plot(np.arange(len(l)) + 1, l, label=f"RNN-{short}-train")
        axes[1].plot(np.arange(len(vl)) + 1, vl, linestyle="--", label=f"RNN-{short}-val")
axes[1].set_title("RNN training curves")
axes[1].set_xlabel("epoch")
axes[1].legend()
fig.tight_layout()
curves_path = FIG_DIR / "week4_cnn_rnn_training_curves.png"
fig.savefig(curves_path, bbox_inches="tight")
plt.close(fig)
LOG(f"saved training curves: {curves_path}")

log_path = OUT_DIR / "week4_nn_compare.log"
log_path.write_text("\n".join(LOGS) + "\n", encoding="utf-8")
LOG(f"saved log: {log_path}")
LOG("=== week4 nn compare done ===")
