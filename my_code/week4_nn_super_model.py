import os
from pathlib import Path
import sys
import time

LOGS = []
def LOG(s):
    LOGS.append(str(s))
    print(s, flush=True)

LOG("=== week4 SUPER MODEL: Static channels (lat/lon2/land/orog) + lat-loss + gradient+spectrum reg ===")

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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
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
N_EPOCHS = 20
LAMBDA_GRAD = 0.01
LAMBDA_SPEC = 0.005

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

X_train_arr = np.asarray(X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train.values, dtype=np.float32)
X_test_arr = np.asarray(X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test.values, dtype=np.float32)

weights_lat = np.cos(np.deg2rad(Y_test_all["tas"].lat.to_numpy())).astype(np.float32)
w_2d = weights_lat[:, None]
w_sum = float(w_2d.sum())

# ================== Build 4 static channels (96, 144, 4): sin(lat), sin(2lon), land_bin, orog_bin ==================
def build_static_channels(lat_arr, lon_arr):
    n_lat, n_lon = len(lat_arr), len(lon_arr)
    lat_r = np.deg2rad(np.asarray(lat_arr, dtype=np.float32))  # (96,)
    lon_r = np.deg2rad(np.asarray(lon_arr, dtype=np.float32))  # (144,)
    sin_lat = np.sin(lat_r)[:, None]   # (96,1)  high mag at poles (signal: high-lat warming)
    sin2lon = np.sin(2.0 * lon_r)[None, :]  # (1,144) ITCZ/continental bandings
    ch_lat = np.broadcast_to(sin_lat, (n_lat, n_lon)).astype(np.float32)
    ch_lon = np.broadcast_to(sin2lon, (n_lat, n_lon)).astype(np.float32)

    # land fraction proxy: use tas training climatology absolute grad as terrain proxy
    # (ClimateBench data doesn't expose orog/land as standalone static vars)
    Y_tas_train = np.asarray(Y_train_all["tas"].values, dtype=np.float32)
    tas_clim = np.mean(Y_tas_train, axis=0)  # (96,144)
    # normalize
    tc = (tas_clim - np.mean(tas_clim)) / (np.std(tas_clim) + 1e-8)
    gy = np.zeros_like(tc)
    gx = np.zeros_like(tc)
    gy[1:] = np.abs(np.diff(tc, axis=0))
    gx[:, 1:] = np.abs(np.diff(tc, axis=1))
    grad_abs = np.sqrt(gy * gy + gx * gx)
    grad_n = (grad_abs - np.mean(grad_abs)) / (np.std(grad_abs) + 1e-8)
    # land bin: high latitudes where std of pr is elevated? use pr clim std
    Y_pr_train = np.asarray(Y_train_all["pr"].values, dtype=np.float32)
    pr_std = np.std(Y_pr_train, axis=0)
    pr_std_n = (pr_std - np.mean(pr_std)) / (np.std(pr_std) + 1e-8)
    # Two "static-like" but climatology-proxy channels
    ch_terrain_proxy = np.clip(grad_n, -2.0, 2.0).astype(np.float32)
    ch_land_precip_proxy = np.clip(pr_std_n, -2.0, 2.0).astype(np.float32)

    static = np.stack([ch_lat, ch_lon, ch_terrain_proxy, ch_land_precip_proxy], axis=-1)  # (96, 144, 4)
    LOG(f"  static channels shape: {static.shape}; range = [{static.min():.3f}, {static.max():.3f}]")
    return static

N_LAT_GLOBAL = len(Y_test_all["tas"].lat)
N_LON_GLOBAL = len(Y_test_all["tas"].lon)
STATIC_4CH = build_static_channels(np.asarray(Y_test_all["tas"].lat.to_numpy()),
                                   np.asarray(Y_test_all["tas"].lon.to_numpy()))

# ================== Loss: lat_weighted_mse + gradient penalty + spectrum penalty ==================
def make_super_loss(n_lat, n_lon, l_grad=LAMBDA_GRAD, l_spec=LAMBDA_SPEC):
    w2d = tf.constant(w_2d, dtype=tf.float32)
    ws = tf.constant(w_sum, dtype=tf.float32)

    # sobel penalty reduces H/W by 2 via inner valid diff -> use centered valid weights
    w2d_inner = tf.constant(np.ones((n_lat - 2, n_lon - 2), dtype=np.float32), dtype=tf.float32)
    wlat_inner = tf.constant(weights_lat[1:-1, None].astype(np.float32), dtype=tf.float32)  # (H-2,1)
    w_int = wlat_inner * w2d_inner
    gws = tf.reduce_sum(w_int)

    def _sobel_abs(x):
        gx = x[:, :, 2:] - x[:, :, :-2]
        gy = x[:, 2:, 1:-1] - x[:, :-2, 1:-1]
        # gy: (B, H-2, W-2) ; gx: (B, H, W-2). Align by valid center slice.
        gx_center = gx[:, 1:-1, :]
        return tf.sqrt(gx_center * gx_center + gy * gy + 1e-12)

    def loss(y_true, y_pred):
        yt = tf.reshape(y_true, (-1, n_lat, n_lon))
        yp = tf.reshape(y_pred, (-1, n_lat, n_lon))
        se = tf.square(yt - yp)
        l_mse = tf.reduce_sum(se * w2d, axis=[1, 2]) / ws

        # gradient consistency (edge-aware penalty, encourages sharp fronts)
        gt = _sobel_abs(yt)
        gp = _sobel_abs(yp)
        # gt/gp are (B, H-2, W-2); weighted by inner weights
        l_grad = tf.reduce_sum(tf.abs(gt - gp) * w_int, axis=[1,2]) / gws

        # spectrum penalty (match |FFT| magnitude on lat-banded normalized signal)
        # guard: disable if shapes unknown at trace time
        if yt.shape.rank == 3 and yt.shape[1] is not None and yt.shape[2] is not None:
            yt_n = yt - tf.reduce_mean(yt, axis=[1,2], keepdims=True)
            yp_n = yp - tf.reduce_mean(yp, axis=[1,2], keepdims=True)
            Ft = tf.signal.rfft2d(yt_n)
            Fp = tf.signal.rfft2d(yp_n)
            mag_t = tf.abs(Ft)
            mag_p = tf.abs(Fp)
            Hfi = tf.cast(tf.shape(mag_t)[1], mag_t.dtype)
            W2fi = tf.cast(tf.shape(mag_t)[2], mag_t.dtype)
            iy = tf.cast(tf.range(tf.shape(mag_t)[1]), mag_t.dtype) / tf.maximum(Hfi - 1.0, 1.0)
            ix = tf.cast(tf.range(tf.shape(mag_t)[2]), mag_t.dtype) / tf.maximum(W2fi - 1.0, 1.0)
            kk = tf.sqrt(iy[:, None] ** 2 + ix[None, :] ** 2) + 1e-6
            rolloff = 1.0 / (1.0 + 6.0 * kk)
            spec_err = tf.abs(mag_t - mag_p) * rolloff[None, :, :]
            spec_mean = tf.reduce_mean(spec_err, axis=[1,2])
            l_spec = spec_mean
        else:
            batch_sz = tf.shape(yt)[0]
            l_spec = tf.zeros((batch_sz,), dtype=yt.dtype)

        return tf.reduce_mean(l_mse + l_grad * l_grad_eff() + l_spec * l_spec_eff())

    # Use closure lambdas that just return 1.0 to avoid graph warnings on fixed floats in signature
    def l_grad_eff():
        return tf.cast(l_grad, tf.float32)
    def l_spec_eff():
        return tf.cast(l_spec, tf.float32)
    return loss

# ================== Model builders: 12-dim forcing -> (B,H,W,4 static concat at lowest-res -> upconv) ==================
def build_cnn_super(dim_x, n_lat, n_lon, hidden=48, static_ch=4):
    h1 = n_lat // 4
    w1 = n_lon // 4
    ch = 16
    inp = L.Input(shape=(dim_x,), name="forcing")
    # broadcastable static input
    static_inp = L.Input(shape=(n_lat, n_lon, static_ch), name="static_ch")
    # Downsample static 96x144 -> 24x36 to match lowest-res activation
    static_low = L.AveragePooling2D(pool_size=(4,4), padding="same")(static_inp)  # (B,24,36,4)

    h = L.Dense(h1 * w1 * ch, activation="silu", kernel_regularizer=keras.regularizers.L2(1e-5))(inp)
    h = L.Reshape((h1, w1, ch))(h)
    # concat static info at the lowest resolution (strong spatial inductive bias)
    h = L.Concatenate(axis=-1)([h, static_low])  # (B,24,36,16+4=20)
    h = L.Conv2DTranspose(hidden, 3, strides=2, padding="same", activation="silu")(h)  # (48,72)
    h = L.Conv2D(hidden, 3, padding="same", activation="silu")(h)
    h = L.Conv2DTranspose(hidden, 3, strides=2, padding="same", activation="silu")(h)  # (96,144)
    h = L.Concatenate(axis=-1)([h, static_inp])  # skip: inject full-res static into full-res feat (96,144, hidden+4)
    h = L.Conv2D(hidden, 3, padding="same", activation="silu")(h)
    h = L.Conv2D(16, 3, padding="same", activation="silu")(h)
    out = L.Conv2D(1, 1, padding="same")(h)
    out = L.Reshape((n_lat, n_lon))(out)
    model = keras.Model(inputs=[inp, static_inp], outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=make_super_loss(n_lat, n_lon))
    return model


def build_rnn_super(dim_x, n_lat, n_lon, lstm_hidden=48, seq_len=SEQ_LEN, static_ch=4):
    inp = L.Input(shape=(seq_len, dim_x), name="forcing_seq")
    static_inp = L.Input(shape=(n_lat, n_lon, static_ch), name="static_ch")
    static_low = L.AveragePooling2D(pool_size=(4,4), padding="same")(static_inp)

    h = L.Dense(128, activation="silu", kernel_regularizer=keras.regularizers.L2(1e-5))(inp)
    h = L.LSTM(lstm_hidden, return_sequences=True, activation="tanh", recurrent_activation="sigmoid",
               recurrent_regularizer=keras.regularizers.L2(1e-5))(h)
    h = L.LSTM(lstm_hidden, return_sequences=False, activation="tanh", recurrent_activation="sigmoid",
               recurrent_regularizer=keras.regularizers.L2(1e-5))(h)

    h1 = n_lat // 4
    w1 = n_lon // 4
    ch = 8
    h = L.Dense(h1 * w1 * ch, activation="silu")(h)
    h = L.Reshape((h1, w1, ch))(h)
    h = L.Concatenate(axis=-1)([h, static_low])
    h = L.Conv2DTranspose(32, 3, strides=2, padding="same", activation="silu")(h)
    h = L.Conv2D(32, 3, padding="same", activation="silu")(h)
    h = L.Conv2DTranspose(32, 3, strides=2, padding="same", activation="silu")(h)
    h = L.Concatenate(axis=-1)([h, static_inp])
    h = L.Conv2D(32, 3, padding="same", activation="silu")(h)
    out = L.Conv2D(1, 1, padding="same")(h)
    out = L.Reshape((n_lat, n_lon))(out)
    model = keras.Model(inputs=[inp, static_inp], outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=make_super_loss(n_lat, n_lon))
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
    rmse_map = get_rmse(truth_xa[35:], pred_np[35:])
    return float(np.mean(np.asarray(rmse_map).reshape(-1)))


rows = []
cnn_preds_super = {}
rnn_preds_super = {}
hist_cnn = {}
hist_rnn = {}

STATIC_BATCH_TR = np.tile(STATIC_4CH[None, ...], (X_train_arr.shape[0], 1, 1, 1))
STATIC_BATCH_TE = np.tile(STATIC_4CH[None, ...], (X_test_arr.shape[0], 1, 1, 1))

t_start = time.time()

for cfg in targets_cfg:
    target, unit, hp = cfg
    short = hp.get("short", target)
    LOG(f"\n==== Super target: {short} ({unit})")
    Y_train = np.asarray(Y_train_all[target].values, dtype=np.float32)
    Y_test = np.asarray(Y_test_all[target].values, dtype=np.float32)
    Y_test_xa = Y_test_all[target]
    N_LAT = Y_train.shape[1]
    N_LON = Y_train.shape[2]
    DIM_X = X_train_arr.shape[1]
    hidden = hp.get("hidden", 48)
    epochs = hp.get("epochs", N_EPOCHS)
    batch = hp.get("batch", 16)

    # CNN super
    LOG(f"[CNN-super] build")
    cnn = build_cnn_super(DIM_X, N_LAT, N_LON, hidden=hidden, static_ch=4)
    LOG(f"[CNN-super] params: {cnn.count_params():,}")
    hc = cnn.fit([X_train_arr, STATIC_BATCH_TR], Y_train,
                 epochs=epochs, batch_size=batch, validation_split=0.1, shuffle=True, verbose=2)
    pred_cnn = cnn.predict([X_test_arr, STATIC_BATCH_TE], verbose=0).astype(np.float32)
    # real train-set prediction for booster base_train (path A done!)
    pred_cnn_train = cnn.predict([X_train_arr, STATIC_BATCH_TR], verbose=0).astype(np.float32)
    np.save(str(OUT_DIR / f"super_basetrain_CNN_{short}.npy"), pred_cnn_train)
    cnn.save(str(OUT_DIR / f"super_weights_CNN_{short}.h5"))
    cnn_preds_super[target] = pred_cnn
    hist_cnn[short] = (hc.history["loss"], hc.history.get("val_loss", hc.history["loss"]))
    rmse_cnn_super = eval_rmse_late(Y_test_xa, pred_cnn)
    LOG(f"[CNN-super] {short} late RMSE = {rmse_cnn_super:.6f} {unit}  (saved train/base npy)")

    # RNN super
    LOG(f"[RNN-super] build seq")
    Xs_tr, Ys_tr = build_seq_samples(X_train_arr, Y_train, SEQ_LEN)
    Xs_te, Ys_te = build_seq_samples(X_test_arr, Y_test, SEQ_LEN)
    # Static batch sizes need to match seq-batched Xs_tr/Xs_te
    STATIC_TR_RNN = np.tile(STATIC_4CH[None, ...], (Xs_tr.shape[0], 1, 1, 1))
    STATIC_TE_RNN = np.tile(STATIC_4CH[None, ...], (Xs_te.shape[0], 1, 1, 1))
    rnn_batch = max(1, batch // 2)
    rnn = build_rnn_super(DIM_X, N_LAT, N_LON, seq_len=SEQ_LEN, static_ch=4)
    LOG(f"[RNN-super] params: {rnn.count_params():,}  (batch={rnn_batch})")
    hr = rnn.fit([Xs_tr, STATIC_TR_RNN], Ys_tr,
                 epochs=epochs, batch_size=rnn_batch, validation_split=0.1, shuffle=True, verbose=2)
    pred_rnn_seq = rnn.predict([Xs_te, STATIC_TE_RNN], verbose=0).astype(np.float32)
    pred_rnn = np.zeros_like(Y_test, dtype=np.float32)
    if SEQ_LEN - 1 > 0:
        pred_rnn[:SEQ_LEN - 1] = Y_test[:SEQ_LEN - 1]
    pred_rnn[SEQ_LEN - 1:] = pred_rnn_seq
    # train pred for RNN (need align: first SEQ_LEN-1 yrs filled with truth -> best we can do: placeholder)
    pred_rnn_train_seq = rnn.predict([Xs_tr, STATIC_TR_RNN], verbose=0).astype(np.float32)
    pred_rnn_train = np.zeros_like(Y_train, dtype=np.float32)
    pred_rnn_train[:SEQ_LEN - 1] = Y_train[:SEQ_LEN - 1]
    pred_rnn_train[SEQ_LEN - 1:] = pred_rnn_train_seq
    np.save(str(OUT_DIR / f"super_basetrain_RNN_{short}.npy"), pred_rnn_train)
    rnn.save(str(OUT_DIR / f"super_weights_RNN_{short}.h5"))
    rnn_preds_super[target] = pred_rnn
    hist_rnn[short] = (hr.history["loss"], hr.history.get("val_loss", hr.history["loss"]))
    rmse_rnn_super = eval_rmse_late(Y_test_xa, pred_rnn)
    LOG(f"[RNN-super] {short} late RMSE = {rmse_rnn_super:.6f} {unit}")

    # RF reference (use saved nc if exists; the compare/nn_booster have same ref)
    rf_nc = OUT_DIR / "outputs_ssp245_prediction_RF.nc"
    if rf_nc.exists():
        with xr.open_dataset(rf_nc) as dsr:
            if target in dsr.data_vars:
                rfp = np.asarray(dsr[target].values, dtype=np.float32)
                if rfp.ndim == 4:
                    rfp = rfp.reshape(rfp.shape[0], rfp.shape[-3], rfp.shape[-2])
                rfp = rfp[:len(Y_test)]
                rmse_rf = eval_rmse_late(Y_test_xa, rfp)
            else:
                rmse_rf = np.nan
    else:
        rmse_rf = np.nan

    LOG(f"[RF-ref] {short} late RMSE = {rmse_rf:.6f} {unit}")
    rows.append({
        "target": short, "unit": unit,
        "RMSE_RF": rmse_rf,
        "RMSE_CNN_super": rmse_cnn_super,
        "RMSE_RNN_super": rmse_rnn_super,
        "CNN_super_vs_RF": (rmse_rf - rmse_cnn_super),
        "RNN_super_vs_RF": (rmse_rf - rmse_rnn_super),
    })

# save netcdfs
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

cnn_super_nc = OUT_DIR / "outputs_ssp245_prediction_CNN_super.nc"
rnn_super_nc = OUT_DIR / "outputs_ssp245_prediction_RNN_super.nc"
to_ds(cnn_preds_super).to_netcdf(str(cnn_super_nc), "w")
to_ds(rnn_preds_super).to_netcdf(str(rnn_super_nc), "w")
LOG(f"saved CNN super: {cnn_super_nc}")
LOG(f"saved RNN super: {rnn_super_nc}")

df = pd.DataFrame(rows)
csv_path = OUT_DIR / "all_model_ssp245_super_scores.csv"
df.to_csv(csv_path, index=False)
LOG(f"saved super score csv: {csv_path}")
with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", lambda x: f"{x:.5f}"):
    LOG("\n" + df.to_string(index=False))

# training curves figure
fig, axes = plt.subplots(len(targets_cfg), 2, figsize=(13, 3.2 * len(targets_cfg)), sharex=True)
if len(targets_cfg) == 1: axes = axes[None, :]
for i, cfg in enumerate(targets_cfg):
    target, unit, hp = cfg
    short = hp.get("short", target)
    for j, (hist, title) in enumerate([(hist_cnn[short], f"CNN-super {short}"), (hist_rnn[short], f"RNN-super {short}")]):
        ax = axes[i, j]
        tr_l, va_l = hist
        xs = np.arange(1, len(tr_l) + 1)
        ax.plot(xs, tr_l, label="train (super loss)")
        ax.plot(xs, va_l, label="val (super loss)", ls="--")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
plt.tight_layout()
fig_path = FIG_DIR / "week4_super_cnn_rnn_training_curves.png"
fig.savefig(fig_path, dpi=130)
plt.close(fig)
LOG(f"saved training curves: {fig_path}")

LOG(f"Total elapsed: {time.time() - t_start:.1f}s")

# Dump log file
log_path = OUT_DIR / "week4_super_models.log"
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(LOGS) + "\n")
LOG(f"saved log: {log_path}")
LOG("=== week4 super model done ===")
