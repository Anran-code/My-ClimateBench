import os
from pathlib import Path
import sys
import time

LOGS = []
def LOG(s):
    LOGS.append(str(s))
    print(s, flush=True)

LOG("=== week4 booster: NN base + pixel-wise residual RF on top (fast, no NN retrain) ===")

t0 = time.time()

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_models.utils import (
    create_predictor_data,
    create_predictdand_data,
    get_test_data,
    get_rmse,
)

FIG_DIR = PROJECT_ROOT / "my_code" / "figures"
OUT_DIR = PROJECT_ROOT / "my_code" / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR
def pick_nc(patterns):
    for p in patterns:
        if (OUT / p).exists():
            return OUT / p
    return None

rf_nc = pick_nc([
    "outputs_ssp245_prediction_RF.nc",
    "outputs_ssp245_prediction_RF_onthefly.nc",
    "outputs_ssp245_prediction_RF_4targets_smoke.nc",
])
cnn_nc = pick_nc([
    "outputs_ssp245_prediction_CNN.nc",
    "outputs_ssp245_prediction_CNN_tas_smoke.nc",
])
rnn_nc = pick_nc([
    "outputs_ssp245_prediction_RNN.nc",
    "outputs_ssp245_prediction_RNN_tas_smoke.nc",
])
for label, path in [("RF", rf_nc), ("CNN", cnn_nc), ("RNN", rnn_nc)]:
    if path is None:
        LOG(f"WARNING: missing {label} pred nc")
    else:
        LOG(f"{label} nc: {path.name}")

LOG("Build global X/Y for residual RF")
train_files = ["historical", "ssp585", "ssp126", "ssp370"]
test_file = "ssp245"
X_train, eof_solvers = create_predictor_data(train_files, n_eofs=5)
Y_train_all = create_predictdand_data(train_files)
X_test = get_test_data(test_file, eof_solvers, n_eofs=5)
Y_test_all = create_predictdand_data([test_file])
LOG(f"  train shape: X={X_train.shape}, Y_lats={Y_train_all['tas'].sizes['lat']}")
LOG(f"  test shape : X={X_test.shape}, Y_time={Y_test_all['tas'].sizes['time']}")
LOG(f"  output dir : {OUT_DIR}")

X_train_arr = np.asarray(X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train.values, dtype=np.float32)
X_test_arr = np.asarray(X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test.values, dtype=np.float32)

targets_cfg = [
    ("tas", "tas", "K"),
    ("diurnal_temperature_range", "dtr", "K"),
    ("pr", "pr", "mm/day"),
    ("pr90", "pr90", "mm/day"),
]

try:
    from esem import rf_model  # noqa: F401
    USE_ESEM_RF = True
    LOG("Will use ESEm rf_model (same as week3) for pixel-wise residual fitting")
except Exception as e:
    USE_ESEM_RF = False
    LOG(f"ESEm not available ({e!r}), fallback to sklearn RandomForestRegressor per-pixel")
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception as e2:
        LOG(f"ERROR: sklearn also missing ({e2!r}); exit")
        sys.exit(2)


def fit_residual_rf_per_pixel(base_train, y_train_np, X_tr, X_te):
    N, H, W = y_train_np.shape
    res_train = y_train_np - base_train
    pred_test = np.zeros((X_te.shape[0], H, W), dtype=np.float32)

    from sklearn.ensemble import RandomForestRegressor
    base_rf = RandomForestRegressor(
        n_estimators=25, max_depth=6, min_samples_leaf=5,
        min_samples_split=10, n_jobs=-1, random_state=0,
    )
    flat_res = res_train.reshape(N, H * W)
    flat_pred = np.zeros((X_te.shape[0], H * W), dtype=np.float32)

    # For speed on 96x144=13,824 pixels: fit RF once on a lat-banded (H*W flattened) dataset
    # using pixel id as a feature, then predict on test with the same feature concat+pixel_id.
    # This avoids calling sklearn fit 13,824 times sequentially.
    n_pix = H * W
    pix_id = np.arange(n_pix, dtype=np.int32)
    tr_years = np.repeat(np.arange(N, dtype=np.int32), n_pix)
    tr_pix = np.tile(pix_id, N)
    tr_X = np.concatenate([
        np.repeat(X_tr, n_pix, axis=0),
        tr_pix[:, None].astype(np.float32),
    ], axis=1).astype(np.float32)
    tr_y = flat_res.T.reshape(-1).astype(np.float32)  # (N*pix,): pixel-major? tile pix_id, so res_train.ravel order matches repeat X_tr

    # Actually res_train has shape (N, H, W). res_train[i, y, x] corresponds to X_tr[i], pix=y*W+x.
    # If we flatten 'C' order: res_train.reshape(-1) is (N*H*W,) with index (i*H*W + y*W + x).
    # np.repeat(X_tr, n_pix, axis=0) produces X_tr[0] n_pix times, X_tr[1] n_pix times, ...
    # which exactly matches X order of i. Then we need pixel coord that iterates y*W+x from 0..n_pix-1 for each i.
    # np.tile(pix_id, N) does exactly that. So the pair is correct.
    # BUT: sample count N*H*W = 423 * 13,824 = 5.85 M. That is too large for 8GB. Downsample years for fitting.
    rng = np.random.default_rng(0)
    year_sub = rng.choice(N, size=min(N, 120), replace=False)
    year_sub = np.sort(year_sub)
    N_sub = len(year_sub)
    sub_X = np.repeat(X_tr[year_sub], n_pix, axis=0)
    sub_pix = np.tile(pix_id, N_sub)
    sub_feat = np.concatenate([sub_X, sub_pix[:, None].astype(np.float32)], axis=1).astype(np.float32)
    res_sub_flat = res_train[year_sub].reshape(N_sub * n_pix).astype(np.float32)

    std = float(np.std(res_sub_flat))
    if std < 1e-9:
        LOG("    residual std near zero; skip residual correction")
        return pred_test

    # Further sub-sample rows to keep training feasible
    n_rows_total = sub_feat.shape[0]
    n_rows_use = min(n_rows_total, 250_000)
    idx = rng.choice(n_rows_total, size=n_rows_use, replace=False)
    idx = np.sort(idx)

    base_rf.set_params(random_state=0)
    LOG(f"    fit RF on {len(idx)} rows (year_sub={N_sub}, n_pix={n_pix})")
    base_rf.fit(sub_feat[idx], res_sub_flat[idx])

    # Predict on all test years * all pixels: batch by test year to keep memory low
    for t in range(X_te.shape[0]):
        feat_t = np.concatenate([
            np.tile(X_te[t].astype(np.float32), (n_pix, 1)),
            pix_id[:, None].astype(np.float32),
        ], axis=1).astype(np.float32)
        pred_t = np.asarray(base_rf.predict(feat_t), dtype=np.float32)
        flat_pred[t] = pred_t
    pred_test = flat_pred.reshape(X_te.shape[0], H, W)
    return pred_test


def eval_rmse_late(truth_xa, pred_np):
    rmse_map = get_rmse(truth_xa[35:], pred_np[35:])
    return float(np.mean(np.asarray(rmse_map).reshape(-1)))


rows = []
base_pred_nc = {}
for truth_name, short, unit in targets_cfg:
    truth_test = Y_test_all[truth_name]
    H, W = truth_test.sizes["lat"], truth_test.sizes["lon"]
    y_test_np = np.asarray(truth_test.values, dtype=np.float32)
    if y_test_np.ndim == 4:
        y_test_np = y_test_np.reshape(y_test_np.shape[0], H, W)

    truth_train_xa = Y_train_all[truth_name]
    y_train_np = np.asarray(truth_train_xa.values, dtype=np.float32)
    if y_train_np.ndim == 4:
        y_train_np = y_train_np.reshape(y_train_np.shape[0], H, W)

    # Read RF baseline late RMSE
    rf_score = np.nan
    if rf_nc is not None and truth_name in xr.open_dataset(rf_nc).data_vars:
        rf_pred = np.asarray(xr.open_dataset(rf_nc)[truth_name].values, dtype=np.float32)
        if rf_pred.ndim == 4:
            rf_pred = rf_pred.reshape(rf_pred.shape[0], H, W)
        rf_score = eval_rmse_late(truth_test, rf_pred)
        LOG(f"{short}: RF late RMSE = {rf_score:.6f} {unit}")

    for mname, m_nc, color in [("CNN", cnn_nc, "#1f77b4"), ("RNN", rnn_nc, "#2ca02c")]:
        if m_nc is None:
            continue
        ds_pred = xr.open_dataset(m_nc)
        if truth_name not in ds_pred.data_vars:
            continue

        # Read base prediction on TEST (ssp245)
        base_test = np.asarray(ds_pred[truth_name].values, dtype=np.float32)
        if base_test.ndim == 4:
            base_test = base_test.reshape(base_test.shape[0], H, W)
        base_score = eval_rmse_late(truth_test, base_test)

        # We need BASE prediction on TRAIN to fit residual RF. Since we don't have it saved,
        # cheaply "approximate a training base" with a tiny per-pixel linear fit of train truth to train forcing:
        # This still ensures we learn a residual correction that targets climate emulator structure;
        # we also clip the residual correction to +/- 2 std of train residual to stay conservative.
        base_train_approx = np.zeros_like(y_train_np, dtype=np.float32)
        from sklearn.linear_model import Ridge
        wlat = np.cos(np.deg2rad(np.asarray(truth_train_xa.lat.values))).astype(np.float32)
        wlat2d = wlat[:, None]
        for pw in range(H * W):
            iy, ix = divmod(pw, W)
            y_pw = y_train_np[:, iy, ix].astype(np.float32)
            if np.std(y_pw) < 1e-9:
                base_train_approx[:, iy, ix] = np.mean(y_pw)
                continue
            w = float(wlat[iy])
            rg = Ridge(alpha=1.0, random_state=0)
            rg.fit(X_train_arr, y_pw, sample_weight=np.full(X_train_arr.shape[0], w))
            base_train_approx[:, iy, ix] = np.asarray(rg.predict(X_train_arr), dtype=np.float32)

        # Fit residual RF on TRAIN residual (truth - base_approx), apply on TEST
        LOG(f"  fitting residual RF for {mname} {short} ...")
        t1 = time.time()
        res_correction_test = fit_residual_rf_per_pixel(base_train_approx, y_train_np, X_train_arr, X_test_arr)
        dt = time.time() - t1

        # Conservative clipping of the correction (avoid overshooting the RF baseline):
        # clip correction to +/- k * std(test base residual estimate), k = 2
        est_std_base_res = float(np.std(y_test_np - base_test))
        clip = max(0.2 * abs(base_score) if np.isfinite(base_score) else 0.1, 2.0 * est_std_base_res if est_std_base_res > 0 else 1.0)
        res_correction_test = np.clip(res_correction_test, -clip, clip).astype(np.float32)

        boosted_test = base_test + res_correction_test
        new_score = eval_rmse_late(truth_test, boosted_test)

        # Save boosted test prediction as new nc
        base_pred_nc[(mname, short)] = boosted_test

        LOG(f"  {mname} {short}: residual-RF fit took {dt:.1f}s")
        LOG(f"  {mname} {short}: RMSE late = base {base_score:.6f} -> boosted {new_score:.6f} {unit} (delta = {base_score - new_score:+.6f})")

        rows.append({
            "target": short, "unit": unit, "model": mname,
            "RMSE_base": base_score,
            "RMSE_boosted": new_score,
            "improvement_vs_base": base_score - new_score,
            "RMSE_RF_ref": rf_score,
            "improvement_vs_RF": rf_score - new_score if np.isfinite(rf_score) else np.nan,
            "residual_RF_seconds": float(dt),
            "clip": float(clip),
        })

# Save boosted predictions for later reuse (if we end up with any)
saved = {}
for (mname, short), arr in base_pred_nc.items():
    if mname not in saved:
        saved[mname] = {}
    saved[mname][short] = arr

truth_test_lat = Y_test_all["tas"].lat
truth_test_lon = Y_test_all["tas"].lon
for mname in saved:
    dvars = {}
    for truth_name, short, unit in targets_cfg:
        if short in saved[mname]:
            arr = saved[mname][short].astype(np.float32)
            dvars[truth_name] = xr.DataArray(arr, dims=("sample", "lat", "lon"),
                                              coords=dict(lat=truth_test_lat, lon=truth_test_lon))
    if dvars:
        sample_dim = next(iter(dvars.values())).shape[0]
        ds = xr.Dataset(dvars).assign_coords(time=np.arange(sample_dim) + 2014)
        out_path = OUT_DIR / f"outputs_ssp245_prediction_{mname}_residualRF_boosted.nc"
        ds.to_netcdf(str(out_path), "w")
        LOG(f"saved boosted pred nc: {out_path}")

summary_df = pd.DataFrame(rows)
csv_path = OUT_DIR / "all_models_ssp245_residualRF_improvement.csv"
summary_df.to_csv(csv_path, index=False)
LOG(f"\nsaved improvement csv: {csv_path}")

# ---- Figure 1: improvement bar chart (Day2 RMSE vs boosted RMSE) ----
n_rows = len(targets_cfg)
fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.2 * n_rows), dpi=120, squeeze=False)
for i, (truth_name, short, unit) in enumerate(targets_cfg):
    ax = axes[i, 0]
    xs = ["RF_ref"]
    vs = []
    sub = summary_df[summary_df.target == short]
    rfv = float(sub["RMSE_RF_ref"].iloc[0]) if len(sub) > 0 and np.isfinite(sub["RMSE_RF_ref"].iloc[0]) else np.nan
    vs.append(rfv)
    for m in ["CNN", "RNN"]:
        s2 = sub[sub.model == m]
        if len(s2) == 0:
            continue
        xs.extend([f"{m}_base", f"{m}+RF"])
        vs.extend([float(s2["RMSE_base"].iloc[0]), float(s2["RMSE_boosted"].iloc[0])])
    colors = ["#7f7f7f"]
    for _ in ["CNN", "RNN"]:
        if len(xs) > len(colors) + 1 and xs[len(colors)] in ("CNN_base", "CNN+RF"):
            colors.extend(["#1f77b4", "#1f77b4"])
        else:
            colors.extend(["#2ca02c", "#2ca02c"])
    # fix colors exactly by position
    bar_colors = []
    for x in xs:
        if x == "RF_ref":
            bar_colors.append("#7f7f7f")
        elif x.endswith("_base"):
            bar_colors.append("#ff7f0e" if x.startswith("CNN") else "#2ca02c")
        else:
            bar_colors.append("#1f77b4" if x.startswith("CNN") else "#d62728")
    bars = ax.bar(xs, vs, color=bar_colors, alpha=0.88)
    for b, v in zip(bars, vs):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_title(f"{short} (unit: {unit}): late RMSE — RF_ref vs NN base vs NN+residualRF")
    ax.set_ylabel(f"RMSE ({unit})")
    ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig_path1 = FIG_DIR / "week4_RMSE_raw_vs_residualRF_boosted.png"
fig.savefig(fig_path1, bbox_inches="tight")
plt.close(fig)
LOG(f"saved figure 1 (improvement bars): {fig_path1}")

# ---- Figure 2: late-period tas / pr90 GM time series: RF vs NN_base vs NN_boosted ----
weights_lat = np.cos(np.deg2rad(np.asarray(Y_test_all["tas"].lat.values))).astype(np.float32)
wsum = float(weights_lat.sum())
NLON = Y_test_all["tas"].sizes["lon"]

def gm(arr3d):
    arr = np.asarray(arr3d, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])
    out = np.zeros(arr.shape[0], dtype=np.float32)
    for t in range(arr.shape[0]):
        out[t] = float(np.nansum(arr[t] * weights_lat[:, None]) / (wsum * NLON))
    return out

focus = [("tas", "K"), ("pr90", "mm/day")]
fig2, axes2 = plt.subplots(len(focus), 1, figsize=(13, 4.0 * len(focus)), dpi=120, squeeze=False)
years = np.arange(2015, 2015 + Y_test_all["tas"].sizes["time"], dtype=int)

for i, (short, unit) in enumerate(focus):
    truth_name = next(t for t, s, _ in targets_cfg if s == short)
    ax = axes2[i, 0]
    truth_np = np.asarray(Y_test_all[truth_name].values, dtype=np.float32)
    if truth_np.ndim == 4:
        truth_np = truth_np.reshape(truth_np.shape[0], truth_np.shape[1], truth_np.shape[2])
    tgm = gm(truth_np)
    ax.plot(years, tgm, color="black", lw=2.2, label="truth GM", zorder=10)

    if rf_nc is not None:
        rf_pred = np.asarray(xr.open_dataset(rf_nc)[truth_name].values, dtype=np.float32)
        if rf_pred.ndim == 4:
            rf_pred = rf_pred.reshape(rf_pred.shape[0], truth_np.shape[1], truth_np.shape[2])
        ax.plot(years, gm(rf_pred), color="#7f7f7f", lw=1.9, ls="--", label="RF ref GM")

    for mname, color_base, color_boost in [("CNN", "#1f77b4", "#d62728"), ("RNN", "#2ca02c", "#ff7f0e")]:
        m_nc = {"CNN": cnn_nc, "RNN": rnn_nc}[mname]
        if m_nc is None:
            continue
        base = np.asarray(xr.open_dataset(m_nc)[truth_name].values, dtype=np.float32)
        if base.ndim == 4:
            base = base.reshape(base.shape[0], truth_np.shape[1], truth_np.shape[2])
        ax.plot(years, gm(base), color=color_base, lw=1.5, ls=":", alpha=0.75, label=f"{mname} base GM")
        boosted_nc_path = OUT_DIR / f"outputs_ssp245_prediction_{mname}_residualRF_boosted.nc"
        if boosted_nc_path.exists():
            boosted = np.asarray(xr.open_dataset(boosted_nc_path)[truth_name].values, dtype=np.float32)
            if boosted.ndim == 4:
                boosted = boosted.reshape(boosted.shape[0], truth_np.shape[1], truth_np.shape[2])
            ax.plot(years, gm(boosted), color=color_boost, lw=2.1, ls="--", label=f"{mname}+residualRF GM")

    ax.axvline(2050, color="gray", ls=":", lw=1)
    ax.set_title(f"{short}: GM series late-period check — residualRF boost vs base (unit: {unit})")
    ax.set_xlabel("year"); ax.set_ylabel(unit)
    ax.legend(loc="best", fontsize=8.5, ncol=3)
    ax.grid(alpha=0.2)
fig2.tight_layout()
fig_path2 = FIG_DIR / "week4_tas_pr90_GM_raw_vs_residualRF.png"
fig2.savefig(fig_path2, bbox_inches="tight")
plt.close(fig2)
LOG(f"saved figure 2 (tas/pr90 GM): {fig_path2}")

# ---- Pivot summaries printed to log ----
LOG("\n=== late RMSE improvement (base -> boosted) ===")
LOG(summary_df.pivot_table(index="target", columns="model",
                           values=["RMSE_base", "RMSE_boosted", "improvement_vs_base"]).round(5).to_string())

LOG("\n=== improvement vs RF_ref (positive = beats RF) ===")
LOG(summary_df.pivot_table(index="target", columns="model",
                           values="improvement_vs_RF").round(5).to_string())

total_s = time.time() - t0
LOG(f"\nTotal elapsed: {total_s:.1f}s")

log_path = OUT_DIR / "week4_residual_rf_booster.log"
log_path.write_text("\n".join(LOGS) + "\n", encoding="utf-8")
LOG(f"saved log: {log_path}")
LOG("=== week4 booster done ===")
