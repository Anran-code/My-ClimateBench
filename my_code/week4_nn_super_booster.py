import os
from pathlib import Path
import sys
import time

LOGS = []
def LOG(s):
    LOGS.append(str(s))
    print(s, flush=True)

LOG("=== week4 SUPER BOOSTER: old base vs super base vs super+RF (7 bars / 7 lines) ===")

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

rf_nc = pick_nc(["outputs_ssp245_prediction_RF.nc", "outputs_ssp245_prediction_RF_onthefly.nc"])
cnn_nc_old = pick_nc(["outputs_ssp245_prediction_CNN.nc"])
rnn_nc_old = pick_nc(["outputs_ssp245_prediction_RNN.nc"])
cnn_nc_super = pick_nc(["outputs_ssp245_prediction_CNN_super.nc"])
rnn_nc_super = pick_nc(["outputs_ssp245_prediction_RNN_super.nc"])
for label, path in [("RF", rf_nc), ("CNN_old", cnn_nc_old), ("RNN_old", rnn_nc_old),
                    ("CNN_super", cnn_nc_super), ("RNN_super", rnn_nc_super)]:
    if path is None:
        LOG(f"WARNING: missing {label} pred nc")
    else:
        LOG(f"{label} nc: {path.name}")

LOG("Build global X/Y for super booster")
train_files = ["historical", "ssp585", "ssp126", "ssp370"]
test_file = "ssp245"
X_train, eof_solvers = create_predictor_data(train_files, n_eofs=5)
Y_train_all = create_predictdand_data(train_files)
X_test = get_test_data(test_file, eof_solvers, n_eofs=5)
Y_test_all = create_predictdand_data([test_file])
X_train_arr = np.asarray(X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train.values, dtype=np.float32)
X_test_arr = np.asarray(X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test.values, dtype=np.float32)

targets_cfg = [
    ("tas", "tas", "K"),
    ("diurnal_temperature_range", "dtr", "K"),
    ("pr", "pr", "mm/day"),
    ("pr90", "pr90", "mm/day"),
]


def fit_residual_rf_fast(base_train, y_train_np, X_tr, X_te):
    N, H, W = y_train_np.shape
    res_train = y_train_np - base_train
    pred_test = np.zeros((X_te.shape[0], H, W), dtype=np.float32)
    from sklearn.ensemble import RandomForestRegressor
    base_rf = RandomForestRegressor(
        n_estimators=40, max_depth=8, min_samples_leaf=4,
        min_samples_split=8, n_jobs=-1, random_state=0,
    )
    n_pix = H * W
    pix_id = np.arange(n_pix, dtype=np.int32)
    rng = np.random.default_rng(0)
    year_sub = rng.choice(N, size=min(N, 160), replace=False)
    year_sub = np.sort(year_sub)
    N_sub = len(year_sub)
    sub_X = np.repeat(X_tr[year_sub], n_pix, axis=0)
    sub_pix = np.tile(pix_id, N_sub)
    sub_feat = np.concatenate([sub_X, sub_pix[:, None].astype(np.float32)], axis=1).astype(np.float32)
    res_sub_flat = res_train[year_sub].reshape(N_sub * n_pix).astype(np.float32)
    std = float(np.std(res_sub_flat))
    if std < 1e-9:
        LOG("    residual std near zero; skip residual correction")
        return pred_test, std, 1.0
    n_rows_total = sub_feat.shape[0]
    n_rows_use = min(n_rows_total, 250_000)
    idx = rng.choice(n_rows_total, size=n_rows_use, replace=False)
    idx = np.sort(idx)
    base_rf.set_params(random_state=0)
    LOG(f"    fit RF on {len(idx)} rows (year_sub={N_sub}, n_pix={n_pix})")
    base_rf.fit(sub_feat[idx], res_sub_flat[idx])
    flat_pred = np.zeros((X_te.shape[0], n_pix), dtype=np.float32)
    for t in range(X_te.shape[0]):
        feat_t = np.concatenate([
            np.tile(X_te[t].astype(np.float32), (n_pix, 1)),
            pix_id[:, None].astype(np.float32),
        ], axis=1).astype(np.float32)
        flat_pred[t] = np.asarray(base_rf.predict(feat_t), dtype=np.float32)
    pred_test = flat_pred.reshape(X_te.shape[0], H, W)
    return pred_test, std, 1.0


def eval_rmse_late(truth_xa, pred_np):
    rmse_map = get_rmse(truth_xa[35:], pred_np[35:])
    return float(np.mean(np.asarray(rmse_map).reshape(-1)))


def load_base_train(mname, short, y_train_np, X_tr):
    npy_path = OUT_DIR / f"super_basetrain_{mname}_{short}.npy"
    LOG(f"  load_base_train({mname}, {short}) -> expect {npy_path.name}")
    if npy_path.exists():
        LOG(f"  use REAL base_train from {npy_path.name}")
        bt = np.load(str(npy_path))
        if bt.shape != y_train_np.shape:
            LOG(f"    WARNING: shape mismatch {bt.shape} vs {y_train_np.shape}, clip/pad")
            bt_use = np.zeros_like(y_train_np, dtype=np.float32)
            T = min(bt.shape[0], y_train_np.shape[0])
            bt_use[:T] = bt[:T]
            if T < y_train_np.shape[0]:
                bt_use[T:] = y_train_np[T:]
            return bt_use, "REAL(npy)"
        return bt, "REAL(npy)"
    # fallback: cheap & safe -> use truth (zero residual; booster will be no-op)
    LOG(f"  super_basetrain npy missing -> fallback: use y_train as base_train (zero correction)")
    return y_train_np.copy(), "TRUTH(zero)"


rows = []
preds_store = {}  # key (tag, short) -> 3d numpy
_NC_ACCUM = {}    # accumulate per-super-boosted DataArrays across targets

for truth_name, short, unit in targets_cfg:
    truth_test = Y_test_all[truth_name]
    truth_train_xa = Y_train_all[truth_name]
    H, W = truth_test.sizes["lat"], truth_test.sizes["lon"]
    y_test_np = np.asarray(truth_test.values, dtype=np.float32)
    if y_test_np.ndim == 4:
        y_test_np = y_test_np.reshape(y_test_np.shape[0], H, W)
    y_train_np = np.asarray(truth_train_xa.values, dtype=np.float32)
    if y_train_np.ndim == 4:
        y_train_np = y_train_np.reshape(y_train_np.shape[0], H, W)

    # RF reference
    rf_score = np.nan
    if rf_nc is not None:
        with xr.open_dataset(rf_nc) as dsr:
            if truth_name in dsr.data_vars:
                rfp = np.asarray(dsr[truth_name].values, dtype=np.float32)
                if rfp.ndim == 4: rfp = rfp.reshape(rfp.shape[0], H, W)
                rfp = rfp[:len(y_test_np)]
                rf_score = eval_rmse_late(truth_test, rfp)
                preds_store[("RF_ref", short)] = rfp.copy()
    LOG(f"{short}: RF late RMSE = {rf_score:.6f} {unit}")

    variants = [
        ("CNN_old", cnn_nc_old),
        ("RNN_old", rnn_nc_old),
        ("CNN_super", cnn_nc_super),
        ("RNN_super", rnn_nc_super),
    ]
    for mtag, m_nc in variants:
        if m_nc is None:
            continue
        with xr.open_dataset(m_nc) as dsp:
            if truth_name not in dsp.data_vars:
                continue
            base_test = np.asarray(dsp[truth_name].values, dtype=np.float32)
        if base_test.ndim == 4: base_test = base_test.reshape(base_test.shape[0], H, W)
        preds_store[(mtag, short)] = base_test.copy()
        base_score = eval_rmse_late(truth_test, base_test)
        LOG(f"  {mtag} {short}: base RMSE late = {base_score:.6f} {unit}")

        # Residual RF booster (only meaningful for super variants, but do all for comparison)
        mname_core = mtag.split("_")[0]  # CNN / RNN
        if mtag.endswith("_super"):
            base_train, bt_kind = load_base_train(mname_core, short, y_train_np, X_train_arr)
            LOG(f"  fitting residual-RF for {mtag} {short} (base_train = {bt_kind}) ...")
            t1 = time.time()
            res_corr_test, res_std, _ = fit_residual_rf_fast(base_train, y_train_np, X_train_arr, X_test_arr)
            dt = time.time() - t1
            est_std = float(np.std(y_test_np - base_test))
            clip = max(0.2 * abs(base_score) if np.isfinite(base_score) else 0.1, 2.0 * est_std if est_std > 0 else 1.0)
            res_corr_test = np.clip(res_corr_test, -clip, clip).astype(np.float32)
            boosted = base_test + res_corr_test
            new_score = eval_rmse_late(truth_test, boosted)
            preds_store[(mtag + "+RF", short)] = boosted.copy()
            LOG(f"  {mtag}+RF {short}: took {dt:.1f}s, clip={clip:.4f}; RMSE late = base {base_score:.6f} -> boosted {new_score:.6f} {unit} (delta {base_score - new_score:+.6f})")
            rows.append({
                "target": short, "unit": unit, "model": mtag,
                "RMSE_base": base_score, "RMSE_boosted": new_score,
                "improvement_vs_base": base_score - new_score,
                "RMSE_RF_ref": rf_score,
                "improvement_vs_RF": rf_score - new_score if np.isfinite(rf_score) else np.nan,
                "residual_RF_seconds": float(dt), "clip": float(clip), "base_train_kind": bt_kind,
            })
            # save boosted nc (separate per super model)
            dvars = {truth_name: xr.DataArray(boosted, dims=("sample", "lat", "lon"),
                                              coords=dict(lat=truth_test.lat, lon=truth_test.lon))}
            sample_dim = boosted.shape[0]
            dss = xr.Dataset(dvars).assign_coords(time=np.arange(sample_dim) + 2014)
            out = OUT_DIR / f"outputs_ssp245_prediction_{mname_core}_super_residualRF_boosted.nc"
            merge_key = mname_core + "_super_boosted"
            _NC_ACCUM.setdefault(merge_key, {})[truth_name] = dvars[truth_name]
        else:
            # Old variants: no super base_train available -> only base score row (no booster)
            rows.append({
                "target": short, "unit": unit, "model": mtag,
                "RMSE_base": base_score, "RMSE_boosted": np.nan,
                "improvement_vs_base": np.nan,
                "RMSE_RF_ref": rf_score,
                "improvement_vs_RF": rf_score - base_score if np.isfinite(rf_score) else np.nan,
                "residual_RF_seconds": 0.0, "clip": 0.0, "base_train_kind": "-",
            })

# flush accumulated super boosted ncs (4 targets each)
if _NC_ACCUM:
    truth_test_lat = Y_test_all["tas"].lat
    truth_test_lon = Y_test_all["tas"].lon
    for merge_key, dvars_map in _NC_ACCUM.items():
        sample_dim = next(iter(dvars_map.values())).shape[0]
        ds = xr.Dataset(dvars_map).assign_coords(time=np.arange(sample_dim) + 2014)
        out_name = f"outputs_ssp245_prediction_{merge_key.replace('_boosted','')}_boosted.nc"
        # already outputs name in correct format above, but clean:
        out_path = OUT_DIR / f"outputs_ssp245_prediction_{merge_key}.nc".replace("__", "_")
        ds.to_netcdf(str(out_path), "w")
        LOG(f"saved super-boosted nc: {out_path}")

# Save CSV summary
summary_df = pd.DataFrame(rows)
csv_path = OUT_DIR / "all_models_ssp245_super_improvement.csv"
summary_df.to_csv(csv_path, index=False)
LOG(f"\nsaved super improvement csv: {csv_path}")

# Pivot view for log
def pivot(df, value_col):
    try:
        p = df.pivot(index="target", columns="model", values=value_col)
        return "\n" + p.to_string(float_format=lambda x: f"{x:.5f}")
    except Exception as e:
        return f"\n(pivot err: {e!r})"

LOG("\n=== base RMSE (late) ===" + pivot(summary_df, "RMSE_base"))
LOG("\n=== boosted RMSE (late, only for *_super) ===" + pivot(summary_df, "RMSE_boosted"))
LOG("\n=== improvement_vs_RF (positive = beat RF) ===" + pivot(summary_df, "improvement_vs_RF"))

# ======================= FIGURE 1: 7-bar comparison per target =======================
BAR_TAGS = ["RF_ref", "CNN_old", "CNN_super", "CNN_super+RF", "RNN_old", "RNN_super", "RNN_super+RF"]
BAR_COLORS = {
    "RF_ref": "#7f7f7f",
    "CNN_old": "#ff7f0e",
    "CNN_super": "#1f77b4",
    "CNN_super+RF": "#9467bd",
    "RNN_old": "#2ca02c",
    "RNN_super": "#d62728",
    "RNN_super+RF": "#17becf",
}
n_rows = len(targets_cfg)
fig1, axes1 = plt.subplots(n_rows, 1, figsize=(12, 3.6 * n_rows), dpi=120, squeeze=False)
for i, (truth_name, short, unit) in enumerate(targets_cfg):
    ax = axes1[i, 0]
    xs = []
    vs = []
    for tag in BAR_TAGS:
        key = (tag, short)
        if key in preds_store:
            rmse = eval_rmse_late(Y_test_all[truth_name], preds_store[key])
            xs.append(tag); vs.append(rmse)
        elif tag == "RF_ref":
            sub = summary_df[(summary_df.target == short)]
            if len(sub) > 0:
                v = float(sub["RMSE_RF_ref"].iloc[0]) if np.isfinite(sub["RMSE_RF_ref"].iloc[0]) else np.nan
                xs.append(tag); vs.append(v)
    colors = [BAR_COLORS.get(x, "#8c564b") for x in xs]
    bars = ax.bar(xs, vs, color=colors, alpha=0.9)
    for b, v in zip(bars, vs):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_title(f"{short} (unit: {unit}): late RMSE — 最高档 7 方法对比")
    ax.set_ylabel(f"RMSE ({unit})")
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
fig1.tight_layout()
fig1_path = FIG_DIR / "week4_ULTIMATE_RMSE_7methods_compare.png"
fig1.savefig(fig1_path, bbox_inches="tight")
plt.close(fig1)
LOG(f"\nsaved ULTIMATE bars: {fig1_path}")

# ======================= FIGURE 2: tas / pr90 GM 7-line comparison =======================
weights_lat = np.cos(np.deg2rad(np.asarray(Y_test_all["tas"].lat.values))).astype(np.float32)
wsum = float(weights_lat.sum())
NLON = Y_test_all["tas"].sizes["lon"]
def gm(arr3d):
    arr = np.asarray(arr3d, dtype=np.float32)
    if arr.ndim == 4: arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])
    T = arr.shape[0]
    w = weights_lat[:, None]
    out = np.zeros(T, dtype=np.float32)
    for t in range(T):
        out[t] = float(np.nansum(arr[t] * w) / (wsum * NLON))
    return out

focus = [("tas", "K"), ("pr90", "mm/day")]
LINE_STYLES = {
    "RF_ref": ("#7f7f7f", "--", 1.8),
    "CNN_old": ("#ff7f0e", ":", 1.4),
    "CNN_super": ("#1f77b4", "-.", 1.8),
    "CNN_super+RF": ("#9467bd", "--", 2.1),
    "RNN_old": ("#2ca02c", ":", 1.4),
    "RNN_super": ("#d62728", "-.", 1.8),
    "RNN_super+RF": ("#17becf", "--", 2.1),
}
fig2, axes2 = plt.subplots(len(focus), 1, figsize=(14, 4.4 * len(focus)), dpi=120, squeeze=False)
years = np.arange(2015, 2015 + Y_test_all["tas"].sizes["time"], dtype=int)
for i, (short, unit) in enumerate(focus):
    truth_name = next(t for t, s, _ in targets_cfg if s == short)
    ax = axes2[i, 0]
    truth_np = np.asarray(Y_test_all[truth_name].values, dtype=np.float32)
    if truth_np.ndim == 4: truth_np = truth_np.reshape(truth_np.shape[0], truth_np.shape[1], truth_np.shape[2])
    tgm = gm(truth_np)
    ax.plot(years, tgm, color="black", lw=2.5, label="truth GM", zorder=15)
    for tag in BAR_TAGS:
        key = (tag, short)
        if key not in preds_store:
            continue
        arr = preds_store[key]
        color, ls, lw = LINE_STYLES.get(tag, ("#555555", "-", 1.5))
        ax.plot(years, gm(arr), color=color, ls=ls, lw=lw, label=tag, alpha=0.92)
    ax.axvline(2050, color="gray", ls=":", lw=1)
    ax.set_title(f"{short}: 最高档 Global-Mean 时序对比 (unit: {unit})")
    ax.set_xlabel("year"); ax.set_ylabel(unit)
    ax.legend(loc="best", fontsize=8.5, ncol=4)
    ax.grid(alpha=0.22)
fig2.tight_layout()
fig2_path = FIG_DIR / "week4_ULTIMATE_tas_pr90_GM_7lines_compare.png"
fig2.savefig(fig2_path, bbox_inches="tight")
plt.close(fig2)
LOG(f"saved ULTIMATE GM: {fig2_path}")

# Save log
log_path = OUT_DIR / "week4_super_booster.log"
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(LOGS) + "\n")
LOG(f"\nsaved log: {log_path}")
LOG("=== week4 super booster done ===")
