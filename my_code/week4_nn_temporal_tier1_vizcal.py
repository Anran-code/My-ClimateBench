import os
from pathlib import Path
import sys

LOGS = []
def LOG(s):
    LOGS.append(str(s))
    print(s, flush=True)

LOG("=== week4 tier-1 clean: GM-visual-cal (left) + raw spatial RMSE (right) ===")

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_models.utils import create_predictdand_data, get_rmse

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

model_cfgs = [
    ("RF",  "#7f7f7f", rf_nc),
    ("CNN", "#1f77b4", cnn_nc),
    ("RNN", "#2ca02c", rnn_nc),
]
for label, color, path in model_cfgs:
    if path is None:
        LOG(f"WARNING: missing {label} pred nc")
    else:
        LOG(f"{label} nc: {path.name}")

Y_test_all = create_predictdand_data(["ssp245"])

targets_cfg = [
    ("tas", "tas", "K"),
    ("diurnal_temperature_range", "dtr", "K"),
    ("pr", "pr", "mm/day"),
    ("pr90", "pr90", "mm/day"),
]

weights_lat = np.cos(np.deg2rad(np.asarray(Y_test_all["tas"].lat.values))).astype(np.float32)
wsum = float(weights_lat.sum())


def gm_and_rmse(ds_var, truth, n_years):
    dim0 = tuple(ds_var.dims)[0]
    p_full = ds_var.sizes[dim0]
    gm = np.full(n_years, np.nan, dtype=np.float32)
    rmse = np.full(n_years, np.nan, dtype=np.float32)
    offset = n_years - p_full if p_full < n_years else 0
    P_np = np.asarray(ds_var.values, dtype=np.float32)
    T_np = np.asarray(truth.values, dtype=np.float32)
    if P_np.ndim == 4: P_np = P_np.reshape(P_np.shape[0], P_np.shape[1], P_np.shape[2])
    if T_np.ndim == 4: T_np = T_np.reshape(T_np.shape[0], T_np.shape[1], T_np.shape[2])
    for i in range(p_full):
        y = offset + i
        pp = P_np[i]
        tt = T_np[y]
        gm[y] = float(np.nansum(pp * weights_lat[:, None]) / (wsum * truth.sizes["lon"]))
        try:
            tt_xa = xr.DataArray(tt, dims=("lat", "lon"), coords={"lat": truth.lat, "lon": truth.lon})
            pp_xa = xr.DataArray(pp, dims=("lat", "lon"), coords={"lat": truth.lat, "lon": truth.lon})
            r = get_rmse(tt_xa, pp_xa)
            rmse[y] = float(np.asarray(r).reshape(-1)[0])
        except Exception:
            rmse[y] = np.nan
    return gm, rmse


def calibrate_gm(years, gm_pred, gm_truth):
    mask_early = years < 2050
    gm_cal = np.full_like(gm_pred, np.nan, dtype=np.float32)
    gain = np.nan
    for is_early in (True, False):
        mask_reg = mask_early if is_early else ~mask_early
        mm = mask_reg & ~np.isnan(gm_pred) & ~np.isnan(gm_truth)
        if mm.sum() < 2:
            continue
        if is_early:
            sp = np.std(gm_pred[mm])
            st = np.std(gm_truth[mm])
            gain = st / sp if sp > 1e-9 else 1.0
        if np.isfinite(gain):
            mp = np.mean(gm_pred[mm])
            mt = np.mean(gm_truth[mm])
            gm_cal[mm] = mt + gain * (gm_pred[mm] - mp)
    return gm_cal, gain


summary_rows = []

fig, axes = plt.subplots(len(targets_cfg), 2,
                         figsize=(17, 4.0 * len(targets_cfg)), dpi=120, squeeze=False)

for i, (truth_name, short, unit) in enumerate(targets_cfg):
    truth = Y_test_all[truth_name]
    n_years = truth.sizes["time"]
    years = np.asarray(truth.time.values).astype(int)

    ax_gm, ax_rmse = axes[i, 0], axes[i, 1]

    # Truth global-mean
    T_np = np.asarray(truth.values, dtype=np.float32)
    if T_np.ndim == 4: T_np = T_np.reshape(T_np.shape[0], T_np.shape[1], T_np.shape[2])
    truth_gm = np.array([
        np.nansum(T_np[t] * weights_lat[:, None]) / (wsum * truth.sizes["lon"])
        for t in range(n_years)
    ], dtype=np.float32)
    ax_gm.plot(years, truth_gm, color="black", lw=2.3, label="truth (GM)", zorder=10)

    for mname, color, path in model_cfgs:
        if path is None: continue
        ds = xr.open_dataset(path)
        if truth_name not in ds.data_vars: continue
        gm_raw, rmse_raw = gm_and_rmse(ds[truth_name], truth, n_years)
        gm_cal, gain = calibrate_gm(years, gm_raw, truth_gm)

        valid = ~np.isnan(gm_raw)
        ax_gm.plot(years[valid], gm_raw[valid], color=color, ls=":", lw=1.4, alpha=0.5,
                   label=f"{mname} raw")
        ax_gm.plot(years[valid], gm_cal[valid], color=color, ls="--", lw=2.1,
                   label=f"{mname} GM-viz cal (g={gain:.2f})")

        valid_r = ~np.isnan(rmse_raw)
        ax_rmse.plot(years[valid_r], rmse_raw[valid_r], color=color, lw=2.1, label=mname)

        mask_late = years >= 2050
        if mask_late.sum() <= 0: mask_late = np.ones_like(years, dtype=bool)

        def safe_mean(x):
            m = ~np.isnan(x)
            return float(np.mean(x[m])) if m.sum() > 0 else np.nan

        late_rmse_raw = safe_mean(rmse_raw[mask_late])
        gm_mae_raw = safe_mean(np.abs(gm_raw[mask_late] - truth_gm[mask_late]))
        gm_mae_cal = safe_mean(np.abs(gm_cal[mask_late] - truth_gm[mask_late]))

        LOG(f"{short:4s} {mname:3s}: g_gm={gain:.3f} | "
            f"late_RMSE(raw)={late_rmse_raw:.5f} {unit} | "
            f"GM_MAE late: raw={gm_mae_raw:.5f} -> cal={gm_mae_cal:.5f} {unit}")
        summary_rows.append({
            "target": short, "unit": unit, "model": mname,
            "gain_gm_cal": gain,
            "rmse_late_raw": late_rmse_raw,
            "gm_mae_late_raw": gm_mae_raw,
            "gm_mae_late_cal": gm_mae_cal,
            "gm_mae_late_improvement": gm_mae_raw - gm_mae_cal,
        })

    ax_gm.axvline(2050, color="gray", ls=":", lw=1)
    ax_gm.set_title(f"{short}: GM series (left) — visual calibration applied on GM curve only ({unit})")
    ax_gm.set_xlabel("year"); ax_gm.set_ylabel(unit)
    ax_gm.legend(loc="best", fontsize=8.2, ncol=2)
    ax_gm.grid(alpha=0.2)

    ax_rmse.axvline(2050, color="gray", ls=":", lw=1)
    ax_rmse.set_title(f"{short}: spatial RMSE (right) — kept RAW (fair held-out metric, {unit})")
    ax_rmse.set_xlabel("year"); ax_rmse.set_ylabel(unit)
    ax_rmse.legend(loc="best", fontsize=9)
    ax_rmse.grid(alpha=0.2)

fig.tight_layout()
fig_path = FIG_DIR / "week4_models_temporal_tier1_vizcal+rawRMSE.png"
fig.savefig(fig_path, bbox_inches="tight")
plt.close(fig)
LOG(f"\nsaved figure: {fig_path}")

summary_df = pd.DataFrame(summary_rows)
csv_path = OUT_DIR / "all_models_ssp245_tier1_vizcal_summary.csv"
summary_df.to_csv(csv_path, index=False)
LOG(f"saved summary csv: {csv_path}")

LOG("\n=== GM MAE late improvement (viz-cal vs raw) ===")
LOG(summary_df.pivot_table(index="target", columns="model",
                           values="gm_mae_late_improvement").round(5).to_string())

LOG("\n=== Late RMSE raw (kept untouched on the right panel) ===")
LOG(summary_df.pivot_table(index="target", columns="model",
                           values="rmse_late_raw").round(5).to_string())

log_path = OUT_DIR / "week4_tier1_vizcal_and_rawRMSE.log"
log_path.write_text("\n".join(LOGS) + "\n", encoding="utf-8")
LOG(f"\nsaved log: {log_path}")
LOG("=== week4 tier1 vizcal done ===")
