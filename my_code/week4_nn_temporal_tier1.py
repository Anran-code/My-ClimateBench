import os
from pathlib import Path
import sys

LOGS = []
def LOG(s):
    LOGS.append(str(s))
    print(s, flush=True)

LOG("=== week4 tier-1 calibration temporal visualize: raw vs calibrated GM + RMSE ===")

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
weights_lat_xr = xr.DataArray(weights_lat, dims=("lat",), coords={"lat": Y_test_all["tas"].lat})
wsum = float(weights_lat.sum())


def gm_and_rmse_series(ds_var, truth, n_years, years):
    dim0 = tuple(ds_var.dims)[0]
    p_full = ds_var.sizes[dim0]
    gm = np.full(n_years, np.nan, dtype=np.float32)
    rmse = np.full(n_years, np.nan, dtype=np.float32)
    if p_full < n_years:
        offset = n_years - p_full
    else:
        offset = 0
    for i in range(p_full):
        pp = np.asarray(ds_var.isel({dim0: i}).values, dtype=np.float32)
        if pp.ndim == 3 and pp.shape[-1] == 1:
            pp = pp.reshape(pp.shape[0], pp.shape[1])
        tt = np.asarray(truth.isel(time=offset + i).values, dtype=np.float32)
        gm[offset + i] = float(np.nansum(pp * weights_lat[:, None]) / (wsum * truth.sizes["lon"]))
        try:
            tt_xa = xr.DataArray(tt, dims=("lat", "lon"),
                                 coords={"lat": truth.lat, "lon": truth.lon})
            pp_xa = xr.DataArray(pp, dims=("lat", "lon"),
                                 coords={"lat": truth.lat, "lon": truth.lon})
            r = get_rmse(tt_xa, pp_xa)
            rmse[offset + i] = float(np.asarray(r).reshape(-1)[0])
        except Exception:
            rmse[offset + i] = np.nan
    return gm, rmse


def calibrate_gm(years, gm_pred, gm_truth):
    mask_early = years < 2050
    gm_cal = np.full_like(gm_pred, np.nan, dtype=np.float32)
    gain = np.nan
    for is_early in [True, False]:
        mask_reg = mask_early if is_early else ~mask_early
        mm = mask_reg & ~np.isnan(gm_pred) & ~np.isnan(gm_truth)
        if mm.sum() < 2:
            continue
        if is_early:
            s_p = np.std(gm_pred[mm])
            s_t = np.std(gm_truth[mm])
            if s_p > 1e-9:
                gain = s_t / s_p
            else:
                gain = 1.0
        if np.isfinite(gain):
            mu_p = np.mean(gm_pred[mm])
            mu_t = np.mean(gm_truth[mm])
            gm_cal[mm] = mu_t + gain * (gm_pred[mm] - mu_p)
    return gm_cal, gain


def spatial_rescale(ds_var, truth, years):
    mask_early = years < 2050
    dim0 = tuple(ds_var.dims)[0]
    p_full = ds_var.sizes[dim0]
    n_years = len(years)
    if p_full < n_years:
        offset = n_years - p_full
        year_idx = np.arange(p_full) + offset
    else:
        offset = 0
        year_idx = np.arange(n_years)

    P_np = np.asarray(ds_var.values, dtype=np.float32)
    T_np = np.asarray(truth.values, dtype=np.float32)
    if P_np.ndim == 4:
        P_np = P_np.reshape(P_np.shape[0], P_np.shape[1], P_np.shape[2])
    if T_np.ndim == 4:
        T_np = T_np.reshape(T_np.shape[0], T_np.shape[1], T_np.shape[2])

    w = weights_lat[:, None].astype(np.float32)
    T_early = T_np[year_idx[mask_early[year_idx]]]
    P_early = P_np[mask_early[year_idx[:p_full]]]

    scalar_gain = 1.0
    if P_early.size >= 10 and T_early.size >= 10:
        num = np.nansum(P_early * T_early * w[None, :, :])
        den = np.nansum((P_early ** 2) * w[None, :, :])
        if den > 1e-12:
            scalar_gain = float(num / den)
        scalar_gain = float(np.clip(scalar_gain, 0.5, 2.0))

    bias_field_early = np.nanmean(T_early - P_early * scalar_gain, axis=0)

    out_full = np.full((n_years, P_np.shape[1], P_np.shape[2]), np.nan, dtype=np.float32)
    for i_local in range(p_full):
        y = year_idx[i_local]
        out_full[y] = P_np[i_local] * scalar_gain + bias_field_early
    return out_full, scalar_gain, bias_field_early


def rmse_from_np(P_full, truth_np):
    n_years = truth_np.shape[0]
    r = np.full(n_years, np.nan, dtype=np.float32)
    for t in range(n_years):
        pp = P_full[t]
        tt = truth_np[t]
        if np.any(~np.isfinite(pp)):
            continue
        se = (pp - tt) ** 2
        w = weights_lat[:, None].astype(np.float32)
        rmse = np.sqrt(np.nansum(se * w[None, :, :]) / np.nansum(w[None, :, :]))
        r[t] = float(rmse)
    return r


summary_rows = []

ncols = 2
fig, axes = plt.subplots(len(targets_cfg), ncols,
                         figsize=(17, 4.0 * len(targets_cfg)), dpi=120, squeeze=False)

for i, (truth_name, short, unit) in enumerate(targets_cfg):
    truth = Y_test_all[truth_name]
    n_years = truth.sizes["time"]
    years = np.asarray(truth.time.values).astype(int)
    truth_np = np.asarray(truth.values, dtype=np.float32)
    if truth_np.ndim == 4:
        truth_np = truth_np.reshape(truth_np.shape[0], truth_np.shape[1], truth_np.shape[2])

    ax_gm, ax_rmse = axes[i, 0], axes[i, 1]

    truth_gm = np.full(n_years, np.nan, dtype=np.float32)
    for t in range(n_years):
        tt = truth_np[t]
        truth_gm[t] = float(np.nansum(tt * weights_lat[:, None]) / (wsum * truth.sizes["lon"]))

    ax_gm.plot(years, truth_gm, color="black", linewidth=2.3, label="truth (GM)", zorder=10)

    for mname, color, path in model_cfgs:
        if path is None:
            continue
        ds = xr.open_dataset(path)
        if truth_name not in ds.data_vars:
            continue
        gm_raw, rmse_raw = gm_and_rmse_series(ds[truth_name], truth, n_years, years)

        gm_cal, gain = calibrate_gm(years, gm_raw, truth_gm)
        P_cal_field, s_gain, bias_field = spatial_rescale(ds[truth_name], truth, years)
        rmse_cal = rmse_from_np(P_cal_field, truth_np)

        valid = ~np.isnan(gm_raw)
        ax_gm.plot(years[valid], gm_raw[valid], color=color, linestyle=":",
                   linewidth=1.7, alpha=0.65, label=f"{mname} raw")
        ax_gm.plot(years[valid], gm_cal[valid], color=color, linestyle="--",
                   linewidth=2.1, label=f"{mname} tier-1 cal (gain={gain:.2f})")

        valid_r = ~np.isnan(rmse_raw)
        ax_rmse.plot(years[valid_r], rmse_raw[valid_r], color=color, linestyle=":",
                     linewidth=1.7, alpha=0.65, label=f"{mname} raw")
        valid_rc = ~np.isnan(rmse_cal)
        ax_rmse.plot(years[valid_rc], rmse_cal[valid_rc], color=color, linestyle="--",
                     linewidth=2.1, label=f"{mname} tier-1 cal (scalar gain={s_gain:.2f})")

        mask_late = years >= 2050
        if mask_late.sum() <= 0:
            mask_late = np.ones_like(years, dtype=bool)

        def safe_mean(x):
            m = ~np.isnan(x)
            return float(np.mean(x[m])) if m.sum() > 0 else np.nan

        late_raw = safe_mean(rmse_raw[mask_late])
        late_cal = safe_mean(rmse_cal[mask_late])
        gm_mae_raw = safe_mean(np.abs(gm_raw[mask_late] - truth_gm[mask_late]))
        gm_mae_cal = safe_mean(np.abs(gm_cal[mask_late] - truth_gm[mask_late]))

        LOG(f"{short:4s} {mname:3s}: gain(gm/spatial) = {gain:.3f}/{s_gain:.3f} | "
            f"late_RMSE: raw={late_raw:.5f} -> cal={late_cal:.5f} | "
            f"late_GM_MAE: raw={gm_mae_raw:.5f} -> cal={gm_mae_cal:.5f}")
        summary_rows.append({
            "target": short, "unit": unit, "model": mname,
            "gain_gm_tier1": gain, "gain_spatial_tier1": s_gain,
            "rmse_late_raw": late_raw, "rmse_late_cal": late_cal,
            "rmse_late_improvement": (late_raw - late_cal),
            "gm_mae_late_raw": gm_mae_raw, "gm_mae_late_cal": gm_mae_cal,
            "gm_mae_late_improvement": (gm_mae_raw - gm_mae_cal),
        })

    ax_gm.axvline(2050, color="gray", linestyle=":", linewidth=1)
    ax_gm.set_title(f"{short}: GM series — raw vs tier-1 calibrated ({unit})")
    ax_gm.set_xlabel("year")
    ax_gm.set_ylabel(unit)
    ax_gm.legend(loc="best", fontsize=8.2, ncol=2)
    ax_gm.grid(alpha=0.2)

    ax_rmse.axvline(2050, color="gray", linestyle=":", linewidth=1)
    ax_rmse.set_title(f"{short}: spatial RMSE — raw vs tier-1 calibrated ({unit})")
    ax_rmse.set_xlabel("year")
    ax_rmse.set_ylabel(unit)
    ax_rmse.legend(loc="best", fontsize=8.2, ncol=2)
    ax_rmse.grid(alpha=0.2)

fig.tight_layout()
fig_path = FIG_DIR / "week4_models_temporal_tier1_calibrated.png"
fig.savefig(fig_path, bbox_inches="tight")
plt.close(fig)
LOG(f"\nsaved figure: {fig_path}")

summary_df = pd.DataFrame(summary_rows)
csv_path = OUT_DIR / "all_models_ssp245_tier1_calibration_summary.csv"
summary_df.to_csv(csv_path, index=False)
LOG(f"saved summary csv: {csv_path}")

LOG("\n=== Tier-1 improvement pivot (late-period) ===")
imp_pivot = summary_df.pivot_table(index="target", columns="model",
                                    values=["rmse_late_improvement", "gm_mae_late_improvement"])
LOG(imp_pivot.round(5).to_string())

LOG("\n=== Tier-1 absolute late RMSE (raw vs cal) ===")
abs_pivot = summary_df.pivot_table(index="target", columns="model",
                                   values=["rmse_late_raw", "rmse_late_cal"])
LOG(abs_pivot.round(5).to_string())

log_path = OUT_DIR / "week4_tier1_temporal_calibrate.log"
log_path.write_text("\n".join(LOGS) + "\n", encoding="utf-8")
LOG(f"\nsaved log: {log_path}")
LOG("=== week4 tier1 temporal visualize done ===")
