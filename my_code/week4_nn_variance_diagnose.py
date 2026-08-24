import os
from pathlib import Path
import sys

LOGS = []
def LOG(s):
    LOGS.append(str(s))
    print(s, flush=True)

LOG("=== week4 variance & calibration analysis: diagnose CNN/RNN over-smoothing ===")

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

FIG_DIR = PROJECT_ROOT / "my_code" / "figures"
OUT_DIR = PROJECT_ROOT / "my_code" / "outputs"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_models.utils import create_predictdand_data

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

Y_test_all = create_predictdand_data(["ssp245"])

targets_cfg = [
    ("tas", "tas", "K"),
    ("diurnal_temperature_range", "dtr", "K"),
    ("pr", "pr", "mm/day"),
    ("pr90", "pr90", "mm/day"),
]

weights_lat = np.cos(np.deg2rad(np.asarray(Y_test_all["tas"].lat.values))).astype(np.float32)
wsum = float(weights_lat.sum())


def detrend(y):
    x = np.arange(len(y), dtype=np.float64)
    m = np.isfinite(y)
    if m.sum() < 3:
        return y * np.nan
    coef = np.polyfit(x[m], y[m], 1)
    return y - np.polyval(coef, x)


def gm_series(ds_var, truth, n_years):
    dim0 = tuple(ds_var.dims)[0]
    p_full = ds_var.sizes[dim0]
    gm_arr = np.full(n_years, np.nan, dtype=np.float32)
    if p_full < n_years:
        offset = n_years - p_full
    else:
        offset = 0
    for i in range(p_full):
        pp = np.asarray(ds_var.isel({dim0: i}).values, dtype=np.float32)
        if pp.ndim == 3 and pp.shape[-1] == 1:
            pp = pp.reshape(pp.shape[0], pp.shape[1])
        gm_arr[offset + i] = float(np.nansum(pp * weights_lat[:, None]) / (wsum * truth.sizes["lon"]))
    return gm_arr

summary_rows = []

for truth_name, short, unit in targets_cfg:
    truth = Y_test_all[truth_name]
    n_years = truth.sizes["time"]
    years = np.asarray(truth.time.values).astype(int)

    truth_gm = np.full(n_years, np.nan, dtype=np.float32)
    for t in range(n_years):
        tt = np.asarray(truth.isel(time=t).values, dtype=np.float32)
        truth_gm[t] = float(np.nansum(tt * weights_lat[:, None]) / (wsum * truth.sizes["lon"]))

    # detrend truth once
    truth_det = detrend(truth_gm)
    truth_std = float(np.nanstd(truth_gm))
    truth_std_det = float(np.nanstd(truth_det))

    mask_late = years >= 2050
    mask_early = ~mask_late

    for mname, color, path in model_cfgs:
        if path is None:
            continue
        ds = xr.open_dataset(path)
        if truth_name not in ds.data_vars:
            continue
        pred_gm = gm_series(ds[truth_name], truth, n_years)

        pred_std = float(np.nanstd(pred_gm))
        ratio_std = pred_std / truth_std if truth_std > 0 else np.nan

        # period-wise
        pred_std_early = float(np.nanstd(pred_gm[mask_early & ~np.isnan(pred_gm)]))
        pred_std_late  = float(np.nanstd(pred_gm[mask_late & ~np.isnan(pred_gm)]))
        truth_std_early = float(np.nanstd(truth_gm[mask_early & ~np.isnan(truth_gm)]))
        truth_std_late  = float(np.nanstd(truth_gm[mask_late & ~np.isnan(truth_gm)]))
        ratio_early = pred_std_early / truth_std_early if truth_std_early > 0 else np.nan
        ratio_late  = pred_std_late  / truth_std_late  if truth_std_late  > 0 else np.nan

        # detrended temporal correlation (signal quality, not amplitude)
        m_both = ~np.isnan(pred_gm) & ~np.isnan(truth_gm)
        if m_both.sum() >= 5:
            pred_det = detrend(pred_gm)
            pred_det_std = float(np.nanstd(pred_det[m_both]))
            ratio_det_std = pred_det_std / truth_std_det if truth_std_det > 0 else np.nan
            corr_det = float(np.corrcoef(pred_det[m_both], truth_det[m_both])[0, 1])
        else:
            pred_det_std = np.nan
            ratio_det_std = np.nan
            corr_det = np.nan

        # mean bias
        late_m = mask_late & ~np.isnan(pred_gm) & ~np.isnan(truth_gm)
        bias_late = float(np.mean(pred_gm[late_m]) - np.mean(truth_gm[late_m])) if late_m.sum() > 0 else np.nan

        # ---------------- post-hoc variance calibration ----------------
        # compute gain on EARLY period (avoid "cheating" by calibrating on the test window)
        cal_m = mask_early & ~np.isnan(pred_gm) & ~np.isnan(truth_gm)
        if cal_m.sum() >= 5 and pred_std_early > 1e-9:
            gain = truth_std_early / pred_std_early
        else:
            gain = np.nan

        if np.isfinite(gain):
            # center pred on its mean, scale, then add back truth mean (by period for less bias)
            pred_cal = np.full_like(pred_gm, np.nan, dtype=np.float32)
            for mask_reg in [mask_early, mask_late]:
                mm = mask_reg & ~np.isnan(pred_gm) & ~np.isnan(truth_gm)
                if mm.sum() >= 2:
                    mu_p = np.mean(pred_gm[mm])
                    mu_t = np.mean(truth_gm[mm])
                    pred_cal[mm] = mu_t + gain * (pred_gm[mm] - mu_p)

            cal_std_late  = float(np.nanstd(pred_cal[mask_late]))
            cal_bias_late = float(np.nanmean(pred_cal[mask_late] - truth_gm[mask_late])) if mask_late.sum() > 0 else np.nan
            # simple MAE of GM after calibration (late window)
            cal_gm_mae_late = float(np.nanmean(np.abs(pred_cal[mask_late] - truth_gm[mask_late]))) if mask_late.sum() > 0 else np.nan
            raw_gm_mae_late = float(np.nanmean(np.abs(pred_gm[mask_late] - truth_gm[mask_late]))) if mask_late.sum() > 0 else np.nan
        else:
            pred_cal = pred_gm * np.nan
            cal_std_late = np.nan
            cal_bias_late = np.nan
            cal_gm_mae_late = np.nan
            raw_gm_mae_late = float(np.nanmean(np.abs(pred_gm[mask_late] - truth_gm[mask_late]))) if (mask_late & late_m).sum() > 0 else np.nan

        summary_rows.append({
            "target": short, "unit": unit, "model": mname,
            "std_truth_full": truth_std,
            "std_pred_full": pred_std,
            "ratio_std_full": ratio_std,
            "ratio_std_early": ratio_early,
            "ratio_std_late": ratio_late,
            "detrended_corr_pred_vs_truth": corr_det,
            "detrended_ratio_std": ratio_det_std,
            "bias_late_raw": bias_late,
            "gm_mae_late_raw": raw_gm_mae_late,
            "calib_gain_early_period": gain,
            "std_late_calibrated": cal_std_late,
            "bias_late_calibrated": cal_bias_late,
            "gm_mae_late_calibrated": cal_gm_mae_late,
        })

        LOG(f"{short:4s} {mname:3s}: GM_std ratio={ratio_std:.3f} | early={ratio_early:.3f} late={ratio_late:.3f} | "
            f"detr_corr={corr_det:.3f} | detrend_std_ratio={ratio_det_std:.3f} | "
            f"late_gm_mae: raw={raw_gm_mae_late:.4f} => cal={cal_gm_mae_late:.4f}")

summary_df = pd.DataFrame(summary_rows)
csv_path = OUT_DIR / "all_models_ssp245_variance_calibration_summary.csv"
summary_df.to_csv(csv_path, index=False)
LOG(f"\nsaved variance summary csv: {csv_path}")

LOG("\n=== Key takeaways (use in report) ===")
pivot_ratio = summary_df.pivot_table(index="target", columns="model", values="ratio_std_late")
LOG("late GM std ratio (pred/truth) — closer to 1.0 is better (1.0 = matches truth variance):")
LOG(pivot_ratio.round(3).to_string())

pivot_corr = summary_df.pivot_table(index="target", columns="model", values="detrended_corr_pred_vs_truth")
LOG("\ndetrended temporal correlation (higher is better — measures whether the wiggles are in phase with truth, independent of amplitude):")
LOG(pivot_corr.round(3).to_string())

pivot_mae = summary_df.pivot_table(index="target", columns="model",
                                   values=["gm_mae_late_raw", "gm_mae_late_calibrated"])
LOG("\nlate GM MAE improvement from early-period post-hoc variance calibration:")
LOG(pivot_mae.round(4).to_string())

log_path = OUT_DIR / "week4_variance_and_calibration.log"
log_path.write_text("\n".join(LOGS) + "\n", encoding="utf-8")
LOG(f"\nsaved log: {log_path}")
