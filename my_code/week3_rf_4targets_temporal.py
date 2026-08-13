import os
from pathlib import Path
import sys

LOGS = []
def P(s):
    LOGS.append(str(s))
    print(s, flush=True)

P("=== week3 4-targets temporal analysis ===")

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

from baseline_models.utils import create_predictdand_data, get_rmse

FIG_DIR = PROJECT_ROOT / "my_code" / "figures"
OUT_DIR = PROJECT_ROOT / "my_code" / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

pred_paths = [
    OUT_DIR / "outputs_ssp245_prediction_RF.nc",
    OUT_DIR / "outputs_ssp245_prediction_RF_4targets_smoke.nc",
]
pred_path = None
for p in pred_paths:
    if p.exists():
        pred_path = p
        break
if pred_path is None:
    P("Missing prediction NetCDF. Please run week3_rf_baseline.py or week3_rf_4targets_smoke.py first.")
    sys.exit(1)

P(f"Using prediction file: {pred_path.name}")

pred_all = xr.open_dataset(pred_path)
Y_test = create_predictdand_data(["ssp245"])

targets_cfg = [
    ("tas", "tas", "K"),
    ("diurnal_temperature_range", "dtr", "K"),
    ("pr", "pr", "mm/day"),
    ("pr90", "pr90", "mm/day"),
]

weights_lat = np.cos(np.deg2rad(Y_test["tas"].lat))
weights_lat.name = "weight"

summary_rows = []

fig, axes = plt.subplots(len(targets_cfg), 2, figsize=(16, 4 * len(targets_cfg)), dpi=120)
if len(targets_cfg) == 1:
    axes = axes.reshape(1, -1)

for row, (truth_name, short_name, unit) in enumerate(targets_cfg):
    truth = Y_test[truth_name]
    pred = pred_all[truth_name]
    if "time" in pred.dims:
        pred_t = pred.rename(time="sample")
    else:
        pred_t = pred

    years = truth.time.values.astype(int) if hasattr(truth.time.values, "astype") else truth.time.values

    truth_gm = float("nan") * np.ones(len(years))
    pred_gm = float("nan") * np.ones(len(years))
    for t in range(len(years)):
        tt = truth.isel(time=t)
        pp = pred_t.isel(sample=t)
        wsum = weights_lat.sum()
        truth_gm[t] = float((tt.weighted(weights_lat).mean(("lat", "lon"))).values)
        pred_gm[t] = float((pp.weighted(weights_lat).mean(("lat", "lon"))).values)

    rmse_ts = np.full(len(years), np.nan)
    for t in range(len(years)):
        tt = truth.isel(time=t)
        pp = pred_t.isel(sample=t)
        r = get_rmse(tt, pp)
        rmse_ts[t] = float(np.asarray(r).reshape(-1)[0])

    ax0, ax1 = axes[row]
    ax0.plot(years, truth_gm, label="truth (GM)", linewidth=2)
    ax0.plot(years, pred_gm, label="RF pred (GM)", linestyle="--", linewidth=2)
    ax0.axvline(2050, color="gray", linestyle=":", linewidth=1, label="2050 split")
    ax0.set_title(f"{short_name}: global-mean time series ({unit})")
    ax0.set_xlabel("year")
    ax0.set_ylabel(unit)
    ax0.legend(loc="best")

    ax1.plot(years, rmse_ts, color="tab:red", linewidth=2, label="lat-weighted RMSE per year")
    ax1.axvline(2050, color="gray", linestyle=":", linewidth=1, label="2050 split")
    ax1.set_title(f"{short_name}: spatial RMSE over time (lat-weighted)")
    ax1.set_xlabel("year")
    ax1.set_ylabel(unit)
    ax1.legend(loc="best")

    mask_eval = years >= 2050
    if mask_eval.sum() <= 0:
        mask_eval = np.ones_like(years, dtype=bool)
    rmse_late = float(np.mean(rmse_ts[mask_eval]))
    rmse_early = float(np.mean(rmse_ts[~mask_eval]))

    truth_gm_late = float(np.mean(truth_gm[mask_eval]))
    pred_gm_late = float(np.mean(pred_gm[mask_eval]))
    bias_gm_late = pred_gm_late - truth_gm_late

    P(f"{short_name}: RMSE early(<2050)={rmse_early:.6f}, late(>=2050)={rmse_late:.6f} {unit}")
    P(f"{short_name}: GM bias in late period(pred - truth) = {bias_gm_late:.6f} {unit}")

    summary_rows.append({
        "target": short_name,
        "unit": unit,
        "rmse_early_mean": rmse_early,
        "rmse_late_mean": rmse_late,
        "gm_bias_late_mean": bias_gm_late,
    })

fig.tight_layout()
fig_path = FIG_DIR / "week3_4targets_temporal_analysis.png"
fig.savefig(fig_path, bbox_inches="tight")
plt.close(fig)
P(f"Saved figure: {fig_path}")

summary_df = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / "rf_ssp245_temporal_summary.csv"
summary_df.to_csv(summary_path, index=False)
P(f"Saved summary: {summary_path}")

log_path = OUT_DIR / "week3_4targets_temporal.log"
log_path.write_text("\n".join(LOGS) + "\n", encoding="utf-8")
P(f"Saved log: {log_path}")
