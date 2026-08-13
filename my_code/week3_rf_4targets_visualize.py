import os
from pathlib import Path
import sys

LOGS = []
def P(s):
    LOGS.append(str(s))
    print(s, flush=True)

P("=== week3 4-targets visualization ===")

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

pred_path = OUT_DIR / "outputs_ssp245_prediction_RF_4targets_smoke.nc"
if not pred_path.exists():
    P(f"Missing {pred_path}, please run week3_rf_4targets_smoke.py first")
    sys.exit(1)

pred_all = xr.open_dataset(pred_path)
Y_test = create_predictdand_data(["ssp245"])

time_eval_idx = 35
year = int(Y_test["tas"].time.values[time_eval_idx])
P(f"Target evaluation year index={time_eval_idx}, year={year}")

targets_cfg = [
    ("tas", "tas", "K"),
    ("diurnal_temperature_range", "dtr", "K"),
    ("pr", "pr", "mm/day"),
    ("pr90", "pr90", "mm/day"),
]

results = {}
fig, axes = plt.subplots(len(targets_cfg), 3, figsize=(18, 4 * len(targets_cfg)), dpi=120)
if len(targets_cfg) == 1:
    axes = axes.reshape(1, -1)

for row, (truth_name, short_name, unit) in enumerate(targets_cfg):
    truth = Y_test[truth_name]
    pred = pred_all[truth_name]
    if "time" in pred.dims:
        pred_t = pred.rename(time="sample")
    else:
        pred_t = pred

    t_true = truth.isel(time=time_eval_idx)
    t_pred = pred_t.isel(sample=time_eval_idx)
    t_err = t_pred - t_true

    rmse_map = get_rmse(truth.isel(time=slice(35, None)), pred_t.isel(sample=slice(35, None)))
    rmse_scalar = float(np.mean(rmse_map))
    P(f"RMSE[{short_name}](35:) = {rmse_scalar:.6f} {unit}")
    results[short_name] = rmse_scalar

    vmin = float(min(t_true.min(), t_pred.min()))
    vmax = float(max(t_true.max(), t_pred.max()))
    if vmin == vmax:
        vmin = vmin - 1e-6
        vmax = vmax + 1e-6
    abs_max = max(abs(float(t_err.min())), abs(float(t_err.max())))
    if abs_max == 0:
        abs_max = 1e-6

    ax0, ax1, ax2 = axes[row]
    im0 = ax0.pcolormesh(t_true.lon, t_true.lat, t_true.values, shading="auto", vmin=vmin, vmax=vmax)
    ax0.set_title(f"Truth {short_name} [{year}] ({unit})")
    plt.colorbar(im0, ax=ax0)

    im1 = ax1.pcolormesh(t_pred.lon, t_pred.lat, t_pred.values, shading="auto", vmin=vmin, vmax=vmax)
    ax1.set_title(f"RF pred {short_name} [{year}] ({unit})")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.pcolormesh(t_err.lon, t_err.lat, t_err.values, shading="auto", vmin=-abs_max, vmax=abs_max, cmap="RdBu_r")
    ax2.set_title(f"Error {short_name} (pred-truth)")
    plt.colorbar(im2, ax=ax2)

    for ax in [ax0, ax1, ax2]:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

fig.tight_layout()
fig_path = FIG_DIR / "week3_4targets_truth_pred_error.png"
fig.savefig(fig_path, bbox_inches="tight")
plt.close(fig)
P(f"Saved figure: {fig_path}")

results_path = OUT_DIR / "rf_ssp245_scores_4targets_viz.csv"
pd.Series(results).to_csv(results_path, header=["mean_lat_weighted_rmse"])
P(f"Saved scores: {results_path}")

log_path = OUT_DIR / "week3_4targets_viz.log"
log_path.write_text("\n".join(LOGS) + "\n", encoding="utf-8")
P(f"Saved log: {log_path}")
