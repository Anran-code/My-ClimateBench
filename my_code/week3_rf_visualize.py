import os
from pathlib import Path
import sys

from pathlib import Path
import sys

LOG = []
def P(s):
    LOG.append(str(s))
    print(s, flush=True)

P("=== week3 visualization ===")

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
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_models.utils import create_predictdand_data, get_rmse

FIG_DIR = PROJECT_ROOT / "my_code" / "figures"
OUT_DIR = PROJECT_ROOT / "my_code" / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

pred_path = OUT_DIR / "outputs_ssp245_prediction_RF_tas_smoke.nc"
if not pred_path.exists():
    P(f"Missing {pred_path}, please run week3_rf_smoke.py first")
    sys.exit(1)

pred = xr.open_dataset(pred_path)
Y_test = create_predictdand_data(["ssp245"])

truth = Y_test["tas"]
tas_pred = pred["tas"]

if "time" in tas_pred.dims:
    tas_pred_t = tas_pred.rename(time="sample")
else:
    tas_pred_t = tas_pred

time_eval_idx = 35
year = int(truth.time.values[time_eval_idx])
P(f"Target evaluation year index={time_eval_idx}, year={year}")

t_true = truth.isel(time=time_eval_idx)
t_pred = tas_pred_t.isel(sample=time_eval_idx)
t_err = t_pred - t_true

rmse_map = get_rmse(truth.isel(time=slice(35, None)), tas_pred_t.isel(sample=slice(35, None)))
rmse_scalar = float(np.mean(rmse_map))
P(f"tas lat-weighted RMSE over time(35:) then mean over samples = {rmse_scalar:.6f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 4), dpi=120)

vmin = float(min(t_true.min(), t_pred.min()))
vmax = float(max(t_true.max(), t_pred.max()))
abs_max = max(abs(float(t_err.min())), abs(float(t_err.max())))

im0 = axes[0].pcolormesh(t_true.lon, t_true.lat, t_true.values, shading="auto", vmin=vmin, vmax=vmax)
axes[0].set_title(f"SSP245 truth tas [{year}]")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(t_pred.lon, t_pred.lat, t_pred.values, shading="auto", vmin=vmin, vmax=vmax)
axes[1].set_title(f"RF pred tas [{year}]")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].pcolormesh(t_err.lon, t_err.lat, t_err.values, shading="auto", vmin=-abs_max, vmax=abs_max, cmap="RdBu_r")
axes[2].set_title(f"pred - truth (bias)")
plt.colorbar(im2, ax=axes[2])

for ax in axes:
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")

fig.tight_layout()
fig_path = FIG_DIR / "week3_tas_truth_pred_error.png"
fig.savefig(fig_path, bbox_inches="tight")
plt.close(fig)
P(f"Saved figure: {fig_path}")

log_path = OUT_DIR / "week3_viz.log"
log_path.write_text("\n".join(LOG), encoding="utf-8")
P(f"Saved log: {log_path}")
