import os
from pathlib import Path
import sys

LOGS = []
def P(s):
    LOGS.append(str(s))
    print(s, flush=True)

P("=== week3 4-targets error hotspots analysis ===")

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

from baseline_models.utils import create_predictdand_data

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

TOPN = 5
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
    mask_late = years >= 2050
    if mask_late.sum() <= 0:
        mask_late = np.ones_like(years, dtype=bool)

    err_late = (pred_t.isel(sample=mask_late) - truth.isel(time=mask_late))
    weights_lat = np.cos(np.deg2rad(truth.lat))
    weights_lat_arr = xr.DataArray(weights_lat, dims=["lat"])
    wabs = np.abs(err_late) * weights_lat_arr
    mae_latlon = wabs.mean(dim="sample", skipna=True) / weights_lat_arr
    mae_latlon_values = np.asarray(mae_latlon.values, dtype=float)
    if mae_latlon_values.ndim == 3:
        mae_latlon_values = np.nanmean(mae_latlon_values, axis=-1)
    mae_latlon_values = mae_latlon_values.reshape(truth.lat.size, truth.lon.size)
    mae_latlon = xr.DataArray(mae_latlon_values, dims=("lat", "lon"), coords={"lat": truth.lat, "lon": truth.lon})

    mae_flat = np.asarray(mae_latlon.values).reshape(-1)
    order = np.argsort(-mae_flat)[:TOPN]
    lat_2d, lon_2d = np.meshgrid(truth.lat.values, truth.lon.values, indexing="ij")
    lat_flat = lat_2d.reshape(-1)
    lon_flat = lon_2d.reshape(-1)

    ax0, ax1 = axes[row]

    vmax = float(mae_latlon.max().values)
    if vmax == 0:
        vmax = 1e-6
    im = ax0.pcolormesh(mae_latlon.lon, mae_latlon.lat, mae_latlon.values, shading="auto", vmin=0, vmax=vmax)
    ax0.set_title(f"{short_name}: |error| mean over late period ({unit})")
    plt.colorbar(im, ax=ax0)
    ax0.set_xlabel("lon")
    ax0.set_ylabel("lat")

    ax1.set_title(f"{short_name}: top-{TOPN} error hotspots (late)")
    im2 = ax1.pcolormesh(mae_latlon.lon, mae_latlon.lat, mae_latlon.values, shading="auto", vmin=0, vmax=vmax, alpha=0.35)
    plt.colorbar(im2, ax=ax1)
    for rank, idx in enumerate(order):
        la = float(lat_flat[idx])
        lo = float(lon_flat[idx])
        v = float(mae_flat[idx])
        ax1.scatter(lo, la, s=40, color="tab:red", edgecolors="black", linewidths=0.6, zorder=5)
        ax1.annotate(f"#{rank+1}\n{v:.2f}{unit}", (lo, la), fontsize=8,
                     textcoords="offset points", xytext=(4, 4))
        region = "land" if abs(la) > 60 else "ocean-like"
        summary_rows.append({
            "target": short_name,
            "rank": rank + 1,
            "lat": la,
            "lon": lo,
            "region_rule": region,
            "unit": unit,
            "abs_error_late_mean": v,
        })
        P(f"{short_name} hotspot #{rank+1}: lat={la:.1f}, lon={lo:.1f}, abs_err={v:.4f}{unit}")
    ax1.set_xlabel("lon")
    ax1.set_ylabel("lat")

fig.tight_layout()
fig_path = FIG_DIR / "week3_4targets_error_hotspots.png"
fig.savefig(fig_path, bbox_inches="tight")
plt.close(fig)
P(f"Saved figure: {fig_path}")

summary_df = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / "rf_ssp245_error_hotspots.csv"
summary_df.to_csv(summary_path, index=False)
P(f"Saved hotspots summary: {summary_path}")

log_path = OUT_DIR / "week3_4targets_hotspots.log"
log_path.write_text("\n".join(LOGS) + "\n", encoding="utf-8")
P(f"Saved log: {log_path}")
