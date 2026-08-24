import os
from pathlib import Path
import sys

LOGS = []
def LOG(s):
    LOGS.append(str(s))
    print(s, flush=True)

LOG("=== week4 nn error compare: RF vs CNN vs RNN (hotspots) ===")

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

from baseline_models.utils import create_predictdand_data

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
        LOG(f"WARNING: missing {label} pred nc, pls run corresponding training script")
    else:
        LOG(f"{label} nc: {path.name}")

Y_test_all = create_predictdand_data(["ssp245"])
targets_cfg = [
    ("tas", "tas", "K"),
    ("diurnal_temperature_range", "dtr", "K"),
    ("pr", "pr", "mm/day"),
    ("pr90", "pr90", "mm/day"),
]
model_cfgs = [
    ("RF", rf_nc),
    ("CNN", cnn_nc),
    ("RNN", rnn_nc),
]

TOPN = 5

def compute_mae_late(pred_xa, truth_xa):
    years = np.asarray(truth_xa.time.values).astype(int)
    mask_late = years >= 2050
    if mask_late.sum() <= 0:
        mask_late = np.ones_like(years, dtype=bool)
    idx = np.where(mask_late)[0]
    n_valid = int(len(idx))

    T_full = truth_xa.sizes["time"]
    P_full = pred_xa.sizes[tuple(pred_xa.dims)[0]]
    if P_full != T_full:
        offset = T_full - P_full
        if offset < 0:
            offset = 0
        pred_idx = idx - offset
        pred_idx = np.clip(pred_idx, 0, P_full - 1)
    else:
        pred_idx = idx

    P_np = np.asarray(pred_xa.values, dtype=np.float32)
    T_np = np.asarray(truth_xa.values, dtype=np.float32)
    if P_np.ndim == 4:
        P_np = P_np.reshape(P_np.shape[0], P_np.shape[1], P_np.shape[2])
    if T_np.ndim == 4:
        T_np = T_np.reshape(T_np.shape[0], T_np.shape[1], T_np.shape[2])
    P_slice = P_np[pred_idx]
    T_slice = T_np[idx]
    N = min(P_slice.shape[0], T_slice.shape[0])
    P_slice = P_slice[:N]
    T_slice = T_slice[:N]
    abs_err = np.abs(P_slice - T_slice)

    weights_lat = np.cos(np.deg2rad(np.asarray(truth_xa.lat.values))).astype(np.float32)
    w2d = weights_lat[:, None]
    mae_latlon = np.nanmean(abs_err * w2d[None, :, :], axis=0) / np.maximum(w2d, 1e-12)
    return mae_latlon

summary_rows = []
fig, axes = plt.subplots(len(targets_cfg), len(model_cfgs),
                         figsize=(6 * len(model_cfgs), 3.8 * len(targets_cfg)), dpi=120, squeeze=False)
for j, (mname, path) in enumerate(model_cfgs):
    if path is None:
        for i in range(len(targets_cfg)):
            axes[i, j].axis("off")
            axes[i, j].text(0.5, 0.5, f"{mname}: no file", ha="center", va="center", color="red")
        continue
    ds_pred = xr.open_dataset(path)
    for i, (tvar, short, unit) in enumerate(targets_cfg):
        truth = Y_test_all[tvar]
        if tvar not in ds_pred.data_vars:
            axes[i, j].axis("off")
            axes[i, j].text(0.5, 0.5, f"{mname}: no {short}", ha="center", va="center", color="red")
            continue
        pred = ds_pred[tvar]
        # align dim name to lat/lon only
        mae = compute_mae_late(pred, truth)
        vmax = float(np.nanmax(mae))
        if vmax == 0:
            vmax = 1e-6
        lat_1d = truth.lat.to_numpy().reshape(-1)
        lon_1d = truth.lon.to_numpy().reshape(-1)
        lat2, lon2 = np.meshgrid(lat_1d, lon_1d, indexing="ij")
        mae_flat = np.asarray(mae, dtype=np.float64).reshape(-1)
        order = np.argsort(-mae_flat)[:TOPN]

        ax = axes[i, j]
        im = ax.pcolormesh(truth.lon, truth.lat, mae, shading="auto", vmin=0, vmax=vmax)
        for rank, idx in enumerate(order):
            la = float(lat2.reshape(-1)[idx])
            lo = float(lon2.reshape(-1)[idx])
            err_v = float(mae_flat[idx])
            ax.scatter(lo, la, s=24, color="tab:red", edgecolors="black", linewidths=0.5, zorder=5)
            if rank < 3:
                ax.annotate(f"#{rank+1}\n{err_v:.2f}", (lo, la), fontsize=7,
                            textcoords="offset points", xytext=(3, 3))
            summary_rows.append({
                "target": short,
                "model": mname,
                "rank": rank + 1,
                "lat": la,
                "lon": lo,
                "unit": unit,
                "abs_err_late_mean": err_v,
            })
        ax.set_title(f"{short}: {mname} late |err| mean ({unit})")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax)

fig.tight_layout()
fig_path = FIG_DIR / "week4_error_hotspots_RF_vs_CNN_vs_RNN.png"
fig.savefig(fig_path, bbox_inches="tight")
plt.close(fig)
LOG(f"saved hotspots figure: {fig_path}")

summary_df = pd.DataFrame(summary_rows)
csv_path = OUT_DIR / "all_model_error_hotspots_summary.csv"
summary_df.to_csv(csv_path, index=False)
LOG(f"saved hotspots csv: {csv_path}")

# quick focus summary: focus tas high-latitude hotspot #1 and pr90 tropical hotspot #1
focus_lines = []
focus_lines.append("--- Focus: tas high-lat hotspot #1 across models ---")
for m in ["RF", "CNN", "RNN"]:
    sub = summary_df[(summary_df.target == "tas") & (summary_df["rank"] == 1) & (summary_df.model == m)]
    if sub.empty:
        focus_lines.append(f"{m}: no tas hotspot 1")
    else:
        r = sub.iloc[0]
        focus_lines.append(f"{m}: lat={r.lat:.1f}, lon={r.lon:.1f}, err={r.abs_err_late_mean:.4f}{r.unit}")
focus_lines.append("--- Focus: pr90 tropical hotspot #1 across models ---")
for m in ["RF", "CNN", "RNN"]:
    sub = summary_df[(summary_df.target == "pr90") & (summary_df["rank"] == 1) & (summary_df.model == m)]
    if sub.empty:
        focus_lines.append(f"{m}: no pr90 hotspot 1")
    else:
        r = sub.iloc[0]
        focus_lines.append(f"{m}: lat={r.lat:.1f}, lon={r.lon:.1f}, err={r.abs_err_late_mean:.4f}{r.unit}")
for line in focus_lines:
    LOG(line)

log_path = OUT_DIR / "week4_nn_error_compare.log"
log_path.write_text("\n".join(LOGS) + "\n", encoding="utf-8")
LOG(f"saved log: {log_path}")
LOG("=== week4 nn error compare done ===")
