import os
from pathlib import Path
import sys

LOGS = []
def LOG(s):
    LOGS.append(str(s))
    print(s, flush=True)

LOG("=== week4 nn temporal compare: RF vs CNN vs RNN time series ===")

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
        LOG(f"WARNING: missing {label} pred nc, will skip this model in plots")
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

summary_rows = []

ncols = 2
fig, axes = plt.subplots(len(targets_cfg), ncols,
                         figsize=(16, 3.9 * len(targets_cfg)), dpi=120, squeeze=False)

for i, (truth_name, short, unit) in enumerate(targets_cfg):
    truth = Y_test_all[truth_name]
    years = np.asarray(truth.time.values).astype(int)
    N = len(years)

    ax_gm, ax_rmse = axes[i, 0], axes[i, 1]

    truth_gm = np.full(N, np.nan, dtype=np.float32)
    for t in range(N):
        tt = np.asarray(truth.isel(time=t).values, dtype=np.float32)
        truth_gm[t] = float(np.nansum(tt * weights_lat[:, None]) / (wsum * truth.sizes["lon"]))
    ax_gm.plot(years, truth_gm, color="black", linewidth=2.2, label="truth (GM)", zorder=10)

    ts_rmse = {}

    for mname, color, path in model_cfgs:
        if path is None:
            continue
        ds = xr.open_dataset(path)
        if truth_name not in ds.data_vars:
            LOG(f"  skip {mname} for {short}: no variable")
            continue
        pred = ds[truth_name]
        dim0 = tuple(pred.dims)[0]
        P_full = pred.sizes[dim0]
        if P_full < N:
            offset = N - P_full
            LOG(f"  {mname} {short}: pred has {P_full} samples vs truth {N}, offset={offset} (earliest years no pred)")
            sample_idx = np.arange(P_full)
            year_idx = sample_idx + offset
        else:
            year_idx = np.arange(N)
            sample_idx = np.arange(N)

        pred_gm = np.full(N, np.nan, dtype=np.float32)
        rmse_per_year = np.full(N, np.nan, dtype=np.float32)
        for yt, sp in zip(year_idx, sample_idx):
            tt = np.asarray(truth.isel(time=yt).values, dtype=np.float32)
            pp = np.asarray(pred.isel({dim0: sp}).values, dtype=np.float32)
            if pp.ndim == 3 and pp.shape[-1] == 1:
                pp = pp.reshape(pp.shape[0], pp.shape[1])
            pred_gm[yt] = float(np.nansum(pp * weights_lat[:, None]) / (wsum * truth.sizes["lon"]))
            try:
                tt_xa = xr.DataArray(tt, dims=("lat", "lon"),
                                     coords={"lat": truth.lat, "lon": truth.lon})
                pp_xa = xr.DataArray(pp, dims=("lat", "lon"),
                                     coords={"lat": truth.lat, "lon": truth.lon})
                r = get_rmse(tt_xa, pp_xa)
                rmse_per_year[yt] = float(np.asarray(r).reshape(-1)[0])
            except Exception as e:
                LOG(f"  {mname} {short} rmse at year {years[yt]} skipped: {e!r}")

        valid = ~np.isnan(pred_gm)
        ax_gm.plot(years[valid], pred_gm[valid], color=color, linestyle="--", linewidth=2,
                   label=f"{mname} pred (GM)")

        valid_r = ~np.isnan(rmse_per_year)
        ax_rmse.plot(years[valid_r], rmse_per_year[valid_r], color=color, linewidth=2,
                     label=f"{mname} RMSE")
        ts_rmse[mname] = rmse_per_year

        mask_late = years >= 2050
        if mask_late.sum() <= 0:
            mask_late = np.ones_like(years, dtype=bool)
        mask_early = ~mask_late

        early_r = rmse_per_year[mask_early & valid_r]
        late_r = rmse_per_year[mask_late & valid_r]
        rmse_early = float(np.mean(early_r)) if len(early_r) > 0 else np.nan
        rmse_late = float(np.mean(late_r)) if len(late_r) > 0 else np.nan

        pg = pred_gm[mask_late & valid]
        tg = truth_gm[mask_late & ~np.isnan(truth_gm)]
        bias_gm_late = (float(np.mean(pg)) - float(np.mean(tg))) if len(pg) > 0 and len(tg) > 0 else np.nan

        LOG(f"{short} | {mname}: RMSE early(<2050)={rmse_early:.6f}, late(>=2050)={rmse_late:.6f} {unit}")
        LOG(f"{short} | {mname}: GM bias late(pred-truth)={bias_gm_late:.6f} {unit}")

        summary_rows.append({
            "target": short,
            "model": mname,
            "unit": unit,
            "rmse_early_mean": rmse_early,
            "rmse_late_mean": rmse_late,
            "gm_bias_late_mean": bias_gm_late,
        })

    ax_gm.axvline(2050, color="gray", linestyle=":", linewidth=1)
    ax_gm.set_title(f"{short}: global-mean time series ({unit})")
    ax_gm.set_xlabel("year")
    ax_gm.set_ylabel(unit)
    ax_gm.legend(loc="best", fontsize=9)
    ax_gm.grid(alpha=0.2)

    ax_rmse.axvline(2050, color="gray", linestyle=":", linewidth=1)
    ax_rmse.set_title(f"{short}: spatial RMSE over time (lat-weighted, {unit})")
    ax_rmse.set_xlabel("year")
    ax_rmse.set_ylabel(unit)
    ax_rmse.legend(loc="best", fontsize=9)
    ax_rmse.grid(alpha=0.2)

fig.tight_layout()
fig_path = FIG_DIR / "week4_models_temporal_compare.png"
fig.savefig(fig_path, bbox_inches="tight")
plt.close(fig)
LOG(f"saved temporal figure: {fig_path}")

summary_df = pd.DataFrame(summary_rows)
csv_path = OUT_DIR / "all_models_ssp245_temporal_summary.csv"
summary_df.to_csv(csv_path, index=False)
LOG(f"saved temporal summary csv: {csv_path}")

late_summary_pivot = summary_df.pivot_table(index="target", columns="model",
                                            values=["rmse_late_mean", "gm_bias_late_mean"])
LOG("--- Pivot: late RMSE & GM bias across models ---")
LOG(late_summary_pivot.to_string())

log_path = OUT_DIR / "week4_nn_temporal_compare.log"
log_path.write_text("\n".join(LOGS) + "\n", encoding="utf-8")
LOG(f"saved log: {log_path}")
LOG("=== week4 nn temporal compare done ===")
