import os
from pathlib import Path
import sys

LOGS = []

def LOG(s):
    LOGS.append(str(s))
    print(s, flush=True)

LOG("=== week3_rf_4targets_smoke start ===")

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

from baseline_models.utils import (
    create_predictor_data,
    create_predictdand_data,
    get_test_data,
    get_rmse,
)
from esem import rf_model

OUT_DIR = PROJECT_ROOT / "my_code" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "week3_4targets_smoke.log"

train_files = ["historical", "ssp585", "ssp126", "ssp370"]
test_file = "ssp245"

LOG("build train data")
X_train, eof_solvers = create_predictor_data(train_files, n_eofs=5)
Y_train = create_predictdand_data(train_files)
LOG(f"X_train {X_train.shape}, Y_train vars={list(Y_train.data_vars)}")

LOG("build test data")
X_test = get_test_data(test_file, eof_solvers, n_eofs=5)
Y_test = create_predictdand_data([test_file])
LOG(f"X_test {X_test.shape}, Y_test tas {Y_test['tas'].shape}")

rf_params = {
    "tas": {"n_estimators": 10, "min_samples_split": 5, "min_samples_leaf": 7, "max_depth": 3},
    "pr": {"n_estimators": 10, "min_samples_split": 15, "min_samples_leaf": 8, "max_depth": 5},
    "pr90": {"n_estimators": 10, "min_samples_split": 15, "min_samples_leaf": 12, "max_depth": 4},
    "diurnal_temperature_range": {"n_estimators": 10, "min_samples_split": 10, "min_samples_leaf": 12, "max_depth": 4},
}
targets = list(rf_params.keys())

predictions = {}
for target in targets:
    LOG(f"train tiny RF for {target}")
    rf = rf_model(
        X_train,
        Y_train[target],
        random_state=0,
        bootstrap=True,
        max_features=1.0,
        **rf_params[target],
    )
    rf.train()
    pred, _ = rf.predict(X_test)
    predictions[target] = pred
    LOG(f"predict done for {target}, shape={pred.shape}")

tas_pred = predictions["tas"]
pr_pred = predictions["pr"]
pr90_pred = predictions["pr90"]
dtr_pred = predictions["diurnal_temperature_range"]

xr_output = xr.Dataset(
    dict(
        tas=tas_pred,
        pr=pr_pred,
        pr90=pr90_pred,
        diurnal_temperature_range=dtr_pred,
    )
).assign_coords(time=tas_pred.sample + 2014)

out_path = OUT_DIR / "outputs_ssp245_prediction_RF_4targets_smoke.nc"
xr_output.to_netcdf(str(out_path), "w")
LOG(f"saved {out_path}")

results = {}
for target, pred in zip(["tas", "diurnal_temperature_range", "pr", "pr90"], [tas_pred, dtr_pred, pr_pred, pr90_pred]):
    truth = Y_test[target]
    rmse_map = get_rmse(truth.isel(time=slice(35, None)), pred.isel(sample=slice(35, None)))
    rmse_scalar = float(np.mean(rmse_map))
    key = "dtr" if target == "diurnal_temperature_range" else target
    LOG(f"RMSE[{key}](35:) = {rmse_scalar:.6f}")
    results[key] = rmse_scalar

results_path = OUT_DIR / "rf_ssp245_scores_4targets_smoke.csv"
pd.Series(results).to_csv(results_path, header=["mean_lat_weighted_rmse"])
LOG(f"saved {results_path}")

LOG_FILE.write_text("\n".join(LOGS) + "\n", encoding="utf-8")
LOG(f"saved log {LOG_FILE}")
LOG("=== week3_rf_4targets_smoke done ===")
