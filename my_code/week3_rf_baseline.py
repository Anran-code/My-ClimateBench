import os
from pathlib import Path
import sys

LOG_PATH = Path(__file__).resolve().parent / "outputs" / "week3_run.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def LOG(s):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(s + "\n")
        f.flush()
    print(s, flush=True)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

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

import netCDF4  # noqa: F401  # preload to fix DLL order on Windows

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

FIG_DIR = PROJECT_ROOT / "my_code" / "figures"
OUT_DIR = PROJECT_ROOT / "my_code" / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def train_rf_model(X, Y, target, params):
    model = rf_model(X, Y[target], random_state=0, bootstrap=True, max_features=1.0, **params)
    model.train()
    return model


def evaluate_and_print(target_name, truth, pred, eval_slice):
    truth_eval = truth[eval_slice]
    pred_eval = pred[eval_slice]
    rmse_map = get_rmse(truth_eval, pred_eval)
    rmse_scalar = float(np.mean(rmse_map))
    LOG(f"[{target_name}] mean(lat-weighted RMSE over space) on eval slice = {rmse_scalar:.6f}")
    return rmse_scalar


def main():
    LOG("=== week3_rf_baseline start ===")
    train_files = ["historical", "ssp585", "ssp126", "ssp370"]
    test_file = "ssp245"

    LOG("Building training data...")
    X_train, eof_solvers = create_predictor_data(train_files, n_eofs=5)
    Y_train = create_predictdand_data(train_files)

    LOG(f"X_train shape: {X_train.shape}")
    LOG(f"X_train columns: {list(X_train.columns)}")
    LOG(f"Y_train targets: {list(Y_train.data_vars)}")

    LOG("Building test data...")
    X_test = get_test_data(test_file, eof_solvers, n_eofs=5)
    Y_test = create_predictdand_data([test_file])

    LOG(f"X_test shape: {X_test.shape}")
    LOG(f"Y_test tas shape: {Y_test['tas'].shape}")

    rf_params = {
        "tas": {"n_estimators": 250, "min_samples_split": 5, "min_samples_leaf": 7, "max_depth": 5},
        "pr": {"n_estimators": 150, "min_samples_split": 15, "min_samples_leaf": 8, "max_depth": 40},
        "pr90": {"n_estimators": 250, "min_samples_split": 15, "min_samples_leaf": 12, "max_depth": 25},
        "diurnal_temperature_range": {"n_estimators": 300, "min_samples_split": 10, "min_samples_leaf": 12, "max_depth": 20},
    }
    targets = list(rf_params.keys())

    models = {}
    predictions = {}

    for target in targets:
        LOG(f"Training RF for {target}...")
        models[target] = train_rf_model(X_train, Y_train, target, rf_params[target])
        pred, _ = models[target].predict(X_test)
        predictions[target] = pred
        LOG(f"Training + predict done for {target}. pred shape={pred.shape}")

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

    out_path = OUT_DIR / "outputs_ssp245_prediction_RF.nc"
    xr_output.to_netcdf(str(out_path), "w")
    LOG(f"Saved prediction NetCDF to: {out_path}")

    eval_slice = slice(35, None)

    results = {}
    results["tas"] = evaluate_and_print("tas", Y_test["tas"], tas_pred, eval_slice)
    results["diurnal_temperature_range"] = evaluate_and_print("dtr", Y_test["diurnal_temperature_range"], dtr_pred, eval_slice)
    results["pr"] = evaluate_and_print("pr", Y_test["pr"], pr_pred, eval_slice)
    results["pr90"] = evaluate_and_print("pr90", Y_test["pr90"], pr90_pred, eval_slice)

    results_path = OUT_DIR / "rf_ssp245_scores.csv"
    pd.Series(results).to_csv(results_path, header=["mean_lat_weighted_rmse"])
    LOG(f"Saved scores to: {results_path}")
    LOG("=== week3_rf_baseline done ===")


if __name__ == "__main__":
    main()
