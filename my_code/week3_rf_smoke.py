import os
from pathlib import Path
import sys

TEST_LOG = Path(__file__).resolve().parent / "outputs" / "smoke.log"
TEST_LOG.parent.mkdir(parents=True, exist_ok=True)

def LOG(s):
    with open(TEST_LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")
        f.flush()
    print(s, flush=True)

LOG("=== start smoke ===")

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

LOG("dll paths done")

import netCDF4  # noqa: F401
LOG("netCDF4 ok")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xarray as xr
LOG("numpy/pandas/xarray ok")

from baseline_models.utils import (
    create_predictor_data,
    create_predictdand_data,
    get_test_data,
    get_rmse,
)
LOG("utils import ok")

from esem import rf_model
LOG("esem import ok")

train_files = ["historical", "ssp585", "ssp126", "ssp370"]
test_file = "ssp245"

LOG("build train data")
X_train, eof_solvers = create_predictor_data(train_files, n_eofs=5)
Y_train = create_predictdand_data(train_files)
LOG(f"X_train {X_train.shape}, Y_train tas {Y_train['tas'].shape}")

LOG("build test data")
X_test = get_test_data(test_file, eof_solvers, n_eofs=5)
Y_test = create_predictdand_data([test_file])
LOG(f"X_test {X_test.shape}, Y_test tas {Y_test['tas'].shape}")

LOG("train tiny RF for tas")
rf = rf_model(
    X_train,
    Y_train["tas"],
    random_state=0,
    bootstrap=True,
    max_features=1.0,
    **{"n_estimators": 10, "min_samples_split": 5, "min_samples_leaf": 7, "max_depth": 3},
)
rf.train()
LOG("train done")

pred, _ = rf.predict(X_test)
LOG(f"predict done, pred sample dim={pred.shape}")

xr_output = xr.Dataset({"tas": pred}).assign_coords(time=pred.sample + 2014)
out_path = PROJECT_ROOT / "my_code" / "outputs" / "outputs_ssp245_prediction_RF_tas_smoke.nc"
xr_output.to_netcdf(str(out_path), "w")
LOG(f"saved {out_path}")

rmse_map = get_rmse(Y_test["tas"][35:], pred[35:])
rmse_scalar = float(np.mean(rmse_map))
LOG(f"tas eval rmse(35:end) = {rmse_scalar:.6f}")

results_path = PROJECT_ROOT / "my_code" / "outputs" / "rf_ssp245_scores_tas_smoke.csv"
pd.Series({"tas": rmse_scalar}).to_csv(results_path, header=["mean_lat_weighted_rmse"])
LOG(f"saved scores {results_path}")
LOG("=== smoke done ===")
