from pathlib import Path
import sys


# Ensure the project root is on sys.path when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline_models.utils import create_predictor_data, create_predictdand_data


def inspect_single_scenarios():
    """Inspect the structure of historical and ssp245 data."""
    X_hist, _ = create_predictor_data("historical")
    Y_hist = create_predictdand_data("historical")

    X_ssp245, _ = create_predictor_data("ssp245")
    Y_ssp245 = create_predictdand_data("ssp245")

    print("=== X historical ===")
    print("shape:", X_hist.shape)
    print("columns:", list(X_hist.columns))
    print("time:", X_hist.index.min(), "->", X_hist.index.max())
    print()

    print("=== Y historical ===")
    print("vars:", list(Y_hist.data_vars))
    print("dims:", Y_hist.dims)
    print("time:", int(Y_hist.time.min()), "->", int(Y_hist.time.max()))
    print()

    print("=== X ssp245 ===")
    print("shape:", X_ssp245.shape)
    print("columns:", list(X_ssp245.columns))
    print("time:", X_ssp245.index.min(), "->", X_ssp245.index.max())
    print()

    print("=== Y ssp245 ===")
    print("vars:", list(Y_ssp245.data_vars))
    print("dims:", Y_ssp245.dims)
    print("time:", int(Y_ssp245.time.min()), "->", int(Y_ssp245.time.max()))
    print()


def inspect_train_concat():
    """Inspect the concatenated training scenarios."""
    datasets = ["historical", "ssp126", "ssp370", "ssp585"]
    X_train, _ = create_predictor_data(datasets)
    Y_train = create_predictdand_data(datasets)

    print("=== Train Concat ===")
    print("datasets:", datasets)
    print("X_train shape:", X_train.shape)
    print("X_train columns:", list(X_train.columns))
    print("Y_train vars:", list(Y_train.data_vars))
    print("Y_train dims:", Y_train.dims)
    print("Y_train time:", int(Y_train.time.min()), "->", int(Y_train.time.max()))
    print()


if __name__ == "__main__":
    inspect_single_scenarios()
    inspect_train_concat()
