from pathlib import Path
import os
import sys


# Ensure the project root is on sys.path when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows can find DLLs shipped inside the conda environment.
if sys.platform == "win32":
    env_root = Path(sys.executable).resolve().parent.parent
    dll_dir = env_root / "Library" / "bin"
    if dll_dir.exists():
        os.add_dll_directory(str(dll_dir))

import matplotlib.pyplot as plt

from baseline_models.utils import create_predictor_data, create_predictdand_data


OUTPUT_DIR = PROJECT_ROOT / "my_code" / "figures"


def plot_input_time_series():
    """Plot the main input features from the historical scenario."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_hist, _ = create_predictor_data("historical")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    features = ["CO2", "CH4", "BC_0", "SO2_0"]
    titles = [
        "Historical CO2",
        "Historical CH4",
        "Historical BC_0",
        "Historical SO2_0",
    ]

    for ax, feature, title in zip(axes.flat, features, titles):
        ax.plot(X_hist.index, X_hist[feature], linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel(feature)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "historical_input_timeseries.png", dpi=150)
    plt.close(fig)


def plot_output_spatial_fields():
    """Plot tas and pr spatial fields for the first historical year."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    Y_hist = create_predictdand_data("historical")
    year = int(Y_hist.time[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    Y_hist["tas"].isel(time=0).plot(ax=axes[0], cmap="coolwarm")
    axes[0].set_title(f"Historical tas ({year})")

    Y_hist["pr"].isel(time=0).plot(ax=axes[1], cmap="viridis")
    axes[1].set_title(f"Historical pr ({year})")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "historical_spatial_fields.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_input_time_series()
    plot_output_spatial_fields()
    print(f"Saved figures to: {OUTPUT_DIR}")
