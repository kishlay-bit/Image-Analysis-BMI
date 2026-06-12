
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from scipy.stats import pearsonr


def compute_metrics(labels, preds, model_name="Model"):
    labels = np.array(labels)
    preds = np.array(preds)

    mae = mean_absolute_error(labels, preds)
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, preds)

    mape = np.mean(
        np.abs((labels - preds) / (labels + 1e-6))
    ) * 100

    pearson = pearsonr(labels, preds)[0]
    bias = np.mean(preds - labels)

    extreme_mask = (labels < 18.5) | (labels > 35)

    if extreme_mask.sum() > 0:
        extreme_mae = mean_absolute_error(
            labels[extreme_mask],
            preds[extreme_mask]
        )
    else:
        extreme_mae = float("nan")

    return {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE%": round(mape, 2),
        "Pearson": round(pearson, 4),
        "Bias": round(bias, 4),
        "Extreme_MAE": round(extreme_mae, 4)
    }


def predict_tta(
    model,
    dataset,
    tta_transforms,
    device,
    batch_size=64
):
    all_preds = []

    for transform in tta_transforms:
        dataset.transform = transform

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        preds = []

        model.eval()

        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)

                with autocast():
                    outputs = model(images)

                preds.extend(
                    outputs.detach().cpu().numpy()
                )

        all_preds.append(np.array(preds))

    return np.mean(all_preds, axis=0)


def plot_predicted_vs_actual(
    labels,
    preds,
    model_name,
    save_path
):
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        labels,
        preds,
        alpha=0.3,
        s=8,
        color="steelblue"
    )

    mn = min(labels)
    mx = max(labels)

    ax.plot(
        [mn, mx],
        [mn, mx],
        "r--",
        lw=1.5,
        label="Perfect"
    )

    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)

    ax.set_title(
        f"{model_name}\\nMAE={mae:.4f}  R2={r2:.4f}"
    )

    ax.set_xlabel("Actual BMI")
    ax.set_ylabel("Predicted BMI")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()


def plot_residuals(
    labels,
    preds,
    model_name,
    save_path
):
    residuals = np.array(labels) - np.array(preds)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        residuals,
        bins=40,
        color="darkorange",
        edgecolor="white",
        alpha=0.85
    )

    ax.axvline(
        0,
        color="red",
        linestyle="--",
        lw=1.5
    )

    ax.set_title(
        f"{model_name} Residuals\\n"
        f"Mean={residuals.mean():.3f}  "
        f"Std={residuals.std():.3f}"
    )

    ax.set_xlabel(
        "Residual (Actual - Predicted)"
    )

    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()


def plot_bmi_range_errors(
    labels,
    preds,
    model_name,
    save_path
):
    labels = np.array(labels)
    preds = np.array(preds)

    bins = {
        "Underweight(<18.5)": (0, 18.5),
        "Normal(18.5-25)": (18.5, 25),
        "Overweight(25-30)": (25, 30),
        "Obese(30-35)": (30, 35),
        "Extreme(>35)": (35, 100)
    }

    range_maes = []
    range_names = []

    for name, (lo, hi) in bins.items():
        mask = (labels >= lo) & (labels < hi)

        if mask.sum() > 0:
            range_maes.append(
                mean_absolute_error(
                    labels[mask],
                    preds[mask]
                )
            )

            range_names.append(
                f"{name}\\n(n={mask.sum()})"
            )

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        range_names,
        range_maes,
        color="steelblue",
        edgecolor="white"
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3
    )

    ax.set_title(
        f"{model_name} - MAE by BMI Range"
    )

    ax.set_ylabel("MAE")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()
