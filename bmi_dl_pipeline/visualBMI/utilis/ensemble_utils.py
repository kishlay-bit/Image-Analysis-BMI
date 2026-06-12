
import numpy as np
import torch
from tqdm import tqdm

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# GENERATE PREDICTIONS
# =========================================================

def generate_predictions(model, loader, device):

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for images, labels in tqdm(loader):

            images = images.to(device)

            outputs = model(images).squeeze(1)

            preds.extend(outputs.cpu().numpy())
            targets.extend(labels.numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    return preds, targets


# =========================================================
# EVALUATE METRICS
# =========================================================

def evaluate_regression(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
