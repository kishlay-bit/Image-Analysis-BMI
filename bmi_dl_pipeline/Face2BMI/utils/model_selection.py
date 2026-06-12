
import optuna
import numpy as np
from sklearn.metrics import mean_absolute_error


def find_3model_ensemble(preds_dict, true_labels, n_trials=500):
    names = list(preds_dict.keys())
    preds = [preds_dict[n] for n in names]

    def objective(trial):
        w     = [trial.suggest_float(f"w{i}", 0.0, 1.0)
                 for i in range(len(names))]
        w_sum = sum(w) + 1e-9
        w_norm = [wi / w_sum for wi in w]
        ensemble = sum(wn * p
                       for wn, p in zip(w_norm, preds))
        return mean_absolute_error(true_labels, ensemble)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials,
                   show_progress_bar=True)
    raw_w  = [study.best_params[f"w{i}"]
              for i in range(len(names))]
    w_sum  = sum(raw_w) + 1e-9
    w_norm = [w / w_sum for w in raw_w]
    return dict(zip(names, w_norm)), study.best_value
