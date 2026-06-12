
import optuna
import numpy as np


def find_best_alpha(results_list):
    """
    Use Optuna to find optimal alpha in:
    score = alpha * (1/MAE) + (1-alpha) * R2
    """
    def objective(trial):
        alpha = trial.suggest_float('alpha', 0.0, 1.0)
        scores = []
        for r in results_list:
            score = alpha * (1.0 / (r['MAE'] + 1e-6)) + (1 - alpha) * r['R2']
            scores.append(score)
        # We want alpha that maximally separates best from worst model
        return max(scores) - min(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200, show_progress_bar=False)

    best_alpha = study.best_params['alpha']
    print(f"✅ Optuna best alpha : {best_alpha:.4f}")
    return best_alpha


def rank_models(results_list, alpha):
    """Rank models using weighted score."""
    ranked = []
    for r in results_list:
        score = alpha * (1.0 / (r['MAE'] + 1e-6)) + (1 - alpha) * r['R2']
        ranked.append({**r, 'Score': round(score, 6)})
    return sorted(ranked, key=lambda x: x['Score'], reverse=True)
