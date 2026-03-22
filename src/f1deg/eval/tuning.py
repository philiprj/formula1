"""Optuna-based hyperparameter tuning for GBM degradation model.

Uses a subset of leave-one-race-out CV folds as the objective function.
"""

import logging

import pandas as pd

from f1deg.eval.cv import leave_one_race_out_cv
from f1deg.models.gbm import GBMDegradationModel

logger = logging.getLogger(__name__)


def tune_gbm(
    df: pd.DataFrame,
    config: dict,
    n_trials: int = 50,
    max_folds: int = 10,
) -> dict:
    """Run Optuna hyperparameter search for GBM model.

    Args:
        df: Full processed dataset.
        config: Base model configuration.
        n_trials: Number of Optuna trials.
        max_folds: Number of LORO folds per trial (subset for speed).

    Returns:
        Dict with best_params, best_mae, and study summary.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tuning_config = config.get("model", {}).get("tuning", {})
    search_space = tuning_config.get(
        "search_space",
        {
            "n_estimators": [300, 1000],
            "max_depth": [4, 8],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.6, 0.9],
            "colsample_bytree": [0.6, 0.95],
            "min_child_weight": [3, 15],
            "reg_alpha": [0.01, 1.0],
            "reg_lambda": [0.1, 5.0],
        },
    )

    backend = config.get("model", {}).get("backend", "xgboost")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", search_space["n_estimators"][0], search_space["n_estimators"][1]
            ),
            "max_depth": trial.suggest_int(
                "max_depth", search_space["max_depth"][0], search_space["max_depth"][1]
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                search_space["learning_rate"][0],
                search_space["learning_rate"][1],
                log=True,
            ),
            "subsample": trial.suggest_float(
                "subsample", search_space["subsample"][0], search_space["subsample"][1]
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                search_space["colsample_bytree"][0],
                search_space["colsample_bytree"][1],
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                search_space["min_child_weight"][0],
                search_space["min_child_weight"][1],
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", search_space["reg_alpha"][0], search_space["reg_alpha"][1], log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", search_space["reg_lambda"][0], search_space["reg_lambda"][1], log=True
            ),
        }

        # Build trial config
        trial_config = config.copy()
        trial_config["model"] = config.get("model", {}).copy()
        trial_config["model"][backend] = params

        result = leave_one_race_out_cv(df, GBMDegradationModel, trial_config, max_folds=max_folds)
        mae = result["aggregate"].get("mae", float("inf"))
        logger.info(f"Trial {trial.number}: MAE={mae:.4f}s params={params}")
        return float(mae)

    study = optuna.create_study(direction="minimize", study_name="gbm_tuning")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    logger.info(f"Best trial: MAE={best.value:.4f}s")
    logger.info(f"Best params: {best.params}")

    return {
        "best_params": best.params,
        "best_mae": best.value,
        "n_trials": len(study.trials),
        "trials_summary": [
            {"number": t.number, "mae": t.value, "params": t.params}
            for t in sorted(study.trials, key=lambda t: t.value)[:5]
        ],
    }
