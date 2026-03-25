"""Optuna-based hyperparameter tuning for GBM degradation model.

Uses a subset of leave-one-race-out CV folds as the objective function.
Supports both XGBoost and LightGBM backends with backend-specific search spaces.
"""

import logging

import pandas as pd

from f1deg.eval.cv import leave_one_race_out_cv
from f1deg.models.gbm import GBMDegradationModel

logger = logging.getLogger(__name__)

_XGBOOST_DEFAULTS = {
    "n_estimators": [300, 1000],
    "max_depth": [4, 8],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.6, 0.9],
    "colsample_bytree": [0.6, 0.95],
    "min_child_weight": [3, 15],
    "reg_alpha": [0.01, 1.0],
    "reg_lambda": [0.1, 5.0],
}

_LIGHTGBM_DEFAULTS = {
    "n_estimators": [300, 1000],
    "num_leaves": [20, 255],
    "max_depth": [4, 8],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.6, 0.9],
    "colsample_bytree": [0.6, 0.95],
    "min_child_samples": [5, 50],
    "min_data_in_leaf": [5, 100],
    "reg_alpha": [0.01, 1.0],
    "reg_lambda": [0.1, 5.0],
}


def _suggest_xgboost_params(trial, search_space: dict) -> dict:
    """Build XGBoost hyperparameter suggestions from Optuna trial."""
    return {
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


def _suggest_lightgbm_params(trial, search_space: dict) -> dict:
    """Build LightGBM hyperparameter suggestions from Optuna trial."""
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", search_space["n_estimators"][0], search_space["n_estimators"][1]
        ),
        "num_leaves": trial.suggest_int(
            "num_leaves", search_space["num_leaves"][0], search_space["num_leaves"][1]
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
        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            search_space["min_child_samples"][0],
            search_space["min_child_samples"][1],
        ),
        "min_data_in_leaf": trial.suggest_int(
            "min_data_in_leaf",
            search_space["min_data_in_leaf"][0],
            search_space["min_data_in_leaf"][1],
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", search_space["reg_alpha"][0], search_space["reg_alpha"][1], log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", search_space["reg_lambda"][0], search_space["reg_lambda"][1], log=True
        ),
    }


def tune_gbm(
    df: pd.DataFrame,
    config: dict,
    n_trials: int = 50,
    max_folds: int = 10,
) -> dict:
    """Run Optuna hyperparameter search for GBM model.

    Supports both XGBoost and LightGBM backends with backend-specific
    search spaces and optional trial pruning.

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
    backend = config.get("model", {}).get("backend", "xgboost")
    use_pruner = tuning_config.get("use_pruner", True)

    # Select backend-specific search space
    if backend == "lightgbm":
        defaults = _LIGHTGBM_DEFAULTS
        search_space = tuning_config.get("lightgbm_search_space", defaults)
        # Merge defaults for any missing keys
        for k, v in defaults.items():
            search_space.setdefault(k, v)
        suggest_fn = _suggest_lightgbm_params
    else:
        defaults = _XGBOOST_DEFAULTS
        search_space = tuning_config.get("search_space", defaults)
        for k, v in defaults.items():
            search_space.setdefault(k, v)
        suggest_fn = _suggest_xgboost_params

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial, search_space)

        # Build trial config
        trial_config = config.copy()
        trial_config["model"] = config.get("model", {}).copy()
        trial_config["model"][backend] = params

        result = leave_one_race_out_cv(df, GBMDegradationModel, trial_config, max_folds=max_folds)
        mae = result["aggregate"].get("mae", float("inf"))
        logger.info(f"Trial {trial.number}: MAE={mae:.4f}s params={params}")
        return float(mae)

    # Use MedianPruner to terminate unpromising trials early
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10) if use_pruner else None
    study = optuna.create_study(
        direction="minimize",
        study_name=f"gbm_{backend}_tuning",
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    logger.info(f"Best trial: MAE={best.value:.4f}s")
    logger.info(f"Best params: {best.params}")

    return {
        "best_params": best.params,
        "best_mae": best.value,
        "backend": backend,
        "n_trials": len(study.trials),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "trials_summary": [
            {"number": t.number, "mae": t.value, "params": t.params}
            for t in sorted(study.trials, key=lambda t: t.value)[:5]
        ],
    }
