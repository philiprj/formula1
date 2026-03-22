"""Model registry and factory."""

import importlib

MODEL_REGISTRY = {
    "linear": "f1deg.models.linear:LinearDegradationModel",
    "bayesian": "f1deg.models.bayesian:BayesianDegradationModel",
    "gbm": "f1deg.models.gbm:GBMDegradationModel",
    "sequence": "f1deg.models.sequence:SequenceDegradationModel",
}


def get_model_class(model_name: str):
    """Import and return the model class by name."""
    module_path, class_name = MODEL_REGISTRY[model_name].rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
