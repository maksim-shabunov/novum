"""Novelty model implementations and the lazy registry that resolves them.

Nothing here imports torch at module scope. The registry maps a model type name
to a (module, class) pair and imports on demand, so `import core.models` stays
safe inside the serving image, which has no training stack installed.
"""

from .base import NoveltyModel
from .registry import available_models, get_model_class, load_model

__all__ = ["NoveltyModel", "available_models", "get_model_class", "load_model"]
