"""NOVUM core: pure-Python (numpy-only) dataset, model, scoring and budget logic.

Nothing in this package may import torch, scikit-learn, scipy or any other
training-only dependency at module scope. The serving layer imports `core`, and
the serving image does not contain those packages. Training-only code lives
behind lazy imports inside `core.models` implementations.
"""

__version__ = "0.1.0"
