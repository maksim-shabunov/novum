"""NOVUM serving layer.

ARCHITECTURAL RULE: nothing reachable from this package may import torch,
scikit-learn, or any other training dependency -- not at module scope, not
lazily, not at all. The API consumes trained weight artifacts; it does not
train. `tests/test_no_training_deps.py` enforces this in a subprocess and will
fail the build if it is ever violated.
"""
