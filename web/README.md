# web/

Intentionally empty. The frontend has not been scaffolded yet.

When it is, it consumes the FastAPI service in `api/` over HTTP and nothing
else. It must not read `artifacts/` or `data/` from disk, and it must not know
that a training stack exists.

Planned surface:

- a sol-ordered filmstrip of frames with their novelty scores
- the two-budget dial: move the downlink and compute sliders, watch the
  selected set change
- a per-class breakdown against the `test_novel` class folders

Nothing else in the repo depends on this directory.
