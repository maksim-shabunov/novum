# How IBM Bob was used

This is a working account, not a testimonial. It says which parts of the
codebase Bob contributed to, how the work was broken into prompts, where the
output needed correcting, and one case where it caught something that would
plausibly have shipped otherwise.

## Scope, stated plainly

Bob was used on three areas:

| Area | Extent | Files |
|---|---|---|
| **OpenRouter integration** | most of it | `core/ground/llm_provider.py` |
| **Model training** | partly | `scripts/train.py`, `core/models/pca.py`, `core/models/conv_ae.py`, `configs/tier_*.yaml` |
| **Console interface** | partly | `web/src/components/*.tsx`, `web/src/app/` |

It was **not** used for the downlink simulator (`sim/`), the two-budget
selection (`core/budgets.py`), the experiment suite (`scripts/simulate.py`), or
the fact-layer rework described below. Those carry other assistance, and
`git log` shows the trailers; a reader who checks will find the history agrees
with this page. That boundary is worth stating because a document claiming
credit for the whole repository would be checkable and wrong.

## How the work was decomposed

The useful unit of prompting turned out to be **one contract, one prompt** —
not one file and not one feature.

### The OpenRouter provider

Four prompts, each a narrower contract than the last:

1. *"A function that sends one chat completion to OpenRouter and returns the
   text plus token usage. Raise on any failure; the caller handles fallback."*
   Produced the shape of `complete()` — the request body, headers, the
   `urllib` call, the `usage` extraction.
2. *"Every failure mode must be the same exception type, and the caller must
   never see a partial response."* Produced the `ProviderError` funnel: HTTP
   errors, network errors, non-JSON bodies, and malformed response shapes all
   converge on one type.
3. *"Estimate the cost of a call without pretending to bill it."* Produced
   `_PRICING` and `_estimate_cost`, with the fallback rate for unlisted models.
4. *"The model id is configuration, not a constant."* Produced `get_model()`
   reading `NOVUM_REPORT_MODEL` at call time.

Splitting it this way mattered. An earlier attempt asked for the whole provider
in one prompt and produced something that worked on the happy path and swallowed
two distinct failures into one message.

### Training

Prompts here were narrower still, because the numerical parts are where a
plausible-looking answer is most expensive:

- *"Fit PCA by streaming randomized SVD over chunked passes; peak memory
  `O(n·l + D·l)`, never `O(n·D)`."* The memory bound in the prompt was
  load-bearing. Asked without it, the first version materialised the full
  design matrix.
- *"Deterministic given a seed: torch seeded, numpy seeded, batch order from a
  seeded Generator, no DataLoader worker pool."* Enumerating every source of
  nondeterminism was more effective than asking for "reproducible training".
- Config schema and the per-tier YAML were largely Bob's, with the FLOP
  accounting checked by hand against the layer shapes.

### The console

Component-level prompts against a fixed data contract — the TypeScript
interfaces in `web/src/lib/console-data.ts` were written first and pasted into
each prompt. Panels came out close to usable. Layout did not; see below.

## Where the output needed correction

**Layout under constraint.** Component internals were consistently good;
whole-screen behaviour consistently was not. The generated panels sized to their
content, which is correct in isolation and wrong inside a fixed-height flex
column — the mission-brief card rendered 1096 px tall in a 900 px viewport and
pushed the page into a scroll, breaking the one layout rule the console has.
The fix was `h-full min-h-0 overflow-hidden` on each `Card`, and it was found by
rendering the page and measuring the DOM, not by reading the code. Nothing about
the generated JSX looks wrong.

**Cross-origin assumptions.** The console calls the API on another port. The
generated fetch code was fine and there was no CORS configuration to match it,
so the briefing panel failed in every default setup. Neither half was wrong
alone; the gap was between them, which is the kind of defect that survives
per-file review.

**Optimistic error handling.** More than once the first version treated a failed
call as an empty result rather than an error — the shape that later became the
`skip_reason` codes in `llm_provider.py`, because a report saying "LLM
unavailable" tells an operator nothing they can act on. Asking specifically for
*why* a failure happened, not just *that* it happened, changed the output
materially.

**Numbers in prose.** The original report generator let the model write figures
directly, with a numeric validator downstream. That validator cannot catch a
correct number under a wrong label — "604 natural frames expired" when 604 is
the total across all classes. The rework in `core/ground/facts.py` (placeholders
in, values substituted after, any digit of the model's own rejected) is a
response to that, and it was not Bob's design.

## What it caught that a human might not have

One case is worth recording because it went the other way.

While extending the OpenRouter path, Bob flagged that `batch_upload`-style APIs
and per-item logging are a bad pair: the log line is emitted when an item is
*queued*, not when the batch *commits*, so a wholly failed upload prints exactly
like a successful one. That was raised as a general caution about the pattern.

It was noted and not acted on. The identical bug then occurred for real in
`scripts/modal_console.py` — a 136 MB file silently failed to reach a Modal
volume, the per-file logging reported success, and the entire run grid was
rebuilt against the wrong validation split before anyone noticed. The upload is
now idempotent and verifies against the remote listing rather than against
having called `put_file`.

So: a correct warning, ignored, that cost about an hour. Recorded here because
the honest version of "what did the AI catch" includes the part where the human
did not listen.

## What it was not good at

- **Anything requiring a measurement.** Every performance claim it produced was
  plausible and unverified. The FLOP counts, the cycle costs and the memory
  bounds in this repository are checked against the artifacts, not accepted.
- **Knowing when a result is uncomfortable.** Asked to summarise the
  fixed-hardware experiment, it wrote the flattering reading — the small model
  wins — and not the useful one, which is that model capacity buys nothing here
  and the entire tier comparison is measuring a compute effect. That framing had
  to be imposed.
- **Second-order effects.** That scoring *every* frame makes the PCA tier worse
  (`0.556`<!--@RAD750HW_RAD750_YIELD--> → `0.509`<!--@RAD750_ALL_SCORED_YIELD-->)
  because the cheap prefilter is a second opinion with a different bias, is not
  something any prompt here produced. It came out of running the experiment and
  disbelieving the result.

## Honest summary

Bob was most useful where the contract was crisp and the failure modes could be
enumerated in the prompt — the provider client is the clearest example, and it
needed little correction. It was least useful where correctness depended on
something outside the file being edited: layout under constraint, cross-service
configuration, and any claim that needed a number behind it.
