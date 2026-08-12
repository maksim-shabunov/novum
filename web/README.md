# NOVUM mission control

A ground-station console. Left: what the rover captured. Right: what reached
Earth. Between them, the decision.

```bash
make console       # precompute public/data/ (needs the dataset; output is committed)
make web-install
make web           # http://localhost:3000
make serve         # optional: the briefing endpoint, in another shell
```

## What it reads

Everything on screen comes from three files under `public/data/`, produced by
`scripts/build_console.py` and committed to the repository:

| file | what it is |
|---|---|
| `atlas.png` | every mission frame as one sprite sheet, 32 px tiles |
| `mission.json` | the frame stream — sol, class, group, bit cost, atlas position |
| `grid.json` | 324 precomputed runs: the full control cross product |

No simulation runs in the browser and none runs on request. Changing a control
is a dictionary lookup into `grid.json`, which is why the slider tracks the
pointer instead of queueing a replay, and why the app runs on a host with no
Python process at all.

The one thing that is not precomputed is the briefing prose, which
`GET /api/brief` renders from the same runs. It defaults to the deterministic
template and needs no API key. If the API is unreachable the panel says so and
nothing else on the page is affected.

## Layout

```
src/
  app/            layout (dark, no theme toggle) and the single page
  components/
    console.tsx        state, data loading, the 1440px grid
    controls.tsx       the five independent controls
    headline.tsx       the result, above the fold, plus the can't-run alert
    buffer-panel.tsx   captured frames, marked when they expire
    downlink-panel.tsx what was selected, in order
    timeline-panel.tsx yield curve for all three policies + window scrubber
    brief-panel.tsx    the operator briefing and its offline badge
    frame-tile.tsx     one atlas tile
    ui/                shadcn components, vendored — read them, they are ours now
  lib/console-data.ts  types and the buffer derivation
```

`bufferAt()` reconstructs what the rover was holding at any window: everything
captured so far minus everything sent or lost. Storing that per window per cell
would have added megabytes to describe something reconstructible in one pass.

## Conventions

- **Dark only.** An operator display that can be switched to white is a
  marketing page.
- **No webfont.** `next/font/google` needs the network at build time, and the
  acceptance test is a clean machine. System stacks never fail.
- **Tabular numerals** on every metric, so figures do not jitter when a slider
  moves.
- **Three data hues, fixed by meaning**: amber for natural science, slate for
  rover hardware, grey for typical terrain. Cyan marks the current window, red
  marks loss. The thumbnails are the only other colour on screen.
- **Nothing scrolls to reach the result.** The thumbnail grids scroll inside
  their own panels; the headline never moves.

## Rules

This directory talks to `api/` over HTTP and reads its own static data. It must
not read `artifacts/` or `data/` from disk, and it must not know that a training
stack exists.
