# Three-minute demo script

Shot by shot, with the words to say and the exact interactions. Total 3:00.

**Before recording**

```bash
docker compose -f docker/docker-compose.yml up      # or: make web + make serve
```

- Browser at **1440 × 900**, no bookmarks bar, no devtools. The console is built
  to fit that viewport exactly; anything narrower reflows the headline row.
- Load <http://localhost:3000> and **let it settle** — `grid.json` is 3.9 MB and
  the first paint after it lands is what you want on camera, not the loading bar.
- Confirm the default state reads: **rad750 / rad750 / 25% / Frozen / NOVUM**.
- Hard-refresh once before the take. The window scrubber auto-selects the
  fullest window, and you want that deterministic.

Screen recording only, cursor visible. No zooming — the type is sized for this.

---

## 0:00–0:30 · The problem

**Show:** the console as loaded. Do not touch anything. Let the buffer grid sit
on screen for a full five seconds before speaking.

> "This is what a Mars rover captured on one traverse — 856 frames of Mastcam
> imagery. On the right is what actually reached Earth. Six frames, this window.
>
> The bottleneck was never the camera. An orbiter is overhead for a few minutes,
> and the bits that fit in that pass are the only science anyone gets that day.
> Everything else waits in a buffer and some of it ages out unsent.
>
> So the question isn't 'can a model classify Martian terrain'. It's: which
> frames do you send?"

**Must be visible:** the asymmetry between the two panels — a hundred and
sixty-odd thumbnails on the left, six on the right. That contrast *is* the
problem statement; do not narrate over a scrolled or partial view of it.

**Do not linger on:** the controls. They are not explained yet and reading them
aloud costs fifteen seconds you need later.

---

## 0:30–1:15 · Selection concentrates on science

**Do:** drag the **Downlink budget** slider from 25% down to **5%**, slowly,
about two seconds. Then back up to 25%. Then leave it at **15%**.

> "Every one of these is a real frame, and they're not equal. The amber rings
> are natural science — veins, meteorites, broken rock. The blue ones are the
> rover's own hardware: drill holes, wheel tracks. Novel to a detector, worthless
> to a geologist.
>
> Watch what happens when I squeeze the downlink."

*(during the drag)*

> "As the budget falls, the selection concentrates. The rover-made frames drop
> out first. That's not a rule anyone wrote — it's ranking by novelty against
> terrain the model has already seen, and the rover's own parts stop being
> surprising quickly.
>
> At an identical bit budget: sending oldest-first delivers fifteen percent of
> the natural science. This delivers fifty-five.
>
> *(live: FIFO **15.4%**<!--@FIFO_YIELD-->, NOVUM **55.6%**<!--@NOVUM_YIELD-->)*"

**Must be visible:** the headline row throughout — **94**<!--@NOVUM_NATURAL_SENT--> of
**169**<!--@NATURAL_TOTAL--> natural frames, **3.6×**<!--@NOVUM_VS_FIFO_RATIO--> versus FIFO, and the
*bits on rover hardware* figure falling as the budget tightens.

**Do not linger on:** the individual thumbnails. Resist hovering; the tooltip is
useful in person and dead time on video.

---

## 1:15–1:50 · The model that cannot run

The strongest interaction in the demo. Give it room.

**Do:** set **Downlink budget** back to **25%**. Then open **Model tier** and
select **snapdragon (conv-AE)**. Say nothing for two full seconds after the
screen updates.

> "Same downlink. Same mission. I've only changed which model is flying — to the
> largest one, on the RAD750 that Curiosity actually carries.
>
> Nothing reached Earth.
>
> One novelty score on that processor costs a hundred and forty-seven million
> cycles. The window budget affords zero point five six of them — less than one
> frame.
>
> So it scores nothing, ranks nothing, and the downlink sits idle while
> the buffer fills and ages out.
>
> *(live figures on the banner: **147,628,032**<!--@RAD750HW_SNAPDRAGON_CYCLES-->
> cycles per score, **0.56**<!--@RAD750HW_SNAPDRAGON_SCORES_2DP--> affordable)*
>
> This is the whole argument. Accuracy you cannot afford to compute is not
> accuracy. Every paper that reports ROC AUC on a fixed test set and stops
> cannot see this."

**Must be visible:** the red alert banner — *"This model cannot run on this
hardware"* — showing **147,628,032**<!--@RAD750HW_SNAPDRAGON_CYCLES--> cycles
per score and **0.56**<!--@RAD750HW_SNAPDRAGON_SCORES_2DP--> scores affordable; the `0/169` headline; the
empty downlink panel reading *"Nothing was transmitted this window"*; and the
window timeline strip turning entirely red for cycles-limited.

**Do not linger on:** switching back. Change **Model tier** to **rad750** and
move on within two seconds — the recovery is not the point, the failure is.

---

## 1:50–2:20 · Frozen versus online

**Do:** click **Adaptation → Online**. Let the chart redraw. Then click
**Frozen** and **Online** once more, about a second apart, so the two curves can
be compared directly.

> "So far the model was trained on the ground before launch. That's optimistic —
> its training set spans sols the rover hasn't reached yet.
>
> This one arrives knowing nothing about the terrain. It bootstraps on the first
> sols and refits in flight, on unlabelled frames.
>
> Watch the curve. It starts at zero and climbs. It reaches forty-six percent
> against FIFO's fifteen — with no prior knowledge of the body it's looking at.
>
> *(live: online **0.462**<!--@ONLINE_YIELD-->, FIFO **0.154**<!--@FIFO_YIELD_DECIMAL-->)*
>
> A detector arriving somewhere nobody has been doesn't need to be told what's
> interesting. It needs to learn what's ordinary. That's the curve."

**Must be visible:** the cumulative-yield chart, specifically the **cold start** —
the amber line flat near zero for the first windows, then climbing, against the
frozen version which jumps early.

**Do not linger on:** the dashed oracle line. Explaining it costs more than it
adds here.

---

## 2:20–2:45 · The ground-side brief

**Do:** scroll the **Mission brief** panel down one screen, slowly. Do not click
anything else.

> "The rover doesn't send this. It sends a decision log — a few hundred bytes per
> window. Counts, budgets, which constraint bound.
>
> A language model on the ground turns that into an operator briefing. And it
> writes no numbers at all. Every figure here was substituted in afterwards from
> the decision log; the model only chooses which facts belong in which section
> and writes the sentences between them.
>
> If it writes a digit of its own, the whole response is thrown away and a
> deterministic template is published instead. It works with no API key — that
> badge says which one you're reading."

**Must be visible:** the `offline — template` badge, and the figure lines with
their labels attached — *"Frames expired unsent: 604 across the mission, all
classes — not only natural"*. That phrasing is the point: the label travels with
the number.

**Do not linger on:** the prose quality. It is a small model and the sentences
are ordinary; the guarantee is about the figures, not the writing.

---

## 2:45–3:00 · Architecture and result

**Do:** leave the console on screen at the default state. No further clicks.

> "Novelty scoring onboard under two budgets — bits and cycles, both of which
> actually bind. A decision log of a few hundred bytes crossing the link. The
> language model on the ground where bandwidth is free.
>
> Three and a half times the science, at an identical bit budget, on real Mars
> imagery and a processor that really flies.
>
> Everything you've seen is precomputed and committed. Clone it and run
> `docker compose up` — no dataset, no training, no key."

**Must be visible:** the headline row at full opacity for the final five seconds:
`94/169`, `3.6×`, `88% of the oracle ceiling`.

---

## Timing notes

| Segment | Budget | Hard limit |
|---|---|---|
| Problem | 0:30 | do not exceed — the payoff is at 1:15 |
| Selection | 0:45 | can lose 10s if needed |
| **Model collapse** | 0:35 | **protect this**; take time from anywhere else |
| Frozen vs online | 0:30 | can lose 10s |
| Brief | 0:25 | first to cut if over |
| Close | 0:15 | fixed |

If the recording runs long, cut the brief segment to fifteen seconds and mention
the guarantee in one sentence. Do not shorten the model-collapse segment; the two
seconds of silence after the screen updates is doing more work than any sentence
in the script.

## If something goes wrong on the take

- **Brief panel says "not reachable"** — the API container is down. Everything
  else is precomputed and unaffected; you can record segments 1–4 and 6 without
  it, and re-record the brief separately.
- **Console stuck on the loading bar** — `grid.json` did not load. Check
  `web/public/data/` exists in the image; a build without it fails the guard in
  `docker/Dockerfile.web`.
- **Numbers differ from this script** — the grid was regenerated. Read the live
  values; do not read this script over a different screen. `make check-figures`
  tells you what the committed artifacts actually say.
