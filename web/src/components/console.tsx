"use client";

import { useEffect, useMemo, useState } from "react";
import { BriefPanel } from "@/components/brief-panel";
import { BufferPanel } from "@/components/buffer-panel";
import { Controls } from "@/components/controls";
import { DownlinkPanel } from "@/components/downlink-panel";
import { Headline } from "@/components/headline";
import { TimelinePanel } from "@/components/timeline-panel";
import { frameTitle } from "@/components/frame-tile";
import { Badge } from "@/components/ui/badge";
import {
  HARDWARE_LABEL,
  bufferAt,
  getCell,
  mostLoadedWindow,
  type Frame,
  type Grid,
  type Mission,
  type Selection,
} from "@/lib/console-data";

/**
 * The console.
 *
 * Every visible number comes from `public/data/grid.json`, which
 * `scripts/build_console.py` produced by replaying the full cross product ahead
 * of time. Changing a control is a dictionary lookup, so the slider tracks the
 * pointer instead of queueing a simulation, and the whole thing runs on a host
 * with no Python process at all.
 */
export function Console() {
  const [mission, setMission] = useState<Mission | null>(null);
  const [grid, setGrid] = useState<Grid | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [picked, setPicked] = useState<number | null>(null);
  const [hovered, setHovered] = useState<Frame | null>(null);

  useEffect(() => {
    Promise.all([
      fetch("/data/mission.json").then((r) => r.json()),
      fetch("/data/grid.json").then((r) => r.json()),
    ])
      .then(([m, g]: [Mission, Grid]) => {
        setMission(m);
        setGrid(g);
        setSelection(g.default);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const cell = grid && selection ? getCell(grid, selection) : undefined;

  // Land on the fullest window: a buffer argued with six thumbnails is not
  // argued at all. The user's own pick wins while it stands, and changing a
  // control clears it (see `onChange` below) so the next configuration gets its
  // own fullest window.
  //
  // Derived rather than pushed into state by an effect. The effect version had
  // to re-enter render to place the initial window, and "which window are we
  // looking at" then had two owners that could disagree for a frame.
  const autoWindow = useMemo(
    () => (mission && cell ? mostLoadedWindow(mission, cell) : null),
    [mission, cell],
  );
  const windowIndex = picked ?? autoWindow;

  const current = useMemo(() => {
    if (!mission || !cell || windowIndex === null) return null;
    return bufferAt(mission, cell, windowIndex);
  }, [mission, cell, windowIndex]);

  if (error) {
    return (
      <Centered>
        <p className="text-sm text-lost">Could not load the precomputed run grid.</p>
        <p className="mt-2 text-xs text-muted-foreground">
          Run <code className="metric">make console</code> to build{" "}
          <code className="metric">web/public/data/</code>. {error}
        </p>
      </Centered>
    );
  }

  if (!mission || !grid || !selection || windowIndex === null || !cell || !current) {
    return (
      <Centered>
        <div className="h-1 w-56 overflow-hidden rounded bg-muted">
          <div className="h-full w-1/3 animate-pulse rounded bg-natural" />
        </div>
        <p className="mt-3 text-xs text-muted-foreground">
          Loading the precomputed mission grid…
        </p>
      </Centered>
    );
  }

  const record = cell.windows.find((w) => w.w === windowIndex);
  const missionWindow = mission.windows.find((w) => w.w === windowIndex);

  return (
    <main className="mx-auto flex h-screen max-w-[1680px] flex-col gap-3 px-5 py-4">
      <header className="flex shrink-0 items-baseline gap-4">
        <h1 className="text-sm font-semibold uppercase tracking-[0.2em]">
          NOVUM<span className="ml-2 text-muted-foreground">mission control</span>
        </h1>
        <p className="text-xs text-muted-foreground">
          {mission.n_frames} frames · sols {mission.sol_min}–{mission.sol_max} ·{" "}
          {mission.composition.natural} natural science ·{" "}
          {mission.windows.length} relay passes
        </p>
        <div className="ml-auto flex items-center gap-2">
          {hovered ? (
            <Badge variant="outline" className="metric text-[10px] font-normal">
              {frameTitle(hovered)}
            </Badge>
          ) : (
            <span className="text-[11px] text-muted-foreground">
              {HARDWARE_LABEL[selection.hardware]} ·{" "}
              {grid.processors?.[selection.hardware]?.processor}
            </span>
          )}
        </div>
      </header>

      <section className="shrink-0 rounded-lg border border-border bg-card/40 px-5 py-4">
        <Controls
          grid={grid}
          selection={selection}
          onChange={(next) => {
            setSelection(next);
            setPicked(null);
          }}
        />
      </section>

      <section className="shrink-0">
        <Headline grid={grid} selection={selection} cell={cell} />
      </section>

      <section className="grid min-h-0 flex-1 grid-cols-[1.15fr_0.85fr_1.1fr] gap-3">
        <BufferPanel
          mission={mission}
          buffer={current}
          record={record}
          onHover={setHovered}
        />
        <div className="flex min-h-0 flex-col gap-3">
          <div className="min-h-0 flex-[1.1]">
            <DownlinkPanel
              mission={mission}
              buffer={current}
              record={record}
              onHover={setHovered}
            />
          </div>
          <div className="min-h-0 flex-1">
            <BriefPanel selection={selection} />
          </div>
        </div>
        <TimelinePanel
          grid={grid}
          mission={mission}
          selection={selection}
          cell={cell}
          window={windowIndex}
          onWindow={setPicked}
        />
      </section>

      <footer className="shrink-0 text-[10px] text-muted-foreground">
        Window {windowIndex}
        {missionWindow
          ? ` · sols ${missionWindow.first_sol}–${missionWindow.last_sol}`
          : ""}{" "}
        · every figure replayed offline from the Mastcam novelty archive
        (Kerner et al., CC-BY-4.0); thumbnails are a fixed false-colour
        composite, not a calibrated product.
      </footer>
    </main>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <main className="flex h-screen flex-col items-center justify-center">
      {children}
    </main>
  );
}
