/**
 * The shape of everything the console reads, and the derivations it does.
 *
 * Nothing here computes a simulation. `scripts/build_console.py` replays the
 * whole cross product ahead of time and ships the result in `public/data/`, so
 * dragging the downlink slider is an object lookup rather than a mission
 * replay. What this module does compute is the buffer contents, which are a
 * pure function of what arrived, what was sent and what was lost -- storing
 * them per window per cell would have multiplied the payload for no gain.
 */

export type Group = "natural" | "rover" | "typical" | "excluded";

export interface Frame {
  i: number;
  sol: number;
  group: Group;
  cls: string;
  bits: number;
  /** Column and row of this frame's tile in atlas.png. */
  ax: number;
  ay: number;
}

export interface MissionWindow {
  w: number;
  first_sol: number;
  last_sol: number;
  /** Frames captured during this window. Identical for every run. */
  arrived: number[];
}

export interface Mission {
  n_frames: number;
  composition: Record<string, number>;
  sol_min: number;
  sol_max: number;
  atlas: { columns: number; rows: number; width: number; height: number };
  frames: Frame[];
  windows: MissionWindow[];
}

export interface WindowRecord {
  w: number;
  sent: number;
  arrived: number;
  buffered: number;
  scored: number;
  unscored: number;
  expired: number;
  evicted: number;
  bound: "bits" | "cycles" | "both" | "neither";
  bits_used: number;
  bits_budget: number;
  cycles_used: number;
  cycles_budget: number;
  nat: number;
  rov: number;
  typ: number;
  cum_nat: number;
  cum_avail: number;
  cum_yield: number;
  recall: number | null;
  refit: boolean;
  /** Mission indices transmitted this window, in the order chosen. */
  sel: number[];
  /** Mission indices that aged out or were evicted without ever being sent. */
  lost: number[];
}

export interface RunCell {
  science_yield: number;
  n_sent: number;
  n_sent_natural: number;
  n_natural_total: number;
  n_expired: number;
  n_expired_natural: number;
  wasted_bit_share: number;
  precision_natural: number;
  bits_used: number;
  bits_available: number;
  prefilter_recall_natural: number;
  n_natural_never_scored: number;
  n_unscored: number;
  n_refits: number;
  cycles_per_score: number;
  scores_affordable_per_window: number | null;
  windows: WindowRecord[];
}

export interface Selection {
  hardware: string;
  tier: string;
  budget: number;
  adaptation: string;
  policy: string;
}

export interface Grid {
  axes: {
    hardware: string[];
    tier: string[];
    budget: number[];
    adaptation: string[];
    policy: string[];
  };
  processors: Record<
    string,
    { processor: string; cycles_per_flop: number; reference_cycles_per_score: number }
  >;
  default: Selection;
  cells: Record<string, RunCell>;
}

/** Must match `cell_key` in scripts/build_console.py. */
export function cellKey(s: Selection): string {
  return [s.hardware, s.tier, formatBudgetKey(s.budget), s.adaptation, s.policy].join("|");
}

/** Python's `%g`: the shortest round-trip form, no trailing zeros. */
export function formatBudgetKey(b: number): string {
  return String(parseFloat(b.toPrecision(6)));
}

export function getCell(grid: Grid, s: Selection): RunCell | undefined {
  return grid.cells[cellKey(s)];
}

/** The same configuration under a different policy, for side-by-side counters. */
export function getCellForPolicy(
  grid: Grid,
  s: Selection,
  policy: string,
): RunCell | undefined {
  return getCell(grid, { ...s, policy });
}

// ---------------------------------------------------------------------------
// Derivations
// ---------------------------------------------------------------------------

export interface BufferState {
  /** Frames held onboard at the end of this window, awaiting a later pass. */
  held: number[];
  /** Frames transmitted this window, in the order the policy chose them. */
  sent: number[];
  /** Frames that left the buffer this window without ever being sent. */
  lost: number[];
}

/**
 * What the rover was holding after window `w`.
 *
 * Everything captured up to and including this window, minus everything ever
 * transmitted, minus everything that aged out or was evicted. Derived rather
 * than stored: the buffer runs to 170 frames and persisting it for 324 cells
 * would have added megabytes to describe something reconstructible in a pass.
 */
export function bufferAt(
  mission: Mission,
  cell: RunCell,
  w: number,
): BufferState {
  const gone = new Set<number>();
  let sent: number[] = [];
  let lost: number[] = [];

  for (const rec of cell.windows) {
    if (rec.w > w) break;
    for (const i of rec.sel) gone.add(i);
    for (const i of rec.lost) gone.add(i);
    if (rec.w === w) {
      sent = rec.sel;
      lost = rec.lost;
    }
  }

  const held: number[] = [];
  for (const win of mission.windows) {
    if (win.w > w) break;
    for (const i of win.arrived) if (!gone.has(i)) held.push(i);
  }
  return { held, sent, lost };
}

/** Cumulative natural frames delivered, window by window. For the yield curve. */
export function yieldSeries(cell: RunCell | undefined): {
  w: number;
  delivered: number;
  captured: number;
  yield: number;
}[] {
  if (!cell) return [];
  return cell.windows.map((r) => ({
    w: r.w,
    delivered: r.cum_nat,
    captured: r.cum_avail,
    yield: r.cum_avail ? r.cum_nat / r.cum_avail : 0,
  }));
}

/**
 * The window a first-time visitor should land on.
 *
 * The one holding the most frames: the buffer grid is the argument, and an
 * argument made with six thumbnails is not made at all. Self-tuning, so the
 * default stays sensible in cells where the mission plays out differently.
 */
export function mostLoadedWindow(mission: Mission, cell: RunCell | undefined): number {
  if (!cell || cell.windows.length === 0) return 0;
  let best = cell.windows[0];
  for (const r of cell.windows) if (r.buffered > best.buffered) best = r;
  return best.w;
}

export const GROUP_LABEL: Record<string, string> = {
  natural: "natural science",
  rover: "rover hardware",
  typical: "typical terrain",
  excluded: "excluded",
};

export const POLICY_LABEL: Record<string, string> = {
  fifo: "FIFO",
  score_first: "NOVUM",
  oracle: "Oracle",
};

export const HARDWARE_LABEL: Record<string, string> = {
  rad750: "RAD750",
  myriad: "Myriad 2",
  snapdragon: "Snapdragon",
};

export const TIER_LABEL: Record<string, string> = {
  rad750: "rad750 (PCA)",
  myriad: "myriad (conv-AE)",
  snapdragon: "snapdragon (conv-AE)",
};
