"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { TriangleAlert } from "lucide-react";
import {
  HARDWARE_LABEL,
  POLICY_LABEL,
  getCellForPolicy,
  type Grid,
  type RunCell,
  type Selection,
} from "@/lib/console-data";

/**
 * The result, above everything else and without scrolling.
 *
 * A visitor who touches nothing must still leave knowing the one number: at an
 * identical bit budget, ranking by novelty delivers several times the science
 * that sending oldest-first does. Everything below this row is the evidence.
 */

function Figure({
  value,
  label,
  sub,
  hint,
  tone = "default",
}: {
  value: string;
  label: string;
  sub?: string;
  hint: string;
  tone?: "default" | "primary" | "muted";
}) {
  return (
    <Tooltip>
      <TooltipTrigger render={<span />}>
        <div className="flex cursor-help flex-col justify-center px-5">
          <div
            className={
              "metric text-[2.1rem] leading-none " +
              (tone === "primary"
                ? "text-natural"
                : tone === "muted"
                  ? "text-muted-foreground"
                  : "text-foreground")
            }
          >
            {value}
          </div>
          <div className="mt-1.5 text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
            {label}
          </div>
          {sub ? (
            <div className="metric mt-0.5 text-xs text-muted-foreground/80">{sub}</div>
          ) : null}
        </div>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{hint}</TooltipContent>
    </Tooltip>
  );
}

export function Headline({
  grid,
  selection,
  cell,
}: {
  grid: Grid;
  selection: Selection;
  cell: RunCell;
}) {
  const fifo = getCellForPolicy(grid, selection, "fifo");
  const oracle = getCellForPolicy(grid, selection, "oracle");

  const total = cell.n_natural_total || 1;
  const delivered = cell.n_sent_natural;
  const ratio = fifo && fifo.n_sent_natural > 0 ? delivered / fifo.n_sent_natural : null;
  const ceiling = oracle && oracle.n_sent_natural > 0
    ? delivered / oracle.n_sent_natural
    : null;

  const bound = cell.windows.length
    ? cell.windows.filter((w) => w.bound === "cycles").length >
      cell.windows.length / 2
      ? "cycles"
      : "bits"
    : "none";

  // The model cannot afford to look at even one frame per window. This is the
  // failure the demo is built to make reachable, so it is stated, not implied.
  const starved =
    cell.scores_affordable_per_window !== null &&
    cell.scores_affordable_per_window < 1 &&
    selection.policy === "score_first";

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-[1.35fr_1fr_1fr_1fr_1.1fr] divide-x divide-border rounded-lg border border-border bg-card/60">
        <Figure
          tone="primary"
          value={`${delivered}/${total}`}
          label="Natural science delivered"
          sub={`${((delivered / total) * 100).toFixed(1)}% of the mission's natural frames`}
          hint="Frames of genuine Mars novelty — veins, meteorites, broken rock — that reached the ground. The denominator is every natural frame the rover captured across the whole mission."
        />
        <Figure
          value={ratio ? `${ratio.toFixed(1)}×` : "—"}
          label="versus FIFO"
          sub={fifo ? `FIFO delivered ${fifo.n_sent_natural}` : undefined}
          hint="Against sending oldest-first at an identical bit budget. FIFO pays no cycle tax at all, which is what makes it the honest baseline rather than a strawman."
        />
        <Figure
          tone="muted"
          value={ceiling ? `${(ceiling * 100).toFixed(0)}%` : "—"}
          label="of the oracle ceiling"
          sub={oracle ? `oracle delivered ${oracle.n_sent_natural}` : undefined}
          hint="The oracle reads ground-truth labels and cannot run onboard. It is the most any selector could deliver under this bit budget — the gap is what is left on the table."
        />
        <Figure
          value={`${(cell.wasted_bit_share * 100).toFixed(0)}%`}
          label="Bits on rover hardware"
          sub="downlink spent on the rover's own parts"
          hint="Share of transmitted bits spent on frames of the rover's own hardware. Every one of those bits is a natural-science frame that stayed onboard."
        />
        <div className="flex flex-col justify-center gap-2 px-5">
          <div className="text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
            Binding constraint
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge
              variant={bound === "cycles" ? "destructive" : "secondary"}
              className="font-mono text-[11px]"
            >
              {bound}
            </Badge>
            <span className="text-xs text-muted-foreground">
              in most of {cell.windows.length} windows
            </span>
          </div>
          <div className="metric text-xs text-muted-foreground">
            {cell.scores_affordable_per_window !== null
              ? `${cell.scores_affordable_per_window.toFixed(1)} scores/window affordable`
              : "compute unbounded"}
          </div>
        </div>
      </div>

      {starved ? (
        <Alert variant="destructive" className="py-2.5">
          <TriangleAlert className="size-4" />
          <AlertTitle className="text-sm">
            This model cannot run on this hardware
          </AlertTitle>
          <AlertDescription className="text-xs">
            One novelty score costs{" "}
            <span className="metric">
              {cell.cycles_per_score.toLocaleString()}
            </span>{" "}
            cycles on the {HARDWARE_LABEL[selection.hardware] ?? selection.hardware}, and
            the window budget affords{" "}
            <span className="metric">
              {cell.scores_affordable_per_window?.toFixed(2)}
            </span>{" "}
            of them. The prefilter promotes frames nothing can score, so{" "}
            {POLICY_LABEL[selection.policy]} has nothing to rank and the downlink goes
            unused — {cell.n_natural_never_scored} natural frames were never looked at.
          </AlertDescription>
        </Alert>
      ) : null}
    </div>
  );
}
