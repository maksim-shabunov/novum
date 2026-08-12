"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { cn } from "@/lib/utils";
import {
  POLICY_LABEL,
  getCellForPolicy,
  yieldSeries,
  type Grid,
  type Mission,
  type RunCell,
  type Selection,
} from "@/lib/console-data";

/**
 * The mission over time, and the scrubber that drives every other panel.
 *
 * The curve is cumulative science delivered against natural frames captured so
 * far. In online mode the cold start is the point: the model has seen nothing,
 * refits on what arrives, and the curve climbs from near zero. Plotting all
 * three policies together means the gap is visible without switching anything.
 *
 * The strip under the chart is one cell per window, coloured by which budget
 * bound it. Click a window; everything else follows.
 */

const chartConfig = {
  novum: { label: "NOVUM", color: "var(--natural)" },
  fifo: { label: "FIFO", color: "var(--rover)" },
  oracle: { label: "Oracle", color: "var(--typical)" },
} satisfies ChartConfig;

export function TimelinePanel({
  grid,
  mission,
  selection,
  cell,
  window: currentWindow,
  onWindow,
}: {
  grid: Grid;
  mission: Mission;
  selection: Selection;
  cell: RunCell;
  window: number;
  onWindow: (w: number) => void;
}) {
  // Sols live on the mission stream, not the run record: the same window spans
  // the same sols whatever policy is selected.
  const sols = new Map(mission.windows.map((w) => [w.w, [w.first_sol, w.last_sol]]));
  const here = cell.windows.find((w) => w.w === currentWindow);
  const novum = yieldSeries(getCellForPolicy(grid, selection, "score_first"));
  const fifo = yieldSeries(getCellForPolicy(grid, selection, "fifo"));
  const oracle = yieldSeries(getCellForPolicy(grid, selection, "oracle"));

  const byWindow = new Map<number, Record<string, number>>();
  const merge = (series: ReturnType<typeof yieldSeries>, key: string) => {
    for (const p of series) {
      const row = byWindow.get(p.w) ?? { w: p.w };
      row[key] = p.yield * 100;
      byWindow.set(p.w, row);
    }
  };
  merge(novum, "novum");
  merge(fifo, "fifo");
  merge(oracle, "oracle");
  const data = [...byWindow.values()].sort((a, b) => a.w - b.w);

  return (
    <Card className="flex h-full min-h-0 flex-col gap-0 overflow-hidden py-0">
      <CardHeader className="shrink-0 gap-1 border-b border-border px-4 py-3">
        <CardTitle className="flex items-baseline justify-between text-sm font-medium">
          <span>Cumulative science yield</span>
          <span className="metric text-xs font-normal text-muted-foreground">
            {here
              ? `window ${currentWindow} · ${here.cum_nat}/${here.cum_avail} natural so far`
              : `window ${currentWindow}`}
          </span>
        </CardTitle>
        <p className="text-xs text-muted-foreground">
          Delivered as a share of natural frames captured so far — the denominator
          grows all mission, so early windows read high off a handful of frames.
        </p>
      </CardHeader>

      <CardContent className="min-h-0 flex-1 px-3 py-3">
        <ChartContainer config={chartConfig} className="h-full w-full">
          <LineChart data={data} margin={{ left: 0, right: 10, top: 10, bottom: 0 }}>
            <CartesianGrid vertical={false} strokeOpacity={0.12} />
            <XAxis
              dataKey="w"
              tickLine={false}
              axisLine={false}
              tickMargin={6}
              fontSize={10}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              width={40}
              fontSize={10}
              domain={[0, 100]}
              tickFormatter={(v) => `${v}%`}
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  labelFormatter={(v) => `Window ${v}`}
                  formatter={(value, name) => [
                    `${Number(value).toFixed(1)}%  `,
                    chartConfig[name as keyof typeof chartConfig]?.label ?? name,
                  ]}
                />
              }
            />
            <ReferenceLine
              x={currentWindow}
              stroke="var(--current)"
              strokeWidth={1.5}
              strokeDasharray="3 3"
            />
            <Line
              dataKey="oracle"
              stroke="var(--color-oracle)"
              strokeWidth={1.5}
              strokeDasharray="4 3"
              dot={false}
              isAnimationActive={false}
            />
            <Line
              dataKey="fifo"
              stroke="var(--color-fifo)"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              dataKey="novum"
              stroke="var(--color-novum)"
              strokeWidth={2.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ChartContainer>
      </CardContent>

      <div className="shrink-0 border-t border-border px-4 py-3">
        <div className="mb-1.5 flex items-center justify-between text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
          <span>Window timeline — click to inspect</span>
          <span className="flex items-center gap-3">
            <LegendDot className="bg-natural" label="bits-limited" />
            <LegendDot className="bg-lost" label="cycles-limited" />
          </span>
        </div>
        <div className="flex gap-[3px]">
          {cell.windows.map((rec) => (
            <button
              key={rec.w}
              type="button"
              onClick={() => onWindow(rec.w)}
              title={`Window ${rec.w} — sols ${sols.get(rec.w)?.[0] ?? "?"}–${
                sols.get(rec.w)?.[1] ?? "?"
              }: ${rec.sent} sent, ${rec.expired + rec.evicted} lost, ${
                rec.bound
              }-limited`}
              className={cn(
                "h-7 flex-1 rounded-[2px] transition-opacity hover:opacity-100",
                rec.bound === "cycles" ? "bg-lost/70" : "bg-natural/70",
                rec.w === currentWindow
                  ? "opacity-100 ring-2 ring-current ring-offset-1 ring-offset-card"
                  : "opacity-45",
              )}
            >
              <span className="sr-only">Window {rec.w}</span>
            </button>
          ))}
        </div>
      </div>
    </Card>
  );
}

function LegendDot({ className, label }: { className: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={cn("size-2 rounded-[2px]", className)} />
      {label}
    </span>
  );
}

export { POLICY_LABEL };
