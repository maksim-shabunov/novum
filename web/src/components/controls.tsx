"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  HARDWARE_LABEL,
  POLICY_LABEL,
  TIER_LABEL,
  type Grid,
  type Selection,
} from "@/lib/console-data";

/**
 * Five independent controls, because their cross product is the finding.
 *
 * Not a preset picker. The interesting states are the incoherent-looking ones
 * -- an expensive model on the cheapest processor -- and a preset list would
 * quietly hide exactly those.
 */

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <Tooltip>
        <TooltipTrigger render={<span />}>
          <span className="w-fit cursor-help text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
            {label}
          </span>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs">{hint}</TooltipContent>
      </Tooltip>
      {children}
    </div>
  );
}

export function Controls({
  grid,
  selection,
  onChange,
}: {
  grid: Grid;
  selection: Selection;
  onChange: (next: Selection) => void;
}) {
  const budgets = grid.axes.budget;
  const budgetIndex = Math.max(0, budgets.indexOf(selection.budget));

  return (
    <div className="grid grid-cols-[190px_210px_1fr_170px_240px] items-end gap-5">
      <Field
        label="Flight hardware"
        hint="The processor actually flying. It sets the cycle budget per window — the budget that processor was provisioned for by its own reference model."
      >
        <Select
          value={selection.hardware}
          onValueChange={(v) => onChange({ ...selection, hardware: String(v) })}
        >
          <SelectTrigger className="h-9 w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {grid.axes.hardware.map((h) => (
              <SelectItem key={h} value={h}>
                {HARDWARE_LABEL[h] ?? h}
                <span className="ml-2 text-xs text-muted-foreground">
                  {grid.processors?.[h]?.processor}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>

      <Field
        label="Model tier"
        hint="Which novelty model is uplinked. It is charged its real cost on the chosen processor, so an expensive model simply affords fewer scores per window."
      >
        <Select
          value={selection.tier}
          onValueChange={(v) => onChange({ ...selection, tier: String(v) })}
        >
          <SelectTrigger className="h-9 w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {grid.axes.tier.map((t) => (
              <SelectItem key={t} value={t}>
                {TIER_LABEL[t] ?? t}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>

      <Field
        label="Downlink budget"
        hint="Bits per relay pass, as a share of the bits captured per window. At 25% the rover can send about a quarter of what it takes."
      >
        <div className="flex items-center gap-3 pb-1">
          <Slider
            className="flex-1"
            min={0}
            max={budgets.length - 1}
            step={1}
            value={[budgetIndex]}
            onValueChange={(v) => {
              const i = Array.isArray(v) ? v[0] : v;
              onChange({ ...selection, budget: budgets[i] });
            }}
          />
          <span className="metric w-14 shrink-0 text-right text-sm text-foreground">
            {Math.round(selection.budget * 100)}%
          </span>
        </div>
      </Field>

      <Field
        label="Adaptation"
        hint="Frozen: trained on the ground, never changes — and optimistic, since its training set spans sols the rover has not reached. Online: bootstraps and refits in flight on unlabelled frames."
      >
        <ToggleGroup
          size="sm"
          variant="outline"
          value={[selection.adaptation]}
          onValueChange={(v) => v[0] && onChange({ ...selection, adaptation: String(v[0]) })}
          className="w-full"
        >
          {grid.axes.adaptation.map((a) => (
            <ToggleGroupItem key={a} value={a} className="flex-1 capitalize">
              {a}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </Field>

      <Field
        label="Policy"
        hint="FIFO sends oldest first and pays no cycle tax. NOVUM ranks by novelty. Oracle reads ground-truth labels — it cannot run onboard and is shown only as the ceiling."
      >
        <ToggleGroup
          size="sm"
          variant="outline"
          value={[selection.policy]}
          onValueChange={(v) => v[0] && onChange({ ...selection, policy: String(v[0]) })}
          className="w-full"
        >
          {grid.axes.policy.map((p) => (
            <ToggleGroupItem key={p} value={p} className="flex-1">
              {POLICY_LABEL[p] ?? p}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </Field>
    </div>
  );
}
