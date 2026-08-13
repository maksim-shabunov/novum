"use client";

import { type ReactNode, useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Selection } from "@/lib/console-data";

/**
 * The ground-side operator briefing for whatever is currently selected.
 *
 * Written from the same decision log the panels above are drawing, by
 * `core.ground.report_gen`. When a language model is configured it arranges the
 * prose; every figure in it comes from the deterministic fact layer either way,
 * so the offline briefing is not a degraded one -- it is the same numbers with
 * plainer sentences. The badge says which, and why, because "LLM unavailable"
 * is not something an operator can act on.
 */

interface BriefResponse {
  text: string;
  mode: "llm" | "offline";
  skip_reason: string | null;
  skip_help: string | null;
  model: string | null;
}

const API = process.env.NEXT_PUBLIC_NOVUM_API ?? "http://127.0.0.1:8000";

export function BriefPanel({ selection }: { selection: Selection }) {
  const [brief, setBrief] = useState<BriefResponse | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams({
      hardware: selection.hardware,
      tier: selection.tier,
      budget: String(selection.budget),
      adaptation: selection.adaptation,
      policy: selection.policy,
    });
    let live = true;
    setFailed(false);
    fetch(`${API}/api/brief?${params}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data: BriefResponse) => live && setBrief(data))
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, [selection]);

  return (
    <Card className="flex h-full min-h-0 flex-col gap-0 overflow-hidden py-0">
      <CardHeader className="shrink-0 gap-1 border-b border-border px-4 py-3">
        <CardTitle className="flex items-center justify-between text-sm font-medium">
          <span>Mission brief</span>
          {brief ? (
            <Tooltip>
              <TooltipTrigger render={<span />}>
                <Badge
                  variant={brief.mode === "llm" ? "secondary" : "outline"}
                  className="cursor-help text-[10px] font-normal"
                >
                  {brief.mode === "llm" ? brief.model : "offline — template"}
                </Badge>
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                {brief.mode === "llm"
                  ? "A language model arranged this prose. Every figure in it was substituted from the decision log after generation — the model never writes numbers."
                  : (brief.skip_help ??
                    "Rendered directly from the decision log by the deterministic template.")}
              </TooltipContent>
            </Tooltip>
          ) : null}
        </CardTitle>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 p-0">
        <ScrollArea className="h-full">
          <div className="px-4 py-3">
            {failed ? (
              <p className="text-xs text-muted-foreground">
                The briefing service is not reachable. Everything else on this page is
                precomputed and unaffected.
              </p>
            ) : brief ? (
              <BriefText text={brief.text} />
            ) : (
              <div className="space-y-2">
                {[...Array(6)].map((_, i) => (
                  <div
                    key={i}
                    className="h-3 animate-pulse rounded bg-muted"
                    style={{ width: `${60 + ((i * 13) % 35)}%` }}
                  />
                ))}
              </div>
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

/**
 * Inline spans: `**bold**`, `` `code` `` and `*italic*`.
 *
 * The line-kind switch below owns block structure; this owns what happens
 * inside a line. Without it the operator reads `**55.6%**` with the asterisks
 * showing -- literal markup around the one figure this panel exists to present.
 *
 * Underscore emphasis is deliberately absent. `cum_science_yield` and
 * `offline_requested` are ordinary words in this text, so an intra-word
 * underscore must never open a span; the only `_..._` the generator emits wraps
 * a whole line, and `unwrapWholeLine` takes it off before this runs.
 */
const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;

function renderInline(text: string): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    if (!part) return null;
    if (part.length > 4 && part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={i} className="font-semibold text-foreground">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
      return (
        <code key={i} className="rounded bg-muted px-1 font-mono text-[0.95em]">
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.length > 2 && part.startsWith("*") && part.endsWith("*")) {
      return <em key={i}>{part.slice(1, -1)}</em>;
    }
    return part;
  });
}

/**
 * A line that is emphasis end to end -- `*Run: 27 windows...*`, and the trailing
 * `_⚠ OFFLINE REPORT ..._` -- is neither a bullet nor an inline span.
 *
 * Separating it out is what earns the inline pass its freedom to ignore
 * underscores: the offline footer wraps a code span whose identifier contains
 * one, and no general rule short of a real parser gets that right.
 */
function unwrapWholeLine(t: string): { body: string; emphasised: boolean } {
  const delim = t.startsWith("_") ? "_" : t.startsWith("*") && !t.startsWith("**") ? "*" : "";
  if (delim && t.length > 2 && t.endsWith(delim) && !t.endsWith(delim + delim)) {
    return { body: t.slice(1, -1), emphasised: true };
  }
  return { body: t, emphasised: false };
}

/**
 * The brief's own structure is figure-lines and prose-lines (see
 * `core.ground.report_gen`), so rendering is a line-kind switch, not a markdown
 * parser. Keeping the two visually distinct is the point: measured quantities
 * and interpretation should never look like the same claim.
 */
function BriefText({ text }: { text: string }) {
  const lines = text.split("\n");
  return (
    <div className="space-y-1.5">
      {lines.map((line, i) => {
        const t = line.trim();
        if (!t) return <div key={i} className="h-1.5" />;
        if (t.startsWith("#")) {
          return (
            <h3
              key={i}
              className="pt-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground"
            >
              {t.replace(/^#+\s*/, "")}
            </h3>
          );
        }
        // The delimiter must be followed by a space. `*Run: 27 windows...*` is a
        // whole-line italic, and reading it as a bullet ate its opening asterisk
        // and left the closing one on screen.
        if (/^[-*]\s/.test(t)) {
          const body = t.replace(/^[-*]\s*/, "");
          const [label, ...rest] = body.split(":");
          return (
            <div key={i} className="metric flex gap-2 text-xs leading-relaxed">
              <span className="text-natural">▪</span>
              <span>
                {rest.length ? (
                  <>
                    <span className="text-muted-foreground">{renderInline(label)}:</span>
                    <span className="text-foreground">{renderInline(rest.join(":"))}</span>
                  </>
                ) : (
                  renderInline(body)
                )}
              </span>
            </div>
          );
        }
        const { body, emphasised } = unwrapWholeLine(t);
        return (
          <p
            key={i}
            className={`text-xs leading-relaxed text-muted-foreground${emphasised ? " italic" : ""}`}
          >
            {renderInline(body)}
          </p>
        );
      })}
    </div>
  );
}
