"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { FrameTile, frameTitle } from "@/components/frame-tile";
import type { BufferState, Frame, Mission, WindowRecord } from "@/lib/console-data";

/**
 * What the rover is holding, and what it just lost.
 *
 * The left half of the argument. Every tile is a real 64x64 Mastcam frame the
 * rover actually captured; the ring says what class it is, and a struck-through
 * tile is one that aged out of the buffer without ever being sent. A judge
 * should be able to see, without reading anything, that the amber-ringed frames
 * are the ones worth the bits.
 */
export function BufferPanel({
  mission,
  buffer,
  record,
  onHover,
}: {
  mission: Mission;
  buffer: BufferState;
  record: WindowRecord | undefined;
  onHover: (f: Frame | null) => void;
}) {
  const byIndex = mission.frames;
  const lost = new Set(buffer.lost);
  // Losses first: a frame that just expired is the news in this panel.
  const tiles: { frame: Frame; state: "held" | "lost" }[] = [
    ...buffer.lost.map((i) => ({ frame: byIndex[i], state: "lost" as const })),
    ...buffer.held.map((i) => ({ frame: byIndex[i], state: "held" as const })),
  ].filter((t) => t.frame && (t.state !== "held" || !lost.has(t.frame.i)));

  const naturalHeld = buffer.held.filter((i) => byIndex[i]?.group === "natural").length;

  return (
    <Card className="flex h-full min-h-0 flex-col gap-0 overflow-hidden py-0">
      <CardHeader className="shrink-0 gap-1 border-b border-border px-4 py-3">
        <CardTitle className="flex items-baseline justify-between text-sm font-medium">
          <span>Onboard buffer</span>
          <span className="metric text-xs font-normal text-muted-foreground">
            {buffer.held.length} held · {naturalHeld} natural
          </span>
        </CardTitle>
        <p className="text-xs text-muted-foreground">
          Captured, awaiting a downlink slot. Struck-through frames aged out unsent.
        </p>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 p-0">
        <ScrollArea className="h-full">
          <div className="flex flex-wrap content-start gap-1 p-3">
            {tiles.length === 0 ? (
              <p className="p-4 text-xs text-muted-foreground">
                Buffer empty at this window.
              </p>
            ) : (
              tiles.map(({ frame, state }) => (
                <FrameTile
                  key={`${state}-${frame.i}`}
                  frame={frame}
                  mission={mission}
                  size={40}
                  state={state}
                  title={frameTitle(frame)}
                  onHover={onHover}
                />
              ))
            )}
          </div>
        </ScrollArea>
      </CardContent>
      {record ? (
        <div className="flex shrink-0 items-center gap-2 border-t border-border px-4 py-2 text-xs text-muted-foreground">
          <Badge variant="outline" className="metric text-[10px]">
            +{record.arrived} arrived
          </Badge>
          {record.expired + record.evicted > 0 ? (
            <Badge
              variant="outline"
              className="metric border-lost/40 text-[10px] text-lost"
            >
              −{record.expired + record.evicted} lost
            </Badge>
          ) : null}
          <span className="metric ml-auto">
            {record.scored} scored / {record.unscored} unaffordable
          </span>
        </div>
      ) : null}
    </Card>
  );
}
