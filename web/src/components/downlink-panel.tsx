"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { FrameTile, frameTitle } from "@/components/frame-tile";
import type { BufferState, Frame, Mission, WindowRecord } from "@/lib/console-data";

/**
 * What reached Earth this window, in the order the policy chose it.
 *
 * The right half of the argument, and deliberately the smaller panel: a relay
 * pass carries six to nine frames out of a buffer of a hundred and seventy. The
 * asymmetry between this panel and the buffer beside it is the problem
 * statement, so the two are shown at the same tile size and never normalised.
 */
export function DownlinkPanel({
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
  const frames = buffer.sent.map((i) => mission.frames[i]).filter(Boolean);
  const bitsPct =
    record && record.bits_budget ? (record.bits_used / record.bits_budget) * 100 : 0;

  return (
    <Card className="flex h-full min-h-0 flex-col gap-0 overflow-hidden py-0">
      <CardHeader className="shrink-0 gap-1 border-b border-border px-4 py-3">
        <CardTitle className="flex items-baseline justify-between text-sm font-medium">
          <span>Downlinked this window</span>
          <span className="metric text-xs font-normal text-muted-foreground">
            {frames.length} frames
          </span>
        </CardTitle>
        <p className="text-xs text-muted-foreground">
          In the order the policy selected them, left to right.
        </p>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 p-0">
        <ScrollArea className="h-full">
          <div className="flex flex-wrap content-start gap-1.5 p-3">
            {frames.length === 0 ? (
              <p className="p-4 text-xs text-lost">
                Nothing was transmitted this window.
              </p>
            ) : (
              frames.map((frame, n) => (
                <div key={frame.i} className="flex flex-col items-center gap-1">
                  <FrameTile
                    frame={frame}
                    mission={mission}
                    size={54}
                    state="sent"
                    title={frameTitle(frame)}
                    onHover={onHover}
                  />
                  <span className="metric text-[9px] text-muted-foreground">
                    {n + 1}
                  </span>
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </CardContent>
      {record ? (
        <div className="flex shrink-0 items-center gap-2 border-t border-border px-4 py-2 text-xs">
          <Badge
            variant={record.bound === "cycles" ? "destructive" : "secondary"}
            className="metric text-[10px]"
          >
            {record.bound}-limited
          </Badge>
          <span className="metric text-muted-foreground">
            {bitsPct.toFixed(0)}% of bit budget
          </span>
          <span className="metric ml-auto text-muted-foreground">
            <span className="text-natural">{record.nat}</span> nat ·{" "}
            <span className="text-rover">{record.rov}</span> rov · {record.typ} typ
          </span>
        </div>
      ) : null}
    </Card>
  );
}
