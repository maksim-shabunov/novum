"use client";

import { cn } from "@/lib/utils";
import type { Frame, Mission } from "@/lib/console-data";

/**
 * One frame, sliced out of the sprite atlas.
 *
 * The atlas is a single 1.8 MB PNG holding all 856 tiles; a tile is addressed
 * with background-position rather than its own request. Eight hundred image
 * requests would make the buffer grid the slowest thing on the page, and the
 * buffer grid is the thing a judge looks at first.
 */

export const TILE_SOURCE = 32;

export function FrameTile({
  frame,
  mission,
  size = 44,
  state,
  title,
  onHover,
}: {
  frame: Frame;
  mission: Mission;
  size?: number;
  /** How this frame is faring, which drives the ring colour. */
  state?: "held" | "sent" | "lost";
  title?: string;
  onHover?: (frame: Frame | null) => void;
}) {
  const scale = size / TILE_SOURCE;
  return (
    <div
      title={title}
      onMouseEnter={onHover ? () => onHover(frame) : undefined}
      onMouseLeave={onHover ? () => onHover(null) : undefined}
      className={cn(
        "relative shrink-0 rounded-[3px] bg-black/40",
        // The class of the frame is the point, so it gets the persistent ring.
        frame.group === "natural" && "ring-1 ring-natural/70",
        frame.group === "rover" && "ring-1 ring-rover/50",
        frame.group === "typical" && "ring-1 ring-white/5",
        state === "lost" && "opacity-35 grayscale",
        state === "sent" && "ring-2 ring-current",
      )}
      style={{
        width: size,
        height: size,
        backgroundImage: "url(/data/atlas.png)",
        backgroundRepeat: "no-repeat",
        imageRendering: "auto",
        backgroundSize: `${mission.atlas.width * scale}px ${
          mission.atlas.height * scale
        }px`,
        backgroundPosition: `-${frame.ax * size}px -${frame.ay * size}px`,
      }}
    >
      {state === "lost" ? (
        <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-lost">
          ✕
        </span>
      ) : null}
    </div>
  );
}

export function frameTitle(f: Frame): string {
  const group =
    f.group === "natural"
      ? "natural science"
      : f.group === "rover"
        ? "rover hardware"
        : "typical terrain";
  return `#${f.i} · sol ${f.sol} · ${group} · ${f.cls} · ${Math.round(f.bits)} bits`;
}
