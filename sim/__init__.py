"""Downlink window simulator: replay the mission under both budgets.

See sim/window.py for the design decisions (adaptation mode, frame retention,
ground feedback) and for why the compute budget genuinely binds.
"""

from .mission import FrameBuffer, MissionFrame, MissionStream, build_mission
from .policy import METHODS, select
from .window import (
    DownlinkWindow,
    SimConfig,
    SimResult,
    WindowRecord,
    chronological_order,
    plan_windows,
    replay,
)

__all__ = [
    "METHODS",
    "DownlinkWindow",
    "FrameBuffer",
    "MissionFrame",
    "MissionStream",
    "SimConfig",
    "SimResult",
    "WindowRecord",
    "build_mission",
    "chronological_order",
    "plan_windows",
    "replay",
    "select",
]
