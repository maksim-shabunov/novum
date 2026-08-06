"""Class taxonomy: which novelty classes are science and which are housekeeping.

The task-1 per-class breakdown showed the aggregate ROC AUC hiding a structural
split. The detector is strong on natural geology (veins 0.94, broken-rock 0.91)
and near or below chance on things the rover did to the terrain itself
(drt 0.42, dump-pile 0.46). For a downlink triage system that is arguably the
*right* behaviour -- a drill hole is not science the geologists have not seen --
but a single aggregate number cannot show it. So the evaluation decomposes:

    natural   novel because Mars made it     -> the frames triage exists to find
    rover     novel because the rover did it -> novelty, but not discovery
    excluded  'other' (n=1) and 'edge_cases' (n=3): too few frames for a rate
              to mean anything; reported separately, never folded into a group

Group membership: a frame belongs to a group if ANY of its labels is in that
group (labels are joined with '|' in the canonical manifest for the five
multi-label frames). In the real archive every multi-label frame combines two
rover classes, so no frame straddles groups -- but the rule is defined anyway,
and a straddling frame would legitimately count in both groups.
"""

from __future__ import annotations

from collections.abc import Iterable

NATURAL_CLASSES: frozenset[str] = frozenset(
    {"veins", "broken-rock", "float", "bedrock", "meteorite"}
)
ROVER_CLASSES: frozenset[str] = frozenset({"drt", "dump-pile", "drill-hole", "scuff"})
EXCLUDED_CLASSES: frozenset[str] = frozenset({"other", "edge_cases"})

KNOWN_CLASSES: frozenset[str] = NATURAL_CLASSES | ROVER_CLASSES | EXCLUDED_CLASSES

GROUP_NATURAL = "natural"
GROUP_ROVER = "rover"
GROUP_EXCLUDED = "excluded"


def split_labels(class_field: str) -> list[str]:
    """'drill-hole|dump-pile' -> ['drill-hole', 'dump-pile']. Never raises."""
    return [part for part in str(class_field or "").split("|") if part]


def groups_for_labels(labels: Iterable[str]) -> set[str]:
    """Every group this frame belongs to. Unknown labels map to excluded."""
    groups: set[str] = set()
    for label in labels:
        if label in NATURAL_CLASSES:
            groups.add(GROUP_NATURAL)
        elif label in ROVER_CLASSES:
            groups.add(GROUP_ROVER)
        else:
            # 'other', 'edge_cases', 'unknown', or anything a future dataset
            # revision introduces: report it, never silently fold it in.
            groups.add(GROUP_EXCLUDED)
    return groups or {GROUP_EXCLUDED}


def group_for_class_field(class_field: str) -> set[str]:
    return groups_for_labels(split_labels(class_field))
