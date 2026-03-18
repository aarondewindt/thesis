from __future__ import annotations

from typing import Dict, Tuple

from .map_spec import MapSpec, Coord


def build_portal_map(spec: MapSpec) -> Dict[Coord, Coord]:
    return {p.src: p.dst for p in spec.portals}


def build_belt_maps(spec: MapSpec) -> Tuple[Dict[Coord, str], Dict[Tuple[str, Coord], Coord]]:
    belt_at: Dict[Coord, str] = {}
    belt_next: Dict[Tuple[str, Coord], Coord] = {}

    for b in spec.belts:
        for c in b.path:
            if c in belt_at:
                raise ValueError(f"Overlapping belt tiles at {c} (belts {belt_at[c]} and {b.id})")
            belt_at[c] = b.id
        for a, nxt in zip(b.path[:-1], b.path[1:]):
            belt_next[(b.id, a)] = nxt
    return belt_at, belt_next
