from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Coord = Tuple[int, int]


class BeltSpec(BaseModel):
    id: str = Field(..., min_length=1)
    path: List[Coord] = Field(..., min_length=2)


class PortalSpec(BaseModel):
    src: Coord
    dst: Coord


class MapSpec(BaseModel):
    """Full environment configuration."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(..., ge=3)
    height: int = Field(..., ge=3)

    start: Coord
    goal: Coord

    walls: List[Coord] = Field(default_factory=list)
    deaths: List[Coord] = Field(default_factory=list)

    belts: List[BeltSpec] = Field(default_factory=list)
    portals: List[PortalSpec] = Field(default_factory=list)

    max_steps: int = Field(default=120, ge=1)
    wait_n_max: int = Field(default=12, ge=0)
    belt_falloff_is_death: bool = True

    @field_validator("start", "goal")
    @classmethod
    def _coord_pair(cls, v: Coord) -> Coord:
        if len(v) != 2:
            raise ValueError("Coordinate must be a pair (x,y).")
        return v

    @model_validator(mode="after")
    def _validate_all(self) -> "MapSpec":
        def in_bounds(c: Coord) -> bool:
            x, y = c
            return 0 <= x < self.width and 0 <= y < self.height

        all_coords: List[Coord] = [self.start, self.goal]
        all_coords += self.walls + self.deaths
        for b in self.belts:
            all_coords += b.path
        for p in self.portals:
            all_coords += [p.src, p.dst]

        for c in all_coords:
            if not in_bounds(c):
                raise ValueError(f"Out-of-bounds coordinate: {c}")

        wall_set = set(self.walls)
        if self.start in wall_set or self.goal in wall_set:
            raise ValueError("start/goal cannot be a WALL.")

        ids = [b.id for b in self.belts]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate belt id found.")

        for b in self.belts:
            if len(set(b.path)) != len(b.path):
                raise ValueError(f"Belt '{b.id}' path contains duplicate coordinates.")
            for a, nxt in zip(b.path[:-1], b.path[1:]):
                ax, ay = a
                bx, by = nxt
                if abs(ax - bx) + abs(ay - by) != 1:
                    raise ValueError(f"Belt '{b.id}' has non-4-connected segment: {a}->{nxt}")

        srcs = [p.src for p in self.portals]
        if len(srcs) != len(set(srcs)):
            raise ValueError("Duplicate portal source coordinate found.")
        src_set = set(srcs)
        for p in self.portals:
            if p.dst in src_set:
                raise ValueError(f"Portal chaining not supported: dst {p.dst} is a portal src.")

        return self
