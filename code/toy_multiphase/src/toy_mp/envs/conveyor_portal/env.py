from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from .map_spec import MapSpec, Coord
from .tiles import Tile, DIRS_8, add_xy, right_of
from .validators import build_belt_maps, build_portal_map


@dataclass
class StepResult:
    obs: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class ConveyorPortalEnv(gym.Env):
    """Conveyor–Portal toy environment (Milestone B: correctness + tests)."""

    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    def __init__(self, map_spec: MapSpec):
        super().__init__()
        self.map_spec = map_spec

        self.portal_map = build_portal_map(map_spec)
        self.belt_at, self.belt_next = build_belt_maps(map_spec)

        self.max_steps = map_spec.max_steps
        self.wait_n_max = map_spec.wait_n_max

        self.observation_space = spaces.Dict(
            {
                "tiles": spaces.Box(
                    low=0, high=int(Tile.GOAL), shape=(map_spec.height, map_spec.width), dtype=np.int8
                ),
                "agent_xy": spaces.Box(
                    low=0, high=max(map_spec.width, map_spec.height), shape=(2,), dtype=np.int16
                ),
                "phase": spaces.Box(low=1, high=3, shape=(1,), dtype=np.int32),  # 1..3 used
                "step": spaces.Box(low=0, high=map_spec.max_steps, shape=(), dtype=np.int32),
            }
        )

        # Single discrete action for all phases.
        # 0..7 moves (8-directional), 8 noop, 9..(9+wait_n_max) wait_n
        self.action_space = spaces.Discrete(9 + (self.wait_n_max + 1))

        self._tiles = self._build_tilemap(map_spec)
        self._agent: Coord = map_spec.start
        self._phase: int = 1
        self._steps: int = 0
        self._terminated: bool = False
        self._truncated: bool = False
        self._active_belt_id: Optional[str] = None

    @staticmethod
    def from_yaml(path: str) -> "ConveyorPortalEnv":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        spec = MapSpec.model_validate(data)
        return ConveyorPortalEnv(spec)

    def _build_tilemap(self, spec: MapSpec) -> np.ndarray:
        tiles = np.full((spec.height, spec.width), int(Tile.EMPTY), dtype=np.int8)
        for x, y in spec.walls:
            tiles[y, x] = int(Tile.WALL)
        for x, y in spec.deaths:
            tiles[y, x] = int(Tile.DEATH)
        for b in spec.belts:
            for x, y in b.path:
                tiles[y, x] = int(Tile.BELT)
        for p in spec.portals:
            x, y = p.src
            tiles[y, x] = int(Tile.PORTAL)
        gx, gy = spec.goal
        tiles[gy, gx] = int(Tile.GOAL)
        return tiles

    def _obs(self) -> Dict[str, Any]:
        return {
            "tiles": self._tiles.copy(),
            "agent_xy": np.array(self._agent, dtype=np.int16),
            "phase": np.array([self._phase], dtype=np.int32),
            "step": np.int32(self._steps),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._agent = self.map_spec.start
        self._phase = 1
        self._steps = 0
        self._terminated = False
        self._truncated = False
        self._active_belt_id = None
        obs = self._obs()
        info: Dict[str, Any] = {"phase": self._phase}
        return obs, info

    def step(self, action: int):
        if self._terminated or self._truncated:
            raise RuntimeError("step() called after episode done; call reset().")

        self._steps += 1
        info: Dict[str, Any] = {"phase": self._phase}

        if self._phase in (1, 3):
            self._step_nav(action, info)
        elif self._phase == 2:
            self._step_belt(action, info)
        else:
            raise RuntimeError(f"Invalid phase {self._phase}")

        if self._steps >= self.max_steps and not self._terminated:
            self._truncated = True

        reward = 1.0 if self._terminated and self._agent == self.map_spec.goal else 0.0
        return self._obs(), reward, self._terminated, self._truncated, info

    def _tile_at(self, c: Coord) -> Tile:
        x, y = c
        return Tile(int(self._tiles[y, x]))

    def _in_bounds(self, c: Coord) -> bool:
        x, y = c
        return 0 <= x < self.map_spec.width and 0 <= y < self.map_spec.height

    def _try_move(self, dir_idx: int) -> None:
        nxt = add_xy(self._agent, DIRS_8[dir_idx])
        if not self._in_bounds(nxt):
            return
        if self._tile_at(nxt) == Tile.WALL:
            return
        self._agent = nxt

    def _apply_entry_effects(self, info: Dict[str, Any]) -> None:
        t = self._tile_at(self._agent)

        if t == Tile.DEATH:
            self._terminated = True
            info["event"] = "death"
            return

        if self._agent == self.map_spec.goal:
            self._terminated = True
            info["event"] = "goal"
            return

        if t == Tile.PORTAL:
            dst = self.portal_map.get(self._agent)
            if dst is None:
                self._terminated = True
                info["event"] = "invalid_portal"
                return
            self._agent = dst
            info["event"] = "portal"
            info["portal_dst"] = dst

            # After teleport, re-check death/goal but do NOT chain portals
            t2 = self._tile_at(self._agent)
            if t2 == Tile.DEATH:
                self._terminated = True
                info["event_after_portal"] = "death"
            elif self._agent == self.map_spec.goal:
                self._terminated = True
                info["event_after_portal"] = "goal"
            return

        if t == Tile.BELT:
            belt_id = self.belt_at.get(self._agent)
            if belt_id is None:
                self._terminated = True
                info["event"] = "invalid_belt"
                return
            self._phase = 2
            self._active_belt_id = belt_id
            info["event"] = "enter_belt"
            info["belt_id"] = belt_id

    def _step_nav(self, action: int, info: Dict[str, Any]) -> None:
        if 0 <= action <= 7:
            self._try_move(action)
        # 8 noop; >=9 treated as noop in nav phases
        self._apply_entry_effects(info)

    def _step_belt(self, action: int, info: Dict[str, Any]) -> None:
        wait_n = 0 if action < 9 else min(action - 9, self.wait_n_max)
        info["wait_n"] = int(wait_n)

        if self._active_belt_id is None:
            self._terminated = True
            info["event"] = "belt_without_id"
            return

        for _ in range(wait_n):
            key = (self._active_belt_id, self._agent)
            if key not in self.belt_next:
                self._terminated = True
                info["event"] = "belt_falloff_death" if self.map_spec.belt_falloff_is_death else "belt_falloff_terminal"
                return
            self._agent = self.belt_next[key]
            if self._tile_at(self._agent) == Tile.DEATH:
                self._terminated = True
                info["event"] = "death_on_belt"
                return
            if self._agent == self.map_spec.goal:
                self._terminated = True
                info["event"] = "goal_on_belt"
                return

        key = (self._active_belt_id, self._agent)
        if key not in self.belt_next:
            self._terminated = True
            info["event"] = "belt_end_no_direction"
            return

        nxt = self.belt_next[key]
        dir_xy = (nxt[0] - self._agent[0], nxt[1] - self._agent[1])
        side = right_of(dir_xy)
        step_off = (self._agent[0] + side[0], self._agent[1] + side[1])
        info["step_off_xy"] = step_off

        if not self._in_bounds(step_off):
            self._terminated = True
            info["event"] = "step_off_oob_death"
            return
        if self._tile_at(step_off) != Tile.PORTAL:
            self._terminated = True
            info["event"] = "step_off_non_portal_death"
            return

        # step onto portal
        self._agent = step_off
        self._apply_entry_effects(info)
        if not self._terminated:
            self._phase = 3
            self._active_belt_id = None
            info["event_phase3"] = "begin_phase3"

    def render(self):
        glyph = {
            Tile.EMPTY: ".",
            Tile.WALL: "#",
            Tile.DEATH: "X",
            Tile.BELT: "=",
            Tile.PORTAL: "O",
            Tile.GOAL: "G",
        }
        lines = []
        for y in range(self.map_spec.height):
            row = []
            for x in range(self.map_spec.width):
                if (x, y) == self._agent:
                    row.append("A")
                else:
                    row.append(glyph[Tile(int(self._tiles[y, x]))])
            lines.append("".join(row))
        return "\n".join(lines) + f"\nphase={self._phase} step={self._steps}"
