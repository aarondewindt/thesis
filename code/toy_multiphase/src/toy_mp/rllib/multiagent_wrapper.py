from __future__ import annotations

from typing import Any, Dict, Optional

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from toy_mp.envs.conveyor_portal.env import ConveyorPortalEnv


PHASE_TO_AGENT_ID = {1: "phase1", 2: "phase2", 3: "phase3"}


class SequentialPhaseMAEnv(MultiAgentEnv):
    """Turn-based MultiAgentEnv wrapper around ConveyorPortalEnv.

    RLlib supports multi-agent environments that are simultaneous or turn-based.
    """

    def __init__(self, env: ConveyorPortalEnv):
        super().__init__()
        self._env = env
        self._last_info: Dict[str, Any] = {}
        self.spec = None

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def _active_agent(self) -> str:
        phase = int(self._env._phase)  # noqa: SLF001
        return PHASE_TO_AGENT_ID.get(phase, "phase1")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self._env.reset(seed=seed, options=options)
        agent_id = self._active_agent()
        self._last_info = info
        return {agent_id: obs}, {agent_id: info}

    def step(self, action_dict: Dict[str, int]):
        if len(action_dict) != 1:
            raise ValueError(f"Expected exactly one acting agent, got {list(action_dict.keys())}")

        (agent_id, action), *_ = action_dict.items()
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        self._last_info = info

        next_agent = self._active_agent()
        obs_dict = {next_agent: obs} if not (terminated or truncated) else {}

        rew_dict = {agent_id: float(reward)}
        term_dict = {"__all__": bool(terminated)}
        trunc_dict = {"__all__": bool(truncated)}
        info_dict = {agent_id: info}
        return obs_dict, rew_dict, term_dict, trunc_dict, info_dict

    def render(self):
        return self._env.render()
