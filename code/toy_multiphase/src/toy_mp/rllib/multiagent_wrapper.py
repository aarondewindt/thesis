from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

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

        # Define all possible agents that can appear in episodes
        self.possible_agents = ["phase1", "phase2", "phase3"]
        
        # Initially only phase1 is active, will be updated during episodes
        self.agents = ["phase1"]
        
        # Each agent has the same observation and action spaces as the underlying env
        self.observation_spaces = {
            agent_id: env.observation_space for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: env.action_space for agent_id in self.possible_agents
        }

    def _active_agent(self) -> str:
        phase = int(self._env._phase)  # noqa: SLF001
        return PHASE_TO_AGENT_ID.get(phase, "phase1")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        obs, info = self._env.reset(seed=seed, options=options)
        agent_id = self._active_agent()
        
        # Update self.agents to reflect the currently active agent
        self.agents = [agent_id]
        
        self._last_info = info
        return {agent_id: obs}, {agent_id: info}

    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        if len(action_dict) != 1:
            raise ValueError(f"Expected exactly one acting agent, got {list(action_dict.keys())}")

        (agent_id, action), *_ = action_dict.items()
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        self._last_info = info

        next_agent = self._active_agent()
        
        # Update self.agents to reflect which agent should act next
        # If episode is done, no agents are active; otherwise the next agent is active
        if terminated or truncated:
            self.agents = []
            obs_dict = {}
            info_dict = {}  # No info when no observations
        else:
            self.agents = [next_agent]
            obs_dict = {next_agent: obs}
            info_dict = {next_agent: info}  # Info only for the active agent

        rew_dict = {agent_id: float(reward)}
        term_dict = {"__all__": bool(terminated)}
        trunc_dict = {"__all__": bool(truncated)}
        return obs_dict, rew_dict, term_dict, trunc_dict, info_dict

    def render(self) -> None:
        self._env.render()
        return None
