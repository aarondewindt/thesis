#!/usr/bin/env python3
"""Simple test to verify the SequentialPhaseMAEnv API compliance without environment dependency."""

import sys
sys.path.append('/workspaces/thesis/code/toy_multiphase/src')

import gymnasium as gym
from toy_mp.rllib.multiagent_wrapper import SequentialPhaseMAEnv

# Create a simple mock environment for testing
class MockConveyorPortalEnv:
    def __init__(self, config):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        self.action_space = gym.spaces.Discrete(3)
        self._phase = 1
        
    def reset(self, seed=None, options=None):
        self._phase = 1
        return [0.0, 0.0, 0.0, 0.0], {}
        
    def step(self, action):
        # Simple mock: alternate phases and terminate after 3 steps
        self._phase = (self._phase % 3) + 1
        obs = [0.1, 0.2, 0.3, 0.4]
        reward = 1.0
        terminated = (self._phase == 1)  # Terminate after cycling through phases
        truncated = False
        info = {"phase": self._phase}
        return obs, reward, terminated, truncated, info
        
    def render(self):
        return f"Mock environment, phase: {self._phase}"

def test_api_compliance():
    """Test that the wrapper properly implements the MultiAgentEnv API."""
    
    print("Testing SequentialPhaseMAEnv API compliance...")
    
    # Create wrapper around mock environment
    mock_env = MockConveyorPortalEnv({})
    ma_env = SequentialPhaseMAEnv(mock_env)
    
    print("\n1. Checking required attributes...")
    
    # Check possible_agents
    assert hasattr(ma_env, 'possible_agents'), "Missing possible_agents"
    assert ma_env.possible_agents == ["phase1", "phase2", "phase3"], f"Wrong possible_agents: {ma_env.possible_agents}"
    print(f"✓ possible_agents: {ma_env.possible_agents}")
    
    # Check agents 
    assert hasattr(ma_env, 'agents'), "Missing agents"
    print(f"✓ agents: {ma_env.agents}")
    
    # Check observation_spaces is a dict
    assert hasattr(ma_env, 'observation_spaces'), "Missing observation_spaces"
    assert isinstance(ma_env.observation_spaces, dict), "observation_spaces should be a dict"
    assert len(ma_env.observation_spaces) == 3, f"Should have 3 observation spaces"
    print(f"✓ observation_spaces: {list(ma_env.observation_spaces.keys())}")
    
    # Check action_spaces is a dict  
    assert hasattr(ma_env, 'action_spaces'), "Missing action_spaces"
    assert isinstance(ma_env.action_spaces, dict), "action_spaces should be a dict"
    assert len(ma_env.action_spaces) == 3, f"Should have 3 action spaces"
    print(f"✓ action_spaces: {list(ma_env.action_spaces.keys())}")
    
    print("\n2. Testing reset behavior...")
    
    obs_dict, info_dict = ma_env.reset()
    
    assert isinstance(obs_dict, dict), "reset() should return observation dict"
    assert isinstance(info_dict, dict), "reset() should return info dict"
    assert len(obs_dict) == 1, f"reset() should return exactly 1 observation"
    assert len(info_dict) == 1, f"reset() should return exactly 1 info"
    
    starting_agent = list(obs_dict.keys())[0]
    print(f"✓ Starting agent: {starting_agent}")
    assert ma_env.agents == [starting_agent], f"agents should be [{starting_agent}]"
    print(f"✓ Active agents after reset: {ma_env.agents}")
    
    print("\n3. Testing step behavior...")
    
    action_dict = {starting_agent: 0}
    obs_dict, rew_dict, term_dict, trunc_dict, info_dict = ma_env.step(action_dict)
    
    # All should be dicts
    assert isinstance(obs_dict, dict), "step() should return observation dict"
    assert isinstance(rew_dict, dict), "step() should return reward dict"
    assert isinstance(term_dict, dict), "step() should return termination dict"
    assert isinstance(trunc_dict, dict), "step() should return truncation dict"
    assert isinstance(info_dict, dict), "step() should return info dict"
    
    # Reward should be for the agent that acted
    assert starting_agent in rew_dict, f"Reward should be for {starting_agent}"
    print(f"✓ Reward for {starting_agent}: {rew_dict[starting_agent]}")
    
    # Check episode state
    episode_done = term_dict.get("__all__", False) or trunc_dict.get("__all__", False)
    if not episode_done:
        assert len(obs_dict) == 1, "Should have exactly one agent to act next"
        next_agent = list(obs_dict.keys())[0]
        print(f"✓ Next agent to act: {next_agent}")
        assert ma_env.agents == [next_agent], f"agents should be [{next_agent}]"
        print(f"✓ Active agents after step: {ma_env.agents}")
    else:
        assert len(obs_dict) == 0, "No observations when episode is done"
        assert ma_env.agents == [], f"No agents should be active when done"
        print("✓ Episode completed, no active agents")
    
    print(f"✓ Termination: {term_dict}")
    print(f"✓ Truncation: {trunc_dict}")
    
    print("\n4. Testing turn transitions...")
    
    # If the episode isn't done, test a few more steps
    step_count = 0
    while not episode_done and step_count < 5:
        current_agent = list(obs_dict.keys())[0] if obs_dict else None
        if current_agent is None:
            break
            
        action_dict = {current_agent: 1}
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = ma_env.step(action_dict)
        episode_done = term_dict.get("__all__", False) or trunc_dict.get("__all__", False)
        
        step_count += 1
        if not episode_done:
            next_agent = list(obs_dict.keys())[0]
            print(f"✓ Step {step_count}: {current_agent} → {next_agent}")
        else:
            print(f"✓ Step {step_count}: {current_agent} → Episode done")
    
    print("\n5. Testing render...")
    result = ma_env.render()
    assert result is None, "render() should return None"
    print("✓ render() returns None as expected")
    
    print("\n🎉 All tests passed! SequentialPhaseMAEnv follows RLlib MultiAgentEnv API correctly.")
    
    # Print summary of compliance
    print("\n📋 API Compliance Summary:")
    print("✅ possible_agents: List of all possible agent IDs")
    print("✅ agents: List of currently active agent IDs (updated throughout episode)")
    print("✅ observation_spaces: Dict mapping agent IDs to observation spaces")
    print("✅ action_spaces: Dict mapping agent IDs to action spaces")
    print("✅ reset(): Returns (obs_dict, info_dict) with only starting agent")
    print("✅ step(): Returns 5-tuple with proper MultiAgentDict types")
    print("✅ obs_dict controls turn order: Only active agent gets observation")
    print("✅ agents attribute updated: Reflects which agent should act next")
    print("✅ Episode termination: No active agents when episode ends")
    
    return True

if __name__ == "__main__":
    test_api_compliance()