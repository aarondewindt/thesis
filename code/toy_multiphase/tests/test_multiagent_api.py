#!/usr/bin/env python3
"""Test the updated SequentialPhaseMAEnv to ensure it follows RLlib MultiAgentEnv API."""

import sys
sys.path.append('/workspaces/thesis/code/toy_multiphase/src')

from toy_mp.envs.conveyor_portal.env import ConveyorPortalEnv
from toy_mp.rllib.multiagent_wrapper import SequentialPhaseMAEnv

def test_multiagent_api_compliance():
    """Test that the wrapper properly implements the MultiAgentEnv API."""
    
    print("Testing SequentialPhaseMAEnv API compliance...")
    
    # Create underlying environment
    env_config = {
        'max_steps': 10,
        'map_spec': {
            'name': 'v1_debug_easy',
            'path': 'toy_mp/experiments/configs/env/v1_debug_easy.yaml'
        }
    }
    
    base_env = ConveyorPortalEnv(env_config)
    ma_env = SequentialPhaseMAEnv(base_env)
    
    # Test 1: Check required attributes exist
    print("\n1. Checking required attributes...")
    
    # Check possible_agents
    assert hasattr(ma_env, 'possible_agents'), "Missing possible_agents attribute"
    assert ma_env.possible_agents == ["phase1", "phase2", "phase3"], f"Wrong possible_agents: {ma_env.possible_agents}"
    print(f"✓ possible_agents: {ma_env.possible_agents}")
    
    # Check agents (initially should be set)
    assert hasattr(ma_env, 'agents'), "Missing agents attribute" 
    print(f"✓ agents: {ma_env.agents}")
    
    # Check observation_spaces is a dict
    assert hasattr(ma_env, 'observation_spaces'), "Missing observation_spaces attribute"
    assert isinstance(ma_env.observation_spaces, dict), "observation_spaces should be a dict"
    assert len(ma_env.observation_spaces) == 3, f"Should have 3 observation spaces, got {len(ma_env.observation_spaces)}"
    print(f"✓ observation_spaces: {list(ma_env.observation_spaces.keys())}")
    
    # Check action_spaces is a dict  
    assert hasattr(ma_env, 'action_spaces'), "Missing action_spaces attribute"
    assert isinstance(ma_env.action_spaces, dict), "action_spaces should be a dict"
    assert len(ma_env.action_spaces) == 3, f"Should have 3 action spaces, got {len(ma_env.action_spaces)}"
    print(f"✓ action_spaces: {list(ma_env.action_spaces.keys())}")
    
    # Test 2: Test reset behavior
    print("\n2. Testing reset behavior...")
    
    obs_dict, info_dict = ma_env.reset()
    
    # Should return dicts
    assert isinstance(obs_dict, dict), "reset() should return observation dict"
    assert isinstance(info_dict, dict), "reset() should return info dict"
    
    # Should have exactly one agent (the starting agent)
    assert len(obs_dict) == 1, f"reset() should return exactly 1 observation, got {len(obs_dict)}"
    assert len(info_dict) == 1, f"reset() should return exactly 1 info, got {len(info_dict)}"
    
    starting_agent = list(obs_dict.keys())[0]
    print(f"✓ Starting agent: {starting_agent}")
    
    # agents should be updated to reflect starting agent
    assert ma_env.agents == [starting_agent], f"agents should be [{starting_agent}], got {ma_env.agents}"
    print(f"✓ Active agents after reset: {ma_env.agents}")
    
    # Test 3: Test step behavior
    print("\n3. Testing step behavior...")
    
    # Take an action with the active agent
    action_dict = {starting_agent: 0}  # Some valid action
    
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
    
    # If episode not done, obs_dict should have the next agent
    if not term_dict.get("__all__", False) and not trunc_dict.get("__all__", False):
        assert len(obs_dict) == 1, "Should have exactly one agent to act next"
        next_agent = list(obs_dict.keys())[0]
        print(f"✓ Next agent to act: {next_agent}")
        
        # agents should be updated to next agent
        assert ma_env.agents == [next_agent], f"agents should be [{next_agent}], got {ma_env.agents}"
        print(f"✓ Active agents after step: {ma_env.agents}")
    else:
        # Episode is done, no agents should be active
        assert len(obs_dict) == 0, "No observations when episode is done"
        assert ma_env.agents == [], f"No agents should be active when done, got {ma_env.agents}"
        print("✓ Episode completed, no active agents")
    
    print(f"✓ Termination: {term_dict}")
    print(f"✓ Truncation: {trunc_dict}")
    
    print("\n🎉 All tests passed! SequentialPhaseMAEnv follows RLlib MultiAgentEnv API correctly.")
    return True

if __name__ == "__main__":
    test_multiagent_api_compliance()