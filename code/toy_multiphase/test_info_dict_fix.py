#!/usr/bin/env python3
"""Test the fixed SequentialPhaseMAEnv to ensure info dict compliance."""

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
        self._step_count = 0
        
    def reset(self, seed=None, options=None):
        self._phase = 1
        self._step_count = 0
        return [0.0, 0.0, 0.0, 0.0], {"phase": 1, "step": 0}
        
    def step(self, action):
        self._step_count += 1
        # Simple mock: terminate after 3 steps
        terminated = (self._step_count >= 3)
        if not terminated:
            self._phase = (self._phase % 3) + 1
            
        obs = [0.1, 0.2, 0.3, 0.4]
        reward = 1.0
        truncated = False
        info = {"phase": self._phase, "step": self._step_count}
        return obs, reward, terminated, truncated, info
        
    def render(self):
        return None

def test_info_dict_compliance():
    """Test that info dict keys are always a subset of observation dict keys."""
    
    print("Testing SequentialPhaseMAEnv info dict compliance...")
    
    # Create wrapper around mock environment
    mock_env = MockConveyorPortalEnv({})
    ma_env = SequentialPhaseMAEnv(mock_env)
    
    print("\n1. Testing reset behavior...")
    obs_dict, info_dict = ma_env.reset()
    
    print(f"✓ Reset - obs_dict keys: {list(obs_dict.keys())}")
    print(f"✓ Reset - info_dict keys: {list(info_dict.keys())}")
    
    # Check that info keys are subset of obs keys
    obs_keys = set(obs_dict.keys())
    info_keys = set(info_dict.keys())
    assert info_keys.issubset(obs_keys.union({"__common__"})), f"Info keys {info_keys} must be subset of obs keys {obs_keys}"
    print("✅ Reset: Info keys are valid subset of obs keys")
    
    print("\n2. Testing step behavior during episode...")
    
    step_count = 0
    episode_done = False
    
    while not episode_done and step_count < 10:  # Safety limit
        # Get current agent
        if obs_dict:
            current_agent = list(obs_dict.keys())[0]
            action_dict = {current_agent: 0}
            
            # Take step
            obs_dict, rew_dict, term_dict, trunc_dict, info_dict = ma_env.step(action_dict)
            
            step_count += 1
            episode_done = term_dict.get("__all__", False) or trunc_dict.get("__all__", False)
            
            print(f"✓ Step {step_count} - obs_dict keys: {list(obs_dict.keys())}")
            print(f"✓ Step {step_count} - info_dict keys: {list(info_dict.keys())}")
            print(f"✓ Step {step_count} - episode_done: {episode_done}")
            
            # Check that info keys are subset of obs keys (or empty when episode done)
            obs_keys = set(obs_dict.keys())
            info_keys = set(info_dict.keys())
            assert info_keys.issubset(obs_keys.union({"__common__"})), \
                f"Step {step_count}: Info keys {info_keys} must be subset of obs keys {obs_keys} (plus __common__)"
            
            if episode_done:
                print("✅ Episode termination: Info dict properly empty when obs dict is empty")
            else:
                print("✅ Mid-episode: Info keys are valid subset of obs keys")
        else:
            print("No observations available - episode likely ended")
            break
    
    print(f"\n3. Final validation...")
    print(f"✓ Total steps taken: {step_count}")
    print(f"✓ Episode completed: {episode_done}")
    
    # Final check - when episode is done, both obs and info should be empty
    if episode_done:
        assert len(obs_dict) == 0, "Obs dict should be empty when episode done"
        assert len(info_dict) == 0, "Info dict should be empty when episode done"
        print("✅ Final state: Both obs_dict and info_dict are empty as expected")
    
    print("\n🎉 All tests passed! Info dict keys are always a subset of obs dict keys.")
    
    # Test the specific error condition that was occurring
    print("\n4. Testing specific error scenario...")
    
    # Reset and then take enough steps to end episode
    obs_dict, info_dict = ma_env.reset()
    
    for i in range(5):  # Take more steps than needed to ensure termination
        if not obs_dict:  # No more observations
            break
            
        current_agent = list(obs_dict.keys())[0]
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = ma_env.step({current_agent: 0})
        
        if term_dict.get("__all__") or trunc_dict.get("__all__"):
            print(f"✓ Episode ended at step {i+1}")
            print(f"✓ Final obs_dict: {obs_dict}")
            print(f"✓ Final info_dict: {info_dict}")
            
            # This should NOT raise the error anymore
            assert len(obs_dict) == len(info_dict) == 0, "Both dicts should be empty at episode end"
            print("✅ No ValueError raised - fix successful!")
            break
    
    return True

if __name__ == "__main__":
    test_info_dict_compliance()