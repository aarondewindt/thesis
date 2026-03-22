#!/usr/bin/env python3
"""Test the fixed experiment runner with shared critic configuration."""

import sys
import os
sys.path.append('/workspaces/thesis/code/toy_multiphase/src')

# Set TensorFlow memory growth
import tensorflow as tf
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

import ray
from toy_mp.rllib.trainers import build_multiagent_config_three_actors_shared_critic

def test_shared_critic_training():
    """Test that the shared critic configuration can train without errors."""
    
    print("Testing shared critic training with fixed MultiAgent wrapper...")
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    try:
        print("\n1. Building configuration...")
        
        # Build the configuration (this should use the fixed wrapper)
        cfg = build_multiagent_config_three_actors_shared_critic(
            env_yaml="/workspaces/thesis/code/toy_multiphase/src/toy_mp/experiments/configs/env/v1_debug_easy.yaml", 
            framework="tf", 
            num_workers=0, 
            num_gpus=0
        ).debugging(seed=42)
        
        print("✓ Configuration built successfully")
        
        print("\n2. Creating algorithm...")
        algo = cfg.build()
        print("✓ Algorithm created successfully")
        
        print("\n3. Running training iterations...")
        
        # Try a few training iterations to see if the error is fixed
        for i in range(3):
            print(f"  Running iteration {i+1}...")
            result = algo.train()
            
            # Check if we got valid results
            episode_reward = result.get("episode_reward_mean", "N/A")
            episodes_total = result.get("episodes_total", "N/A")
            timesteps_total = result.get("timesteps_total", "N/A")
            
            print(f"    ✓ Iteration {i+1} completed")
            print(f"      Episode reward mean: {episode_reward}")
            print(f"      Episodes total: {episodes_total}")
            print(f"      Timesteps total: {timesteps_total}")
        
        print("\n✅ SUCCESS: All training iterations completed without ValueError!")
        print("✅ The info dict compliance fix is working correctly")
        
        # Clean up
        algo.stop()
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        return False
    
    finally:
        # Clean up Ray if we initialized it
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    success = test_shared_critic_training()
    if success:
        print("\n🎉 Shared critic training test passed!")
        print("The MultiAgent wrapper fix successfully resolves the info dict compliance issue.")
    else:
        print("\n💥 Test failed - there may be additional issues to resolve.")