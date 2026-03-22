#!/usr/bin/env python3
"""Test script to verify the updated build_multiagent_config_three_actors_shared_critic works."""

import os
import sys

# Configure TensorFlow to avoid GPU memory crashes
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import ray
from toy_mp.rllib.trainers import build_multiagent_config_three_actors_shared_critic


def test_config_creation():
    """Test that the configuration can be created successfully."""
    print("Testing build_multiagent_config_three_actors_shared_critic...")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    try:
        # Test configuration creation
        config = build_multiagent_config_three_actors_shared_critic(
            env_yaml="src/toy_mp/experiments/configs/env/v1_debug_easy.yaml",
            framework="tf",
            num_gpus=0,
            num_workers=0,
            critic_id="test_shared_critic"
        )
        
        print("✓ Configuration created successfully")
        
        # Check that policies are configured correctly
        # Access the multi-agent configuration correctly
        multi_agent_dict = config.to_dict().get("multi_agent", {})
        policies = multi_agent_dict.get("policies", {})
        print(f"✓ Found {len(policies)} policies: {list(policies.keys())}")
        
        # Check that all policies use the shared critic model
        for policy_name, policy_spec in policies.items():
            if isinstance(policy_spec, (tuple, list)) and len(policy_spec) >= 4:
                policy_class, obs_space, action_space, policy_config = policy_spec[:4]
                model_config = policy_config.get("model", {})
                custom_model = model_config.get("custom_model")
                custom_model_config = model_config.get("custom_model_config", {})
                critic_id = custom_model_config.get("critic_id")
                
                print(f"  Policy {policy_name}:")
                print(f"    - Custom model: {custom_model.__name__ if custom_model else 'None'}")
                print(f"    - Critic ID: {critic_id}")
            else:
                print(f"  Policy {policy_name}: {policy_spec}")
        
        # Try to build the algorithm (this tests that everything is compatible)
        print("✓ Testing algorithm creation...")
        algo = config.build()
        print("✓ Algorithm created successfully!")
        
        # Clean up
        algo.stop()
        print("✓ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        ray.shutdown()
    
    return True


if __name__ == "__main__":
    success = test_config_creation()
    if success:
        print("\n🎉 SUCCESS: The updated trainer configuration works correctly!")
    else:
        print("\n❌ FAILURE: There are issues with the trainer configuration.")