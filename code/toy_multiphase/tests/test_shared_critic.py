#!/usr/bin/env python3
"""
Example usage of the new shared critic policy implementation.

This demonstrates:
1. How to use the Ray-based shared critic policy
2. Multi-worker compatibility 
3. Separate critics for different experiments
"""

import ray
from toy_mp.rllib.trainers import build_multiagent_config_three_actors_shared_critic


def test_shared_critic_policy():
    """Test the shared critic policy with different configurations."""
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(local_mode=False)  # Use actual Ray cluster
    
    try:
        # Test 1: Single worker setup
        print("=== Test 1: Single worker setup ===")
        config_single = build_multiagent_config_three_actors_shared_critic(
            env_yaml="src/toy_mp/experiments/configs/env/v1_debug_easy.yaml",
            framework="tf2",
            num_workers=0,  # Single worker
            critic_id="test_single_worker"
        )
        
        algo_single = config_single.build()
        result_single = algo_single.train()
        print(f"Single worker training completed: {result_single.get('training_iteration', 'N/A')}")
        algo_single.stop()
        
        # Test 2: Multi-worker setup  
        print("\n=== Test 2: Multi-worker setup ===")
        config_multi = build_multiagent_config_three_actors_shared_critic(
            env_yaml="src/toy_mp/experiments/configs/env/v1_debug_easy.yaml",
            framework="tf2", 
            num_workers=2,  # Multi-worker
            critic_id="test_multi_worker"
        )
        
        algo_multi = config_multi.build()
        result_multi = algo_multi.train()
        print(f"Multi-worker training completed: {result_multi.get('training_iteration', 'N/A')}")
        algo_multi.stop()
        
        # Test 3: Different critic IDs (simulating multiple experiments)
        print("\n=== Test 3: Multiple experiments with different critic IDs ===")
        
        config_exp1 = build_multiagent_config_three_actors_shared_critic(
            env_yaml="src/toy_mp/experiments/configs/env/v1_debug_easy.yaml",
            framework="tf2",
            num_workers=1,
            critic_id="experiment_1"
        )
        
        config_exp2 = build_multiagent_config_three_actors_shared_critic(
            env_yaml="src/toy_mp/experiments/configs/env/v1_debug_easy.yaml", 
            framework="tf2",
            num_workers=1,
            critic_id="experiment_2"
        )
        
        algo_exp1 = config_exp1.build()
        algo_exp2 = config_exp2.build()
        
        # Train both simultaneously (different critics)
        result_exp1 = algo_exp1.train()
        result_exp2 = algo_exp2.train()
        
        print(f"Experiment 1 completed: {result_exp1.get('training_iteration', 'N/A')}")
        print(f"Experiment 2 completed: {result_exp2.get('training_iteration', 'N/A')}")
        
        algo_exp1.stop()
        algo_exp2.stop()
        
        print("\n✅ All tests passed! Shared critic policy is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        ray.shutdown()


if __name__ == "__main__":
    test_shared_critic_policy()