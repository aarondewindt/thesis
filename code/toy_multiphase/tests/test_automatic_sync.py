#!/usr/bin/env python3
"""Test automatic weight synchronization in the improved shared critic model."""

import os
import tensorflow as tf

# Enable TensorFlow memory growth
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

import ray
from ray.rllib.algorithms.ppo import PPOConfig

import sys
sys.path.append('/workspaces/thesis/code/toy_multiphase/src')

from toy_mp.rllib.models.shared_critic_model_tf import SharedCriticRayActor, PhaseActorSharedCriticTFModel
from toy_mp.rllib.trainers import build_multiagent_config_three_actors_shared_critic
from toy_mp.envs.conveyor_portal.env import ConveyorPortalEnv
from toy_mp.rllib.callbacks import ToyMetricsCallback

def test_automatic_sync():
    """Test that weight synchronization happens automatically during training."""
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Clean up any existing actors
    try:
        for actor in ray.util.list_named_actors(all_namespaces=True):
            ray.kill(ray.get_actor(actor['name'], namespace=actor['namespace']))
    except:
        pass
    
    # Create environment for configuration
    env = ConveyorPortalEnv({
        'max_steps': 10,
        'map_spec': {
            'name': 'v1_debug_easy',
            'path': 'toy_mp/experiments/configs/env/v1_debug_easy.yaml'
        }
    })
    
    print("Creating PPO configuration with automatic sync...")
    
    # Build configuration with our custom callback
    config = (
        PPOConfig()
        .environment(env=ConveyorPortalEnv, env_config={
            'max_steps': 10,
            'map_spec': {
                'name': 'v1_debug_easy',
                'path': 'toy_mp/experiments/configs/env/v1_debug_easy.yaml'
            }
        })
        .rollouts(num_rollout_workers=0)  # Local testing
        .training(
            train_batch_size=64,
            sgd_minibatch_size=32,
            num_sgd_iter=1,
        )
        .callbacks(ToyMetricsCallback)  # This enables automatic sync
    )
    
    # Apply multi-agent configuration
    build_multiagent_config_three_actors_shared_critic(config, env)
    
    print("Building algorithm...")
    algo = config.build()
    
    print("Testing automatic weight synchronization...")
    
    # Get the shared critic actor to monitor weights
    shared_critic_actor = ray.get_actor("shared_critic_actor")
    
    # Get initial weights
    initial_weights = ray.get(shared_critic_actor.get_critic_weights.remote())
    initial_layer_weights = initial_weights['test_shared_critic']['dense_4']['kernel:0']
    print(f"Initial first layer weights (first 5): {initial_layer_weights.flat[:5]}")
    
    # Do one training step - this should trigger automatic sync via callback
    print("\nPerforming training step...")
    result = algo.train()
    print(f"Training step completed. Episode reward mean: {result.get('episode_reward_mean', 'N/A')}")
    
    # Check if weights were updated
    updated_weights = ray.get(shared_critic_actor.get_critic_weights.remote())
    updated_layer_weights = updated_weights['test_shared_critic']['dense_4']['kernel:0']
    print(f"Updated first layer weights (first 5): {updated_layer_weights.flat[:5]}")
    
    # Check if weights actually changed
    weights_changed = not tf.reduce_all(tf.equal(initial_layer_weights, updated_layer_weights)).numpy()
    
    if weights_changed:
        print("✓ SUCCESS: Weights were automatically synchronized during training!")
    else:
        print("⚠ WARNING: Weights did not change during training step")
    
    # Do another training step to ensure it keeps working
    print("\nPerforming second training step...")
    result2 = algo.train()
    
    # Check weights again
    final_weights = ray.get(shared_critic_actor.get_critic_weights.remote())
    final_layer_weights = final_weights['test_shared_critic']['dense_4']['kernel:0']
    
    second_change = not tf.reduce_all(tf.equal(updated_layer_weights, final_layer_weights)).numpy()
    print(f"Final first layer weights (first 5): {final_layer_weights.flat[:5]}")
    
    if second_change:
        print("✓ SUCCESS: Weights continue to be automatically synchronized!")
    else:
        print("⚠ NOTE: No additional weight changes in second training step")
    
    # Clean up
    algo.stop()
    
    print(f"\n🎉 Test completed! Automatic synchronization: {'WORKING' if weights_changed else 'NEEDS VERIFICATION'}")
    
    return weights_changed

if __name__ == "__main__":
    test_automatic_sync()