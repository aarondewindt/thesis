#!/usr/bin/env python3
"""Test automatic weight synchronization in the improved shared critic model."""

import os
import numpy as np
import tensorflow as tf

# Enable TensorFlow memory growth
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

import sys
sys.path.append('/workspaces/thesis/code/toy_multiphase/src')

import ray
from ray.rllib.algorithms.ppo import PPOConfig

from toy_mp.rllib.models.shared_critic_model_tf import SharedCriticRayActor, PhaseActorSharedCriticTFModel
from toy_mp.rllib.callbacks import ToyMetricsCallback

def test_callback_sync():
    """Test that the callback properly calls weight synchronization."""
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    print("Testing ToyMetricsCallback weight synchronization...")
    
    # Create a mock shared critic actor
    actor_ref = SharedCriticRayActor.remote()
    
    # Initialize with some weights using the correct API
    dummy_weights = [
        np.array([[1.0, 2.0], [3.0, 4.0]]),  # layer 1 weights
        np.array([0.1, 0.2])                 # layer 1 biases
    ]
    ray.get(actor_ref.update_critic_weights.remote("test_critic", dummy_weights))
    
    print("✓ Created test shared critic actor")
    
    # Create a mock algorithm with PhaseActorSharedCriticTFModel policies
    class MockModel(PhaseActorSharedCriticTFModel):
        def __init__(self):
            # Initialize without calling super() to avoid complex setup
            self.push_called = False
            
        def push_critic_weights(self):
            """Mock push method that tracks calls."""
            self.push_called = True
            print("  → Mock model.push_critic_weights() called")
    
    class MockPolicy:
        def __init__(self, has_shared_critic=True):
            if has_shared_critic:
                self.model = MockModel()
            else:
                self.model = object()  # Different model type
    
    class MockAlgorithm:
        def __init__(self):
            self.workers = MockWorkers()
            
    class MockWorkers:
        def __init__(self):
            self.local_worker = MockLocalWorker()
            
    class MockLocalWorker:
        def __init__(self):
            # Create policies with our shared critic models
            self.policy_map = {
                'policy_1': MockPolicy(has_shared_critic=True),
                'policy_2': MockPolicy(has_shared_critic=True),
                'normal_policy': MockPolicy(has_shared_critic=False),  # This one shouldn't be called
            }
    
    # Test the callback
    callback = ToyMetricsCallback()
    mock_algorithm = MockAlgorithm()
    
    # Reset push tracking for shared critic models
    shared_critic_policies = [p for p in mock_algorithm.workers.local_worker.policy_map.values() 
                             if hasattr(p, 'model') and isinstance(p.model, PhaseActorSharedCriticTFModel)]
    
    for policy in shared_critic_policies:
        policy.model.push_called = False
    
    print("\nTesting callback.on_train_result()...")
    
    # Simulate a training result
    mock_result = {
        'training_iteration': 1,
        'episode_reward_mean': 0.5
    }
    
    # Call the callback
    callback.on_train_result(algorithm=mock_algorithm, result=mock_result)
    
    # Check if push was called on all shared critic policies
    push_calls = [policy.model.push_called for policy in shared_critic_policies]
    successful_pushes = sum(push_calls)
    total_shared_critic_policies = len(shared_critic_policies)
    
    print(f"✓ Push called on {successful_pushes}/{total_shared_critic_policies} shared critic policies")
    
    if successful_pushes == total_shared_critic_policies:
        print("🎉 SUCCESS: Callback properly triggers weight synchronization!")
    else:
        print("⚠ WARNING: Not all shared critic policies received push calls")
    
    # Clean up
    ray.kill(actor_ref)
    
    return successful_pushes == total_shared_critic_policies

if __name__ == "__main__":
    success = test_callback_sync()
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: Automatic synchronization callback test")