#!/usr/bin/env python3
"""Test the fixed shared critic implementation with multiple Ray workers."""

import ray
import os
import sys

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from toy_mp.rllib.shared_critic_policy import SharedCriticStore, SharedCriticActor


@ray.remote
def test_worker(worker_id: str):
    """Simulate a worker accessing the shared critic."""
    print(f"Worker {worker_id}: Starting test")
    
    # Each worker tries to get the same shared critic
    critic = SharedCriticStore.get_shared_critic("test_critic", obs_dim=10)
    
    print(f"Worker {worker_id}: Got critic with name: {critic.name}")
    print(f"Worker {worker_id}: Critic has {len(critic.layers)} layers")
    
    # Check if the critic is actually the same across workers by checking weights
    weights_summary = sum([w.numpy().sum() for w in critic.trainable_weights])
    print(f"Worker {worker_id}: Weights sum: {weights_summary}")
    
    return {
        'worker_id': worker_id,
        'critic_name': critic.name,
        'num_layers': len(critic.layers),
        'weights_sum': float(weights_summary)
    }


def main():
    """Test shared critic across multiple workers."""
    try:
        print("Testing Ray-based shared critic implementation...")
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
        
        print("Ray initialized successfully")
        
        # Clear any existing critics
        SharedCriticStore.clear_critics()
        print("Cleared existing critics")
        
        # Launch multiple workers to test shared critic
        num_workers = 3
        print(f"Launching {num_workers} workers...")
        
        # Submit work to multiple workers
        futures = []
        for i in range(num_workers):
            future = test_worker.remote(f"worker_{i}")
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        
        print("\nResults from all workers:")
        for result in results:
            print(f"  {result}")
        
        # Check if all workers got the same critic
        critic_names = [r['critic_name'] for r in results]
        weights_sums = [r['weights_sum'] for r in results]
        
        print(f"\nCritic names: {critic_names}")
        print(f"Weight sums: {weights_sums}")
        
        # Verify that all workers are using the same critic instance
        same_critic_name = len(set(critic_names)) == 1
        same_weights = len(set([round(w, 6) for w in weights_sums])) == 1  # Allow for small floating point differences
        
        print(f"\nAll workers using same critic name: {same_critic_name}")
        print(f"All workers have same weights: {same_weights}")
        
        if same_critic_name and same_weights:
            print("✅ SUCCESS: Shared critic is working correctly across workers!")
        else:
            print("❌ FAILURE: Workers are not sharing the same critic instance.")
        
        # Clean up
        SharedCriticStore.clear_critics()
        ray.shutdown()
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        try:
            ray.shutdown()
        except:
            pass


if __name__ == "__main__":
    main()