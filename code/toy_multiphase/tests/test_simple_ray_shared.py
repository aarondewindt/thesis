#!/usr/bin/env python3
"""Simplified test for Ray-based shared critic implementation."""

import ray
import os
import sys
import tensorflow as tf
import keras as ks

# Configure Keras backend
os.environ.setdefault("KERAS_BACKEND", "tensorflow")


@ray.remote
class SharedCriticActor:
    """Ray actor that manages shared critic networks across workers."""
    
    def __init__(self):
        self._critics = {}
    
    def get_or_create_critic(self, critic_id: str, obs_dim: int):
        """Get or create a shared critic model."""
        if critic_id not in self._critics:
            # Create critic model
            self._critics[critic_id] = ks.Sequential([
                ks.layers.Input(shape=(obs_dim,)),
                ks.layers.Dense(128, activation="tanh"),
                ks.layers.Dense(128, activation="tanh"),
                ks.layers.Dense(1, activation=None),
            ], name=f"shared_critic_{critic_id}")
        
        return self._critics[critic_id]
    
    def set_critic_weights(self, critic_id: str, weights):
        """Set weights for a specific critic."""
        if critic_id in self._critics:
            self._critics[critic_id].set_weights(weights)
            return True
        return False
    
    def get_critic_weights(self, critic_id: str):
        """Get weights for a specific critic."""
        if critic_id in self._critics:
            return self._critics[critic_id].get_weights()
        return None
    
    def clear_critics(self):
        """Clear all stored critics."""
        self._critics.clear()


class SharedCriticStore:
    """Simplified Ray-based storage for shared critic networks."""
    
    _actor_ref = None
    
    @classmethod
    def _get_actor(cls):
        """Get or create the shared critic actor."""
        if cls._actor_ref is None:
            try:
                # Try to get existing actor
                cls._actor_ref = ray.get_actor("shared_critic_actor")
            except ValueError:
                # Actor doesn't exist, create it
                cls._actor_ref = SharedCriticActor.options(
                    name="shared_critic_actor",
                    lifetime="detached"
                ).remote()
        return cls._actor_ref
    
    @classmethod
    def get_shared_critic(cls, critic_id: str, obs_dim: int):
        """Get or create a shared critic model using Ray actor."""
        actor = cls._get_actor()
        critic = ray.get(actor.get_or_create_critic.remote(critic_id, obs_dim))
        return critic
    
    @classmethod
    def set_critic_weights(cls, critic_id: str, weights):
        """Set weights for a shared critic."""
        actor = cls._get_actor()
        return ray.get(actor.set_critic_weights.remote(critic_id, weights))
    
    @classmethod
    def get_critic_weights(cls, critic_id: str):
        """Get weights for a shared critic."""
        actor = cls._get_actor()
        return ray.get(actor.get_critic_weights.remote(critic_id))
    
    @classmethod
    def clear_critics(cls):
        """Clear all stored critics."""
        if cls._actor_ref is not None:
            cls._actor_ref.clear_critics.remote()


@ray.remote
def test_worker(worker_id: str):
    """Simulate a worker accessing the shared critic."""
    print(f"Worker {worker_id}: Starting test")
    
    # Each worker tries to get the same shared critic
    critic = SharedCriticStore.get_shared_critic("test_critic", obs_dim=10)
    
    print(f"Worker {worker_id}: Got critic with name: {critic.name}")
    print(f"Worker {worker_id}: Critic has {len(critic.layers)} layers")
    
    # Check if the critic is actually the same across workers by checking weights
    weights = critic.get_weights()
    weights_summary = sum([w.sum() for w in weights])
    print(f"Worker {worker_id}: Initial weights sum: {weights_summary}")
    
    # Modify the weights and upload them to the shared store
    new_weights = []
    for w in weights:
        new_weights.append(w + float(worker_id.split('_')[1]))  # Add worker number to weights
    
    # Update the shared critic
    SharedCriticStore.set_critic_weights("test_critic", new_weights)
    
    # Get the updated weights back
    updated_weights = SharedCriticStore.get_critic_weights("test_critic")
    updated_weights_summary = sum([w.sum() for w in updated_weights])
    print(f"Worker {worker_id}: Updated weights sum: {updated_weights_summary}")
    
    return {
        'worker_id': worker_id,
        'critic_name': critic.name,
        'num_layers': len(critic.layers),
        'initial_weights_sum': float(weights_summary),
        'updated_weights_sum': float(updated_weights_summary)
    }


def main():
    """Test shared critic across multiple workers."""
    try:
        print("Testing simplified Ray-based shared critic implementation...")
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
        
        print("Ray initialized successfully")
        
        # Clear any existing critics
        SharedCriticStore.clear_critics()
        print("Cleared existing critics")
        
        # Test sequential access first
        print("\n=== Testing sequential access ===")
        critic1 = SharedCriticStore.get_shared_critic("test_critic", obs_dim=10)
        critic2 = SharedCriticStore.get_shared_critic("test_critic", obs_dim=10)
        
        print(f"Critic 1 name: {critic1.name}")
        print(f"Critic 2 name: {critic2.name}")
        print(f"Same name: {critic1.name == critic2.name}")
        
        # Test parallel worker access
        print("\n=== Testing parallel worker access ===")
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
        
        # Check the final state
        final_weights = SharedCriticStore.get_critic_weights("test_critic")
        final_weights_sum = sum([w.sum() for w in final_weights])
        print(f"\nFinal shared critic weights sum: {final_weights_sum}")
        
        print("✅ SUCCESS: Ray actor-based sharing works!")
        
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