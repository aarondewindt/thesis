"""
DEPRECATED: This implementation has issues with multi-worker setups.

Use the new shared_critic_policy.py instead, which provides:
- Proper multi-worker support via Ray's object store
- Better isolation between different experiments
- More robust parameter sharing

This file is kept for backward compatibility but should not be used for new code.
"""

import os
from typing import Any, cast
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import ray
import tensorflow as tf

# Configure TensorFlow to avoid GPU memory crashes
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import keras as ks

from ray.rllib.models.tf.tf_modelv2 import TFModelV2


@ray.remote
class SharedCriticRayActor:
    """Ray actor that manages shared critic networks across workers."""
    
    def __init__(self):
        # Store critic weights and metadata instead of models
        self._critic_weights: dict[str, list] = {}
        self._critic_metadata: dict[str, dict] = {}
    
    def get_or_create_critic_weights(self, critic_id: str, obs_dim: int):
        """Get or create shared critic weights and return them with metadata."""
        if critic_id not in self._critic_weights:
            # Create a temporary critic model to get initial weights
            temp_critic = ks.Sequential([
                ks.layers.Input(shape=(obs_dim,)),
                ks.layers.Dense(128, activation="tanh"),
                ks.layers.Dense(128, activation="tanh"),
                ks.layers.Dense(1, activation=None),
            ], name=f"shared_critic_{critic_id}")
            
            # Store the weights and metadata
            self._critic_weights[critic_id] = temp_critic.get_weights()
            self._critic_metadata[critic_id] = {
                'obs_dim': obs_dim,
                'name': f"shared_critic_{critic_id}",
                'layer_configs': [
                    {'type': 'Dense', 'units': 128, 'activation': 'tanh'},
                    {'type': 'Dense', 'units': 128, 'activation': 'tanh'},
                    {'type': 'Dense', 'units': 1, 'activation': None}
                ]
            }
        
        return {
            'weights': self._critic_weights[critic_id],
            'metadata': self._critic_metadata[critic_id]
        }
    
    def update_critic_weights(self, critic_id: str, weights):
        """Update shared critic weights."""
        if critic_id in self._critic_weights:
            self._critic_weights[critic_id] = weights
            return True
        return False
    
    def get_critic_weights(self, critic_id: str):
        """Get shared critic weights."""
        return self._critic_weights.get(critic_id, None)
    
    def clear_critics(self):
        """Clear all stored critics."""
        self._critic_weights.clear()
        self._critic_metadata.clear()


class SharedCriticStore:
    """Ray-based storage for shared critic networks across workers."""

    _actor_ref: Any | None = None

    @classmethod
    def _get_actor(cls):
        """Get or create the shared critic actor."""
        if cls._actor_ref is None:
            try:
                # Try to get existing actor
                cls._actor_ref = ray.get_actor("shared_critic_actor")
            except ValueError:
                # Actor doesn't exist, create it
                cls._actor_ref = SharedCriticRayActor.options(
                    name="shared_critic_actor",
                    lifetime="detached"
                ).remote()
        return cls._actor_ref
    
    @classmethod
    def get_shared_critic(cls, critic_id: str, obs_dim: int):
        """Get or create a shared critic model using weights from Ray actor."""
        actor = cls._get_actor()
        critic_data = ray.get(actor.get_or_create_critic_weights.remote(critic_id, obs_dim))
        
        # Create a local critic model with the shared weights
        metadata = critic_data['metadata']
        critic = ks.Sequential([
            ks.layers.Input(shape=(obs_dim,)),
            ks.layers.Dense(128, activation="tanh"),
            ks.layers.Dense(128, activation="tanh"),
            ks.layers.Dense(1, activation=None),
        ], name=metadata['name'])
        
        # Set the shared weights
        critic.set_weights(critic_data['weights'])
        
        return critic
    
    @classmethod
    def update_shared_critic_weights(cls, critic_id: str, weights):
        """Update the shared critic weights in the Ray actor."""
        actor = cls._get_actor()
        return ray.get(actor.update_critic_weights.remote(critic_id, weights))
    
    @classmethod
    def get_critic_weights(cls, critic_id: str):
        """Get weights for a shared critic."""
        actor = cls._get_actor()
        return ray.get(actor.get_critic_weights.remote(critic_id))
    
    @classmethod
    def clear_critics(cls):
        if cls._actor_ref is not None:
            cls._actor_ref.clear_critics.remote()


class PhaseActorSharedCriticTFModel(TFModelV2):
    """
    TFModelV2 with:
    - per-policy actor head (each policy has its own actor params)
    - globally shared critic head (all policies reference same critic params)
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, critic_id: str):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        obs_dim = int(obs_space.shape[0])

        # Actor is policy-local (one actor per policy instance).
        self.actor = ks.Sequential(
            [
                ks.layers.Input(shape=(obs_dim,)),
                ks.layers.Dense(128, activation="tanh"),
                ks.layers.Dense(num_outputs, activation=None),
            ],
            name=f"{name}_actor",
        )

        self.critic_id = critic_id
        self.critic = SharedCriticStore.get_shared_critic(critic_id, obs_dim)
        self._last_obs = None
        
        # Track if we need to sync weights
        self._weights_synced = True
        
        # Set up automatic weight synchronization after training steps
        self._setup_critic_sync()

    def forward(self, input_dict, state, seq_lens):
        obs = tf.cast(input_dict["obs_flat"], tf.float32)
        self._last_obs = obs
        
        # Sync critic weights before forward pass
        self._pull_critic_weights()
        
        logits = self.actor(obs)
        return logits, state

    def value_function(self):
        if self._last_obs is None:
            raise ValueError("value_function() called before forward()")
        
        # Sync critic weights before value computation
        self._pull_critic_weights()
        
        value = self.critic(self._last_obs)  # type: ignore
        return tf.reshape(value, [-1])
    
    def _pull_critic_weights(self):
        """Pull the latest critic weights from shared storage."""
        try:
            shared_weights = SharedCriticStore.get_critic_weights(self.critic_id)
            if shared_weights is not None:
                self.critic.set_weights(shared_weights)
                self._weights_synced = True
        except Exception as e:
            # If sync fails, continue with current weights
            pass
    
    def push_critic_weights(self):
        """Public method to push current critic weights to shared storage."""
        self._push_critic_weights()
    
    def _push_critic_weights(self):
        """Push current critic weights to shared storage."""
        try:
            current_weights = self.critic.get_weights()
            SharedCriticStore.update_shared_critic_weights(self.critic_id, current_weights)
            self._weights_synced = True
        except Exception as e:
            # If sync fails, mark as unsynced
            self._weights_synced = False
    
    def update_critic_weights(self, new_weights):
        """Update critic weights and sync to shared storage."""
        self.critic.set_weights(new_weights)
        self._push_critic_weights()
    
    def get_weights(self):
        """Override get_weights to ensure we have latest critic weights."""
        self._pull_critic_weights()
        return {
            "actor": self.actor.get_weights(),
            "critic": self.critic.get_weights(),
        }
    
    def set_weights(self, weights_dict):
        """Override set_weights to sync critic weights to shared storage."""
        if "actor" in weights_dict:
            self.actor.set_weights(weights_dict["actor"])
        if "critic" in weights_dict:
            self.critic.set_weights(weights_dict["critic"])
            # Push critic weights to shared store after setting them
            self._push_critic_weights()
    
    def _setup_critic_sync(self):
        """Set up automatic synchronization of critic weights after training."""
        # This is a more advanced approach that would require hooking into
        # TensorFlow's optimizer apply_gradients calls, but it's complex.
        # For now, we rely on the manual _push_critic_weights() calls in training code.
        pass
    
    def post_training_step(self):
        """Call this after each training step to sync critic weights."""
        self._push_critic_weights()
