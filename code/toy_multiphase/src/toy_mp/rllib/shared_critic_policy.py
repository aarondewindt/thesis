"""Custom TensorFlow policy with shared critic using Ray's object store."""

import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import ray
import tensorflow as tf

# Configure TensorFlow to avoid GPU memory crashes
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import keras as ks
from typing import Dict, Any, Optional

from ray.rllib.policy.dynamic_tf_policy import DynamicTFPolicy
from ray.rllib.utils.typing import TensorType


@ray.remote
class SharedCriticActor:
    """Ray actor that manages shared critic networks across workers."""
    
    def __init__(self):
        self._critics: Dict[str, ks.Model] = {}
    
    def get_or_create_critic(self, critic_id: str, obs_dim: int) -> ks.Model:
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
    
    def clear_critics(self):
        """Clear all stored critics (useful for testing)."""
        self._critics.clear()


class SharedCriticStore:
    """Ray-based storage for shared critic networks across workers."""
    
    _actor_ref: Optional[Any] = None  # ray.ActorHandle type annotation causes issues
    
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
    def clear_critics(cls):
        """Clear all stored critics (useful for testing)."""
        if cls._actor_ref is not None:
            cls._actor_ref.clear_critics.remote()


class SharedCriticTFPolicy(DynamicTFPolicy):
    """
    Custom TensorFlow policy with:
    - Per-policy actor networks (separate parameters)
    - Shared critic network (shared parameters via Ray object store)
    """
    
    def __init__(self, observation_space, action_space, config):
        # Extract configuration
        self.critic_id = config.get("critic_id", "default_shared_critic")
        
        super().__init__(
            obs_space=observation_space,
            action_space=action_space,
            config=config
        )
        
        # Initialize models
        self._build_models()
        
        # Store last observation for value function
        self._last_obs = None
    
    def _build_models(self):
        """Build actor and critic models."""
        obs_dim = int(self.observation_space.shape[0])
        num_outputs = int(self.action_space.n)
        
        # Actor network - separate for each policy instance
        self.actor = ks.Sequential([
            ks.layers.Input(shape=(obs_dim,)),
            ks.layers.Dense(128, activation="tanh", name="actor_hidden1"),
            ks.layers.Dense(128, activation="tanh", name="actor_hidden2"),
            ks.layers.Dense(num_outputs, activation=None, name="actor_output"),
        ], name=f"actor_{id(self)}")  # Unique name per instance
        
        # Shared critic network - retrieved from Ray object store
        try:
            self.critic = SharedCriticStore.get_shared_critic(
                critic_id=self.critic_id,
                obs_dim=obs_dim
            )
            # Type assertion to help the type checker
            assert hasattr(self.critic, '__call__'), "Critic should be a callable model"
        except Exception as e:
            # Fallback to individual critic if Ray storage fails
            print(f"Warning: Failed to get shared critic ({e}), using individual critic")
            self.critic = ks.Sequential([
                ks.layers.Input(shape=(obs_dim,)),
                ks.layers.Dense(128, activation="tanh"),
                ks.layers.Dense(128, activation="tanh"),
                ks.layers.Dense(1, activation=None),
            ], name=f"fallback_critic_{id(self)}")
    
    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        """Compute actions using the actor network."""
        # Convert observations to tensor
        obs_tensor = tf.cast(obs_batch, tf.float32)
        self._last_obs = obs_tensor
        
        # Get action logits from actor
        action_logits = self.actor(obs_tensor)
        
        # Sample actions from logits
        actions = tf.random.categorical(action_logits, 1)
        actions = tf.squeeze(actions, axis=-1)
        
        # Convert to numpy for RLlib
        actions_np = actions.numpy()
        
        return actions_np, state_batches, {}
    
    def compute_log_likelihoods(self, actions, obs_batch, state_batches=None,
                               prev_action_batch=None, prev_reward_batch=None):
        """Compute log probabilities of given actions."""
        obs_tensor = tf.cast(obs_batch, tf.float32)
        action_logits = self.actor(obs_tensor)
        
        # Compute log probabilities
        log_probs = tf.nn.log_softmax(action_logits)
        actions_tensor = tf.cast(actions, tf.int32)
        
        # Gather log probabilities for taken actions
        action_log_probs = tf.gather(log_probs, actions_tensor, axis=1, batch_dims=1)
        
        return action_log_probs
    
    def get_weights(self):
        """Get trainable weights (actor only, critic is managed separately)."""
        return {"actor": self.actor.get_weights()}
    
    def set_weights(self, weights):
        """Set trainable weights (actor only)."""
        if "actor" in weights:
            self.actor.set_weights(weights["actor"])
    
    def value_function(self):
        """Compute value estimates using shared critic."""
        if self._last_obs is None:
            raise ValueError("value_function() called before compute_actions()")
        
        # Type checker has trouble with Ray actor returns, but this is safe at runtime
        values = self.critic(self._last_obs)  # type: ignore
        return tf.squeeze(values, axis=-1)


def create_shared_critic_policy_spec(obs_space, action_space, critic_id: str = "shared_critic"):
    """Helper function to create PolicySpec for shared critic policy."""
    from ray.rllib.policy.policy import PolicySpec
    
    return PolicySpec(
        policy_class=SharedCriticTFPolicy,
        observation_space=obs_space,
        action_space=action_space,
        config={"critic_id": critic_id}
    )