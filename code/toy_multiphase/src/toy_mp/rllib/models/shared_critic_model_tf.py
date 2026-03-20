import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf
import keras as ks

from ray.rllib.models.tf.tf_modelv2 import TFModelV2


class SharedCriticStore:
    """Singleton-like storage for one critic network shared across policies."""

    critic: ks.Model | None = None


class PhaseActorSharedCriticTFModel(TFModelV2):
    """
    TFModelV2 with:
    - per-policy actor head (each policy has its own actor params)
    - globally shared critic head (all policies reference same critic params)
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
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

        # Critic is shared across all policy instances.
        if SharedCriticStore.critic is None:
            SharedCriticStore.critic = ks.Sequential(
                [
                    ks.layers.Input(shape=(obs_dim,)),
                    ks.layers.Dense(128, activation="tanh"),
                    ks.layers.Dense(1, activation=None),
                ],
                name="shared_critic",
            )

        self.critic = SharedCriticStore.critic
        self._last_obs = None

    def forward(self, input_dict, state, seq_lens):
        obs = tf.cast(input_dict["obs_flat"], tf.float32)
        self._last_obs = obs
        logits = self.actor(obs)
        return logits, state

    def value_function(self):
        if self._last_obs is None:
            raise ValueError("value_function() called before forward()")
        value = self.critic(self._last_obs)
        return tf.reshape(value, [-1])
