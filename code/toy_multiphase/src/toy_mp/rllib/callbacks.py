from __future__ import annotations

from ray.rllib.callbacks.callbacks import RLlibCallback
from toy_mp.rllib.models.shared_critic_model_tf import PhaseActorSharedCriticTFModel


class ToyMetricsCallback(RLlibCallback):
    """Minimal callback for success and episode length.

    Callback lifecycle and `on_episode_end` are documented in RLlib.
    
    Also handles synchronization of shared critic weights after training steps.
    """

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs) -> None:
        try:
            ret = float(episode.get_return())
        except Exception:
            ret = None

        if metrics_logger is not None and ret is not None:
            metrics_logger.log_value("success", 1.0 if ret >= 1.0 else 0.0)

        try:
            length = int(episode.length)
        except Exception:
            length = None

        if metrics_logger is not None and length is not None:
            metrics_logger.log_value("ep_len", float(length))
    
    def on_train_result(self, *, algorithm, metrics_logger=None, result=None, **kwargs) -> None:
        """Called after each training iteration to sync shared critic weights."""
        try:
            # Access policies from the algorithm's workers
            worker = getattr(algorithm.workers, 'local_worker', None)
            if worker is None:
                return
                
            policies = getattr(worker, 'policy_map', {})
            for policy in policies.values():
                # Check if policy uses PhaseActorSharedCriticTFModel
                if hasattr(policy, 'model') and isinstance(policy.model, PhaseActorSharedCriticTFModel):
                    # Push critic weights to shared store after training
                    policy.model.push_critic_weights()
        except AttributeError:
            # API might have changed, silently continue
            return
        except Exception as e:
            # Don't fail training if sync fails, just log a warning
            print(f"Warning: Failed to sync shared critic weights: {e}")
