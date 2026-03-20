from __future__ import annotations

from ray.rllib.callbacks.callbacks import RLlibCallback


class ToyMetricsCallback(RLlibCallback):
    """Minimal callback for success and episode length.

    Callback lifecycle and `on_episode_end` are documented in RLlib.
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
