"""Drift monitor — wraps River's ADWIN detector."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from river.drift import ADWIN

logger = logging.getLogger("audio_drift")


@dataclass
class DriftEvent:
    """Record of a single detected drift event."""
    timestamp: int
    drift_score: float
    mean_loss: float
    loss_variance: float


class DriftMonitor:
    """Online drift detector backed by ADWIN.

    Feed drift scores one at a time; the monitor logs detected drifts.
    """

    def __init__(
        self,
        delta: float = 0.002,
        clock: int | None = None,
        grace_period: int | None = None,
        min_window_length: int | None = None,
    ) -> None:
        """Initialise the drift monitor.

        Args:
            delta: ADWIN sensitivity parameter (smaller → more sensitive).
            clock: how often (in observations) ADWIN runs its drift check.
                river's default is 32; left as None it keeps that default so
                existing pipelines are unchanged. Pass clock=1 to check every
                observation (removes the check-interval contribution to
                detection latency — important when comparing window sizes).
            grace_period, min_window_length: optional ADWIN parameters; None
                keeps river's defaults.
        """
        self._adwin_kwargs: dict[str, Any] = {"delta": delta}
        if clock is not None:
            self._adwin_kwargs["clock"] = clock
        if grace_period is not None:
            self._adwin_kwargs["grace_period"] = grace_period
        if min_window_length is not None:
            self._adwin_kwargs["min_window_length"] = min_window_length
        self.adwin = ADWIN(**self._adwin_kwargs)
        self.drift_events: list[DriftEvent] = []
        self.scores: list[float] = []
        self.mean_losses: list[float] = []
        self.variances: list[float] = []

    def update(
        self,
        timestamp: int,
        drift_score: float,
        mean_loss: float,
        loss_variance: float,
    ) -> bool:
        """Feed a new observation and check for drift.

        Args:
            timestamp: Current time step.
            drift_score: Combined drift score.
            mean_loss: Mean reconstruction loss.
            loss_variance: Ensemble loss variance.

        Returns:
            ``True`` if drift is detected at this step.
        """
        self.scores.append(drift_score)
        self.mean_losses.append(mean_loss)
        self.variances.append(loss_variance)

        self.adwin.update(drift_score)
        if self.adwin.drift_detected:
            event = DriftEvent(timestamp, drift_score, mean_loss, loss_variance)
            self.drift_events.append(event)
            logger.warning(
                "  DRIFT DETECTED at t=%d  (score=%.4f, mean_loss=%.4f, var=%.4f)",
                timestamp, drift_score, mean_loss, loss_variance,
            )
            return True
        return False

    def reset(self) -> None:
        """Reset the detector state (e.g. between experiments)."""
        self.adwin = ADWIN(**self._adwin_kwargs)
        self.drift_events.clear()
        self.scores.clear()
        self.mean_losses.clear()
        self.variances.clear()

    @property
    def detected_timestamps(self) -> list[int]:
        """Return list of timestamps where drift was detected."""
        return [e.timestamp for e in self.drift_events]