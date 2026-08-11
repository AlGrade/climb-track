"""Timestamp-aware One Euro filter implementation."""

import math


class OneEuro2D:
    """Adaptive low-pass filter that keeps fast intentional motion responsive."""

    def __init__(
        self,
        *,
        min_cutoff: float,
        beta: float,
        derivative_cutoff: float,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self._timestamp: float | None = None
        self._raw: tuple[float, float] | None = None
        self._filtered: tuple[float, float] | None = None
        self._derivative = (0.0, 0.0)

    def update(self, timestamp: float, x: float, y: float) -> tuple[float, float]:
        """Filter one position; the first sample passes through unchanged."""
        if self._timestamp is None or self._raw is None or self._filtered is None:
            self._timestamp = timestamp
            self._raw = (x, y)
            self._filtered = (x, y)
            return x, y
        dt = timestamp - self._timestamp
        if dt <= 0:
            raise ValueError("One Euro timestamps must be strictly increasing")

        derivative = ((x - self._raw[0]) / dt, (y - self._raw[1]) / dt)
        derivative_alpha = _alpha(dt, self.derivative_cutoff)
        filtered_derivative = (
            _mix(self._derivative[0], derivative[0], derivative_alpha),
            _mix(self._derivative[1], derivative[1], derivative_alpha),
        )
        speed = math.hypot(*filtered_derivative)
        position_alpha = _alpha(dt, self.min_cutoff + self.beta * speed)
        filtered = (
            _mix(self._filtered[0], x, position_alpha),
            _mix(self._filtered[1], y, position_alpha),
        )
        self._timestamp = timestamp
        self._raw = (x, y)
        self._filtered = filtered
        self._derivative = filtered_derivative
        return filtered


def _alpha(dt: float, cutoff: float) -> float:
    time_constant = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + time_constant / dt)


def _mix(previous: float, current: float, alpha: float) -> float:
    return alpha * current + (1.0 - alpha) * previous
