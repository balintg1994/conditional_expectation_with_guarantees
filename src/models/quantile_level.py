from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantileLevel:
    level: float

    def __post_init__(self):
        if not 0 < self.level < 1:
            raise ValueError("Quantile level must be between 0 and 1 (exclusive)")
