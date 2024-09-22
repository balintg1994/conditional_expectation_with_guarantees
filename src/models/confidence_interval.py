from __future__ import annotations

from dataclasses import dataclass

from src.models.quantile_level import QuantileLevel


@dataclass
class ConfidenceInterval:
    level: QuantileLevel
    lower_bound: float
    upper_bound: float
