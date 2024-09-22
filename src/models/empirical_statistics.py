from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.models.confidence_interval import ConfidenceInterval
from src.models.quantile_level import QuantileLevel


@dataclass
class EmpiricalStatistics:
    mean: float
    variance: float
    n_samples: int

    @classmethod
    def from_samples(cls, samples: np.ndarray) -> EmpiricalStatistics:
        mean = np.mean(samples)
        variance = np.var(samples, ddof=1)
        return EmpiricalStatistics(
            mean=mean.item(),
            variance=variance.item(),
            n_samples=len(samples),
        )

    def compute_confidence_interval(
        self,
        quantile_level: QuantileLevel,
        one_sided: bool,
    ) -> ConfidenceInterval:
        """Compute the confidence interval for the mean with the given alpha level."""
        if one_sided:
            z_score = stats.norm.ppf(1 - quantile_level.level)
            margin_of_error = z_score * np.sqrt(self.variance / self.n_samples)
            if z_score < 0:
                lower_bound = -np.inf
                upper_bound = self.mean + margin_of_error
            else:
                lower_bound = self.mean - margin_of_error
                upper_bound = np.inf
        else:
            margin_of_error = stats.norm.ppf(1 - quantile_level.level / 2) * np.sqrt(
                self.variance / self.n_samples,
            )
            lower_bound = self.mean - margin_of_error
            upper_bound = self.mean + margin_of_error

        return ConfidenceInterval(
            level=quantile_level,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
