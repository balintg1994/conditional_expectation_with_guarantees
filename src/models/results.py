from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.models.confidence_interval import ConfidenceInterval
from src.models.empirical_statistics import EmpiricalStatistics
from src.models.exceptions import SampleSizeMismatchError
from src.models.quantile_level import QuantileLevel


@dataclass
class OutputSamples:
    U: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    D: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    F: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    C: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    def append_samples(self, batch_output_samples):
        """Appends samples from a batch to the existing arrays."""
        self.U = np.concatenate([self.U, batch_output_samples.U.reshape(-1)])
        self.D = np.concatenate([self.D, batch_output_samples.D.reshape(-1)])
        self.F = np.concatenate([self.F, batch_output_samples.F.reshape(-1)])
        self.C = np.concatenate([self.C, batch_output_samples.C.reshape(-1)])
        self._ensure_equal_sample_sizes()

    def _ensure_equal_sample_sizes(self):
        """Ensures that U, D, F, and C have the same number of samples. Raises error if not."""
        sizes = [len(self.U), len(self.D), len(self.F), len(self.C)]
        if not all(size == sizes[0] for size in sizes):
            raise SampleSizeMismatchError(
                "Mismatch in sample sizes among U, D, F, and C.",
            )


@dataclass
class MajorResults:
    U_confidence_interval: ConfidenceInterval
    D_confidence_interval: ConfidenceInterval
    rel_error: float
    rel_error_confidence_bound: ConfidenceInterval

    def print_results_table(self):
        """Prints a pretty table containing the results."""
        headers = ["Statistic", "Lower Bound", "Upper Bound", "Estimate"]
        data = [
            [
                "U",
                np.round(self.U_confidence_interval.lower_bound, 5),
                np.round(self.U_confidence_interval.upper_bound, 5),
                "--",
            ],
            [
                "D",
                np.round(self.D_confidence_interval.lower_bound, 5),
                np.round(self.D_confidence_interval.upper_bound, 5),
                "--",
            ],
            ["Relative Error", "--", "--", f"{np.round(100 * self.rel_error, 2)}%"],
            [
                "Relative Error CB",
                f"{np.round(100 * self.rel_error_confidence_bound.lower_bound, 2)}%",
                f"{np.round(100 * self.rel_error_confidence_bound.upper_bound, 2)}%",
                "--",
            ],
        ]
        print(tabulate(data, headers=headers, tablefmt="grid"))

    def export_to_excel(self, writer):
        """Export major results to an Excel file."""
        major_data = {
            "Statistic": ["U", "D", "Relative Error", "Relative Error CB"],
            "Lower Bound": [
                self.U_confidence_interval.lower_bound,
                self.D_confidence_interval.lower_bound,
                np.nan,
                self.rel_error_confidence_bound.lower_bound,
            ],
            "Upper Bound": [
                self.U_confidence_interval.upper_bound,
                self.D_confidence_interval.upper_bound,
                np.nan,
                self.rel_error_confidence_bound.upper_bound,
            ],
            "Estimate": [np.nan, np.nan, self.rel_error, np.nan],
        }
        major_df = pd.DataFrame(major_data)
        major_df.to_excel(writer, sheet_name="Major Results", index=False)


@dataclass
class AppendixResults:
    U_empirical_mean: float
    U_empirical_std_error: float
    D_empirical_mean: float
    D_empirical_std_error: float
    F_empirical_mean: float
    F_empirical_std_error: float
    C_empirical_mean: float
    C_empirical_std_error: float

    def print_results_table(self):
        """Prints a pretty table containing the results."""
        headers = ["Statistic", "Mean", "Standard Error"]
        data = [
            [
                "U",
                np.round(self.U_empirical_mean, 5),
                np.round(self.U_empirical_std_error, 5),
            ],
            [
                "D",
                np.round(self.D_empirical_mean, 5),
                np.round(self.D_empirical_std_error, 5),
            ],
            [
                "F",
                np.round(self.F_empirical_mean, 5),
                np.round(self.F_empirical_std_error, 5),
            ],
            [
                "C",
                np.round(self.C_empirical_mean, 5),
                np.round(self.C_empirical_std_error, 5),
            ],
        ]
        print(tabulate(data, headers=headers, tablefmt="grid"))

    def export_to_excel(self, writer):
        """Export appendix results to an Excel file."""
        appendix_data = {
            "Statistic": ["U", "D", "F", "C"],
            "Mean": [
                self.U_empirical_mean,
                self.D_empirical_mean,
                self.F_empirical_mean,
                self.C_empirical_mean,
            ],
            "Standard Error": [
                self.U_empirical_std_error,
                self.D_empirical_std_error,
                self.F_empirical_std_error,
                self.C_empirical_std_error,
            ],
        }
        appendix_df = pd.DataFrame(appendix_data)
        appendix_df.to_excel(writer, sheet_name="Appendix Results", index=False)


@dataclass
class ConditionalExpectationWithGuaranteesResults:
    U_statistics: EmpiricalStatistics
    D_statistics: EmpiricalStatistics
    F_statistics: EmpiricalStatistics
    C_statistics: EmpiricalStatistics

    @classmethod
    def from_output_samples(cls, output_samples: OutputSamples):
        U_statistics = EmpiricalStatistics.from_samples(output_samples.U)
        D_statistics = EmpiricalStatistics.from_samples(output_samples.D)
        F_statistics = EmpiricalStatistics.from_samples(output_samples.F)
        C_statistics = EmpiricalStatistics.from_samples(output_samples.C)

        return ConditionalExpectationWithGuaranteesResults(
            U_statistics=U_statistics,
            D_statistics=D_statistics,
            F_statistics=F_statistics,
            C_statistics=C_statistics,
        )

    def get_major_results(self, quantile_level: QuantileLevel):
        U_confidence_interval = self.U_statistics.compute_confidence_interval(
            quantile_level,
            one_sided=False,
        )
        D_confidence_interval = self.D_statistics.compute_confidence_interval(
            quantile_level,
            one_sided=False,
        )
        rel_error = np.sqrt(self.F_statistics.mean / self.C_statistics.mean)
        F_confidence_interval = self.F_statistics.compute_confidence_interval(
            quantile_level,
            one_sided=True,
        )
        rel_error_confidence_bound = ConfidenceInterval(
            level=quantile_level,
            lower_bound=0.0,
            upper_bound=np.sqrt(
                F_confidence_interval.upper_bound / self.C_statistics.mean,
            ),
        )
        return MajorResults(
            U_confidence_interval=U_confidence_interval,
            D_confidence_interval=D_confidence_interval,
            rel_error=rel_error,
            rel_error_confidence_bound=rel_error_confidence_bound,
        )

    def get_appendix_results(self):
        return AppendixResults(
            U_empirical_mean=self.U_statistics.mean,
            U_empirical_std_error=np.sqrt(
                self.U_statistics.variance / self.U_statistics.n_samples,
            ),
            D_empirical_mean=self.D_statistics.mean,
            D_empirical_std_error=np.sqrt(
                self.D_statistics.variance / self.D_statistics.n_samples,
            ),
            F_empirical_mean=self.F_statistics.mean,
            F_empirical_std_error=np.sqrt(
                self.F_statistics.variance / self.F_statistics.n_samples,
            ),
            C_empirical_mean=self.C_statistics.mean,
            C_empirical_std_error=np.sqrt(
                self.C_statistics.variance / self.C_statistics.n_samples,
            ),
        )

    def print_major_and_appendix_tables(self, quantile_level: QuantileLevel):
        major_results = self.get_major_results(quantile_level)
        appendix_results = self.get_appendix_results()
        print("Major Results:")
        major_results.print_results_table()
        print("\nAppendix Results:")
        appendix_results.print_results_table()

    def export_major_and_appendix_tables_to_excel(
        self,
        quantile_level: QuantileLevel,
        filename,
    ):
        """Export major and appendix tables to an Excel file."""
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            major_results = self.get_major_results(quantile_level)
            appendix_results = self.get_appendix_results()
            major_results.export_to_excel(writer)
            appendix_results.export_to_excel(writer)
