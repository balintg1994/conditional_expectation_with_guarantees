from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from src.conditional_expectation_with_guarantees import (
    ConditionalExpectationWithGuarantees,
)
from src.estimators.polynomial_regression import LinearRegression
from src.experiments.non_polynomial_example import NonPolynomialExample
from src.models.quantile_level import QuantileLevel
from src.utilities.numerical_precision import NumericalPrecision


@hydra.main(config_path="config_files", config_name="config", version_base="1.1")
def run_experiment(cfg: DictConfig):
    experiment = NonPolynomialExample(
        include_additional_feature=cfg.experiment.include_additional_feature,
        precision=NumericalPrecision[cfg.experiment.precision],
    )
    estimator = LinearRegression()

    cond_exp_guar = ConditionalExpectationWithGuarantees(
        experiment=experiment,
        estimator=estimator,
    )
    results = cond_exp_guar.run(
        n_training_samples=cfg.runtime.n_training_samples,
        n_test_batches=cfg.runtime.n_test_batches,
        n_test_samples_per_batch=cfg.runtime.n_test_samples_per_batch,
    )

    quantile_level = QuantileLevel(level=cfg.quantile_level.level)
    results.print_major_and_appendix_tables(quantile_level)

    output_file_path = os.path.join(os.getcwd(), "results.xlsx")

    results.export_major_and_appendix_tables_to_excel(
        quantile_level=quantile_level,
        filename=output_file_path,
    )


if __name__ == "__main__":
    run_experiment()
