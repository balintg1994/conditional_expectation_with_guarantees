from __future__ import annotations

import numpy as np
import tensorflow.compat.v1 as tf

from src.estimators.base_estimator import BaseEstimator
from src.estimators.neural_network import NeuralNetworkRegression
from src.estimators.polynomial_regression import PolynomialRegression
from src.experiments.base_experiment import BaseExperiment
from src.experiments.base_experiment import DataSet
from src.experiments.non_polynomial_example import NonPolynomialExample
from src.experiments.polynomial_example import PolynomialExample
from src.models.results import ConditionalExpectationWithGuaranteesResults
from src.models.results import OutputSamples

tf.disable_v2_behavior()


class ConditionalExpectationWithGuarantees:
    def __init__(
        self,
        experiment: BaseExperiment,
        estimator: BaseEstimator,
    ):
        self.experiment = experiment
        self.estimator = estimator

    def run(
        self,
        n_training_samples: int,
        n_test_batches: int,
        n_test_samples_per_batch: int,
    ) -> ConditionalExpectationWithGuaranteesResults:
        training_data = self.generate_monte_carlo_samples_from_experiment(
            # # TODO: should be 2000000 for polynomial and non polynomial examples, 500000 for high dimensional financial examples
            n_samples=n_training_samples,
        )
        self.fit_estimator(
            X=training_data.X,
            Y=training_data.Y,
        )

        output_samples = self._collect_output_samples_in_batches(
            n_test_batches,
            n_test_samples_per_batch,
        )

        return ConditionalExpectationWithGuaranteesResults.from_output_samples(
            output_samples,
        )

    def _collect_output_samples_in_batches(
        self,
        n_test_batches: int,
        n_test_samples_per_batch: int,
    ) -> OutputSamples:
        output_samples = OutputSamples()
        for _ in range(n_test_batches):
            test_data_batch = self.generate_monte_carlo_samples_from_experiment(
                n_samples=n_test_samples_per_batch,
                # TODO: should be 100000 for polynomial and non polynomial examples, 10000 for high dimensional financial examples
            )

            batch_output_samples = self.compute_U_D_F_C_samples(
                test_data=test_data_batch,
            )
            output_samples.append_samples(batch_output_samples)
        return output_samples

    def generate_monte_carlo_samples_from_experiment(self, n_samples: int) -> DataSet:
        return self.experiment.generate_monte_carlo_samples(
            n_samples=n_samples,
        )

    def fit_estimator(self, X: tf.Tensor, Y: tf.Tensor):
        # TODO: these kwargs should not be defined inside this class
        kwargs: dict = {}
        if isinstance(self.estimator, PolynomialRegression):
            is_low_dimensional_experiment = isinstance(
                self.experiment,
                (PolynomialExample, NonPolynomialExample),
            )
            use_cholesky_decomposition = (
                is_low_dimensional_experiment
                and not self.experiment.include_additional_feature
            )
            ols_solver = (
                "Cholesky"
                if use_cholesky_decomposition
                else "pseudo_inverse_based_on_truncated_SVD"
            )
            kwargs["OLS_solver"] = ols_solver

        if isinstance(self.estimator, NeuralNetworkRegression):
            kwargs["n_iterations"] = (
                1000
                # TODO: make this a user parameters + together with other SGD parameters create a dedicated dataclass
            )
        self.experiment.fit_estimator(
            X=X,
            Y=Y,
            estimator=self.estimator,
            **kwargs,
        )

    def compute_U_D_F_C_samples(self, test_data: DataSet) -> OutputSamples:
        output_tensor = self.experiment.predict_estimator(
            X=test_data.X,
            estimator=self.estimator,
        )
        init = tf.compat.v1.global_variables_initializer()
        # TODO: neural network regression might require a tf.Saver object
        with tf.compat.v1.Session() as test_session:
            init.run()
            Y_samples, Z_samples, predictions = test_session.run(
                [
                    test_data.Y,
                    test_data.Z,
                    output_tensor,
                ],
            )
        U_samples = self._compute_U_samples(
            Y_samples=Y_samples,
            predictions=predictions,
        )

        D_samples = self._compute_D_samples(
            Y_samples=Y_samples,
            Z_samples=Z_samples,
        )

        F_samples = self._compute_F_samples(
            Y_samples=Y_samples,
            Z_samples=Z_samples,
            predictions=predictions,
        )

        C_samples = self._compute_C_samples(
            Y_samples=Y_samples,
            Z_samples=Z_samples,
        )

        return OutputSamples(
            U=U_samples,
            D=D_samples,
            F=F_samples,
            C=C_samples,
        )

    @staticmethod
    def _compute_U_samples(
        Y_samples: np.ndarray,
        predictions: np.ndarray,
    ) -> np.ndarray:
        return (Y_samples - predictions) ** 2

    @staticmethod
    def _compute_D_samples(
        Y_samples: np.ndarray,
        Z_samples: np.ndarray,
    ) -> np.ndarray:
        return Y_samples * (Y_samples - Z_samples)

    @staticmethod
    def _compute_F_samples(
        Y_samples: np.ndarray,
        Z_samples: np.ndarray,
        predictions: np.ndarray,
    ) -> np.ndarray:
        return Y_samples * Z_samples + predictions * (
            predictions - Y_samples - Z_samples
        )

    @staticmethod
    def _compute_C_samples(
        Y_samples: np.ndarray,
        Z_samples: np.ndarray,
    ) -> np.ndarray:
        return Y_samples * Z_samples
