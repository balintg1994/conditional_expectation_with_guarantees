from __future__ import annotations

import tensorflow.compat.v1 as tf

from src.experiments.base_experiment import BaseExperiment
from src.utilities.numerical_precision import NumericalPrecision


class PolynomialExample(BaseExperiment):
    NAME = "four-dimensional polynomial example"
    INPUT_DIM = 4

    def __init__(self, include_additional_feature: bool, precision: NumericalPrecision):
        super().__init__(
            include_additional_feature=include_additional_feature,
            precision=precision,
        )

    def sample_from_marginal_distribution_of_X(
        self,
        n_samples: int,
        include_additional_feature: bool,
    ) -> tf.Tensor:
        X = tf.compat.v1.random_normal(
            (n_samples, self.INPUT_DIM),
            dtype=self.precision.value["tf_precision"],
        )

        if include_additional_feature:  # TODO: move to base class (maybe?)
            additional_feature = tf.expand_dims(
                X[:, 0] + X[:, 1] ** 2 + X[:, 2] * X[:, 3],
                axis=1,
            )
            X = tf.concat((X, additional_feature), axis=1)

        return X

    def sample_from_marginal_distribution_of_V(
        self,
        n_samples: int,
        include_additional_feature: bool,
    ) -> tf.Tensor:
        V = tf.compat.v1.random_normal(
            (n_samples, self.OUTPUT_DIM),
            dtype=self.precision.value["tf_precision"],
        )
        return V

    def function_h(self, X: tf.Tensor, V: tf.Tensor) -> tf.Tensor:
        Y = tf.expand_dims(
            X[:, 0] + X[:, 1] ** 2 + X[:, 2] * X[:, 3],
            axis=1,
        )
        Y += V

        return Y
