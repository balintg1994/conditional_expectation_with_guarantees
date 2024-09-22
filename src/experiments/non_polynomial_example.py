from __future__ import annotations

import tensorflow.compat.v1 as tf

from src.experiments.base_experiment import BaseExperiment
from src.utilities.numerical_precision import NumericalPrecision


class NonPolynomialExample(BaseExperiment):
    NAME = "five-dimensional non-polynomial example"
    INPUT_DIM = 5

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

        if include_additional_feature:
            additional_feature = (
                5
                * tf.expand_dims(
                    tf.math.logger(5 + ((X[:, 0]) ** 2) * (X[:, 1] ** 2))
                    * tf.tanh((X[:, 2]) * (X[:, 3]) * ((X[:, 4]) ** 2)),
                    axis=1,
                ),
            )
            X = tf.concat((X, additional_feature), axis=1)

        return X

    def sample_from_marginal_distribution_of_V(
        self,
        n_samples: int,
        include_additional_feature: bool,
    ) -> tf.Tensor:
        V = tf.compat.v1.random_normal(
            (n_samples, self.INPUT_DIM),
            dtype=self.precision.value["tf_precision"],
        )
        return V

    def function_h(self, X: tf.Tensor, V: tf.Tensor) -> tf.Tensor:
        Y = 5 * tf.expand_dims(
            tf.math.log(5 + ((X[:, 0] + V[:, 0]) ** 2) * (X[:, 1] ** 2) + V[:, 1] ** 2)
            * tf.tanh(
                (X[:, 2] + V[:, 2]) * (X[:, 3] + V[:, 3]) * ((X[:, 4] + V[:, 4]) ** 2),
            ),
            axis=1,
        )
        return Y
