from __future__ import annotations

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from src.experiments.base_experiment import BaseExperiment
from src.utilities.numerical_precision import NumericalPrecision


class MaxCallOptionPricing(BaseExperiment):
    NAME = "pricing of a 100-dimensional max-call option under a standard multidimensional Black-Scholes model"
    INPUT_DIM = 100

    def __init__(self, include_additional_feature: bool, precision: NumericalPrecision):
        super().__init__(
            include_additional_feature=include_additional_feature,
            precision=precision,
        )
        # TODO: make these parameters
        self.t = tf.constant(
            1.0 / 52.0,
            dtype=self.precision.value["tf_precision"],
        )  # initial time step
        self.T = tf.constant(
            1.0 / 3.0,
            dtype=self.precision.value["tf_precision"],
        )  # terminal time step
        self.tau = self.T - self.t
        self.r = tf.constant(
            0.0,
            dtype=self.precision.value["tf_precision"],
        )
        self.mu = np.zeros(
            self.INPUT_DIM,
            dtype=self.precision.value["np_precision"],
        )
        self.sigmas = np.zeros(
            self.INPUT_DIM,
            dtype=self.precision.value["np_precision"],
        )
        for i in range(self.INPUT_DIM):
            self.sigmas[i] = 0.1 + (i + 1) / 200.0

        self.S0_vec = (self.INPUT_DIM) * [10.0]
        self.K = tf.constant(
            16.3,
            dtype=self.precision.value["tf_precision"],
        )
        mean = np.zeros(
            self.INPUT_DIM,
            dtype=self.precision.value["np_precision"],
        )
        rho = 0.3  # correlation between the BMs
        cov = rho * np.ones(
            (self.INPUT_DIM, self.INPUT_DIM),
            dtype=self.precision.value["np_precision"],
        )
        for i in range(self.INPUT_DIM):
            cov[i, i] = 1
        self.mvn = tfp.distributions.MultivariateNormalTriL(
            loc=mean,
            scale_tril=tf.linalg.cholesky(cov),
        )

    def sample_from_marginal_distribution_of_X(
        self,
        n_samples: int,
        include_additional_feature: bool,
    ) -> tf.Tensor:
        self.S0 = np.zeros(
            (n_samples, self.INPUT_DIM),
            dtype=self.precision.value["np_precision"],
        )
        for mc in range(n_samples):
            self.S0[mc, :] = self.S0_vec

        self.rand1 = tf.cast(
            self.mvn.sample(sample_shape=n_samples),
            dtype=self.precision.value["tf_precision"],
        )  # Brownian Motion increment from 0 to t (dynamics under P)

        X = self.S0 * tf.math.exp(
            (self.mu - 0.5 * self.sigmas**2) * self.t
            + tf.math.sqrt(self.t) * self.sigmas * self.rand1,
        )

        if include_additional_feature:
            additional_feature = tf.expand_dims(tf.reduce_max(X, axis=1), axis=1)
            X = tf.concat((X, additional_feature), axis=1)

        return X

    def sample_from_marginal_distribution_of_V(
        self,
        n_samples: int,
        include_additional_feature: bool,
    ) -> tf.Tensor:
        V = tf.cast(
            self.mvn.sample(sample_shape=n_samples),
            dtype=self.precision.value["tf_precision"],
        )  # Brownian Motion increment from t to T (dynamics under Q)
        return V

    def function_h(self, X: tf.Tensor, V: tf.Tensor) -> tf.Tensor:
        terminal_prices = X * tf.math.exp(
            (self.r - 0.5 * self.sigmas**2) * self.tau
            + tf.math.sqrt(self.tau) * self.sigmas * V,
        )
        payoff = tf.nn.relu(tf.reduce_max(terminal_prices, axis=1) - self.K)
        Y = tf.math.exp(-self.r * self.tau) * payoff
        Y = tf.expand_dims(Y, axis=1)

        return Y
