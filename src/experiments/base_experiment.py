from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

import tensorflow.compat.v1 as tf

from src.estimators.base_estimator import BaseEstimator
from src.utilities.numerical_precision import NumericalPrecision


@dataclass
class DataSet:
    X: tf.Tensor
    V: tf.Tensor
    V_tilde: tf.Tensor
    Y: tf.Tensor
    Z: tf.Tensor


class BaseExperiment(ABC):
    OUTPUT_DIM = 1

    def __init__(
        self,
        include_additional_feature: bool,
        precision: NumericalPrecision,
    ):
        self.include_additional_feature = include_additional_feature
        self.precision = precision

    @abstractmethod
    def sample_from_marginal_distribution_of_X(
        self,
        n_samples: int,
        include_additional_feature: bool,
    ) -> tf.Tensor:
        pass

    @abstractmethod
    def sample_from_marginal_distribution_of_V(
        self,
        n_samples: int,
        include_additional_feature: bool,
    ) -> tf.Tensor:
        pass

    @abstractmethod
    def function_h(self, X: tf.Tensor, V: tf.Tensor) -> tf.Tensor:
        pass

    def generate_monte_carlo_samples(
        self,
        n_samples: int,
    ) -> DataSet:
        X = self.sample_from_marginal_distribution_of_X(
            n_samples=n_samples,
            include_additional_feature=self.include_additional_feature,
        )
        V = self.sample_from_marginal_distribution_of_V(
            n_samples=n_samples,
            include_additional_feature=self.include_additional_feature,
        )
        V_tilde = self.sample_from_marginal_distribution_of_V(
            n_samples=n_samples,
            include_additional_feature=self.include_additional_feature,
        )
        Y = self.function_h(X=X, V=V)
        Z = self.function_h(X=X, V=V_tilde)
        return DataSet(X=X, V=V, V_tilde=V_tilde, Y=Y, Z=Z)

    def define_feature_matrix(self, X: tf.Tensor, estimator: BaseEstimator):
        return estimator.define_feature_matrix(X=X)

    def fit_estimator(
        self,
        X: tf.Tensor,
        Y: tf.Tensor,
        estimator,
        **kwargs,
    ) -> None:
        X_feature = self.define_feature_matrix(
            X=X,
            estimator=estimator,
        )
        estimator.fit(X_feature=X_feature, Y=Y, **kwargs)

    def predict_estimator(
        self,
        X: tf.Tensor,
        estimator: BaseEstimator,
    ) -> tf.Tensor:
        X_feature = self.define_feature_matrix(
            X=X,
            estimator=estimator,
        )
        output_tensor = estimator.define_output_tensor(X_feature=X_feature)
        return output_tensor


class ExperimentNotDefinedError(ValueError):
    pass
