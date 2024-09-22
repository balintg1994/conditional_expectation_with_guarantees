from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import tensorflow.compat.v1 as tf


class BaseEstimator(ABC):
    """
    Base class for the supported regression estimators
    """

    @abstractmethod
    def fit(
        self,
        X_feature: tf.Tensor,
        Y: tf.Tensor,
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def define_feature_matrix(
        self,
        X: tf.Tensor,
    ) -> tf.Tensor:
        pass

    @abstractmethod
    def define_output_tensor(
        self,
        X_feature: tf.Tensor,
    ) -> tf.Tensor:
        pass
