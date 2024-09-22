from __future__ import annotations

import numpy as np
import tensorflow.compat.v1 as tf

from src.estimators.base_estimator import BaseEstimator


class PolynomialRegression(BaseEstimator):
    def __init__(self, degree: int):
        self.degree = degree

    def fit(self, X_feature: tf.Tensor, Y: tf.Tensor, **kwargs) -> None:
        if kwargs.get("OLS_solver") not in [
            "Cholesky",
            "pseudo_inverse_based_on_truncated_SVD",
        ]:
            raise ValueError(
                "Polynomial Regression requires OLS_solver as a kwarg. Please provide either 'Cholesky' or 'pseudo_inverse_based_on_truncated_SVD'.",
            )

        solver_methods = {
            "Cholesky": self._cholesky_solver,
            "pseudo_inverse_based_on_truncated_SVD": self._truncated_svd_solver,
        }

        vandermonde_transpose = tf.transpose(X_feature)
        gram_matrix = tf.matmul(vandermonde_transpose, X_feature)
        b = tf.matmul(vandermonde_transpose, Y)
        ols_equation_solver = solver_methods[kwargs["OLS_solver"]]
        beta = ols_equation_solver(A=gram_matrix, b=b)
        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as training_session:
            init.run()
            self.beta_hat = np.reshape(training_session.run([beta]), -1)

    def define_output_tensor(
        self,
        X_feature: tf.Tensor,
    ) -> tf.Tensor:
        n_features = X_feature.shape[1]
        outputs = self.beta_hat[0]
        for i in range(1, n_features):
            outputs += self.beta_hat[i] * X_feature[:, i]
        outputs = tf.expand_dims(outputs, axis=1)
        return outputs

    def _add_intercept_column_to_feature_matrix(
        self,
        X_feature: tf.Tensor,
    ) -> tf.Tensor:
        batch_size = X_feature.shape[0]
        vandermonde_matrix = tf.concat(
            (
                tf.ones((batch_size, 1), dtype=X_feature.dtype),
                X_feature,
            ),
            axis=1,
        )
        return vandermonde_matrix

    @staticmethod
    def _cholesky_solver(
        A: tf.Tensor,
        b: tf.Tensor,
    ):
        """
        solves Ax = b using the Cholesky decomposition of A.T @ A
        """
        L = tf.linalg.cholesky(A)
        return tf.linalg.cholesky_solve(L, b)

    @staticmethod
    def _truncated_svd_solver(
        A: tf.Tensor,
        b: tf.Tensor,
    ):
        """
        solves Ax = b using the pseudo-inverse based on a truncated SVD decomposition of A
        """
        inverse = tf.linalg.pinv(A, rcond=1e-6)
        return tf.matmul(inverse, b)


class LinearRegression(PolynomialRegression):
    def __init__(
        self,
    ):
        super().__init__(degree=1)

    def define_feature_matrix(
        self,
        X: tf.Tensor,
    ) -> tf.Tensor:
        return self._add_intercept_column_to_feature_matrix(X_feature=X)


class QuadraticPolynomialRegression(PolynomialRegression):
    def __init__(
        self,
        include_mixed_terms: bool = True,
    ):
        super().__init__(degree=2)
        self.include_mixed_terms = include_mixed_terms

    def define_feature_matrix(
        self,
        X: tf.Tensor,
    ):
        if not self.include_mixed_terms:
            X_feature = tf.concat((X, X**2), axis=1)
            return self._add_intercept_column_to_feature_matrix(X_feature)

        num_features = X.shape[1]
        X_feature_list = [X]
        for i in range(num_features):
            for j in range(i, num_features):
                mixed_term = tf.expand_dims(X[:, i] * X[:, j], axis=1)
                X_feature_list.append(mixed_term)

        X_feature = tf.concat(X_feature_list, axis=1)
        return self._add_intercept_column_to_feature_matrix(X_feature)
