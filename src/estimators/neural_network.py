from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar
from typing import Callable
from typing import Literal

import tensorflow.compat.v1 as tf
from tqdm import tqdm

from src.estimators.base_estimator import BaseEstimator


def log_sum_exp(x, alpha=0.01):
    return tf.math.log(tf.math.exp(x) + tf.math.exp(alpha * x))


ActivationFunction = Callable[[tf.Tensor], tf.Tensor]

ACTIVATION_FUNCTIONS: dict[str, ActivationFunction] = {
    "tf.nn.relu": tf.nn.relu,
    "tf.nn.tanh": tf.nn.tanh,
    "log_sum_exp": log_sum_exp,
}


@dataclass
class NeuralNetworkParameters:
    n_layers: int
    n_neurons: list[int]
    activation_function_input: InitVar[
        Literal["tf.nn.relu", "tf.nn.tanh", "log_sum_exp"]
    ]
    readout_function_input: InitVar[
        Literal["tf.nn.relu", "tf.nn.tanh", "log_sum_exp"] | None
    ] = None
    use_batch_normalization: bool = True
    activation_function: ActivationFunction = field(init=False)
    readout_function: ActivationFunction | None = field(init=False, default=None)

    def __post_init__(self, activation_function_input, readout_function_input):
        self.activation_function = ACTIVATION_FUNCTIONS[activation_function_input]
        if readout_function_input:
            self.readout_function = ACTIVATION_FUNCTIONS[readout_function_input]


class NeuralNetworkRegression(BaseEstimator):
    def __init__(self, parameters: NeuralNetworkParameters):
        self.parameters = parameters

    def fit(
        self,
        X_feature: tf.Tensor,
        Y: tf.Tensor,
        **kwargs,
    ) -> None:
        if "n_iterations" not in kwargs.keys():
            raise ValueError(
                "Neural Network Regression is trained using a variant of Stochastic Gradient Descent and thus requires 'n_iterations' to be defined. ",
            )
        outputs = self.define_output_tensor(X_feature=X_feature)
        training_op = self._define_neural_network_optimisation_node(
            Y=Y,
            outputs=outputs,
        )

        init = tf.compat.v1.global_variables_initializer()
        if self.parameters.use_batch_normalization:
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.compat.v1.Session() as training_session:
            init.run()

            for _ in tqdm(range(kwargs["n_iterations"])):
                if self.parameters.use_batch_normalization:
                    training_session.run(
                        [training_op, extra_update_ops],
                        feed_dict={self.training: True},
                    )
                else:
                    training_session.run(training_op)

    def define_output_tensor(
        self,
        X_feature: tf.Tensor,
    ) -> tf.Tensor:
        self.training = tf.compat.v1.placeholder_with_default(
            False,
            shape=(),
            name="training",
        )

        with tf.compat.v1.variable_scope(
            "neural_network",
            reuse=tf.compat.v1.AUTO_REUSE,
        ):  # TODO: confirm that AUTO_REUSE is correct
            hidden_layers: list[tf.layers] = []

            for i in range(self.parameters.n_layers):
                with tf.compat.v1.variable_scope(f"hidden_layer_{i + 1}"):
                    activation_fn = (
                        None
                        if self.parameters.use_batch_normalization
                        else self.parameters.activation_function
                    )
                    hidden_layer = self._dense_layer(
                        inputs=X_feature if i == 0 else hidden_layers[-1],
                        output_units=self.parameters.n_neurons[i],
                        activation=activation_fn,
                        name=f"hidden_{i + 1}",
                    )

                    if self.parameters.use_batch_normalization:
                        hidden_layer = self._batch_norm_layer(
                            hidden_layer,
                            self.training,
                        )
                        hidden_layer = self.parameters.activation_function(hidden_layer)

                    hidden_layers.append(hidden_layer)

            # Output layer
            outputs = self._dense_layer(
                inputs=hidden_layers[-1],
                output_units=1,  # TODO: link to self.experiment.OUTPUT_DIM,
                activation=None,
                name="output_layer",
            )

            if self.parameters.readout_function is not None:
                outputs = self.parameters.readout_function(outputs)

        return outputs

    def define_feature_matrix(
        self,
        X: tf.Tensor,
    ) -> tf.Tensor:
        return X

    def _dense_layer(self, inputs, output_units, activation=None, name=None):
        return tf.layers.dense(
            inputs=inputs,
            units=output_units,
            activation=activation,
            kernel_initializer=tf.initializers.glorot_uniform(),
            bias_initializer=tf.initializers.constant(0.0),
            name=name,
        )

    def _batch_norm_layer(self, inputs, training):
        return tf.layers.batch_normalization(
            inputs=inputs,
            training=training,
            momentum=0.9,
        )

    def _setup_learning_rate(self):
        """Configures the learning rate with a piecewise constant decay."""
        lr_boundaries = [1000, 5000, 25000, 50000, 100000, 150000]
        lr_values = [0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

        global_step = tf.compat.v1.train.get_or_create_global_step()
        learning_rate = tf.compat.v1.train.piecewise_constant(
            global_step,
            lr_boundaries,
            lr_values,
        )

        return learning_rate, global_step

    def _define_loss(self, Y, outputs):
        """Defines the loss function for the optimization."""
        return tf.reduce_mean(tf.square(Y - outputs))  # L2 loss

    def _define_optimizer(self, learning_rate, loss, global_step):
        """Sets up the optimizer and training operation."""
        update_ops = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.UPDATE_OPS,
            scope="neural_network",
        )
        with tf.control_dependencies(
            update_ops,
        ):  # Ensure update_ops are executed before the minimize operation
            optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08,
            )
            training_op = optimizer.minimize(loss, global_step=global_step)
        return training_op

    def _define_neural_network_optimisation_node(
        self,
        Y: tf.Tensor,
        outputs: tf.Tensor,
    ) -> tf.Tensor:
        """Sets up the neural network optimization node."""
        with tf.device("/cpu:0"), tf.compat.v1.variable_scope("learning_rate"):
            learning_rate, global_step = self._setup_learning_rate()

        with tf.compat.v1.name_scope("loss_function"):
            loss = self._define_loss(Y, outputs)

        with tf.compat.v1.name_scope("optimizer"):
            training_op = self._define_optimizer(learning_rate, loss, global_step)

        return training_op
