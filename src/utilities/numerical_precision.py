from __future__ import annotations

from enum import Enum

import numpy as np
import tensorflow.compat.v1 as tf


class NumericalPrecision(Enum):
    SIMPLE = {"tf_precision": tf.float32, "np_precision": np.float32}
    DOUBLE = {"tf_precision": tf.float64, "np_precision": np.float64}
