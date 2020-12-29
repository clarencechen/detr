# Copyright (c) Google, Inc. and its affiliates. All Rights Reserved
"""
tf.random RNG that works when run inside a tf.distribute.Strategy.

Mostly copy-paste from Keras Preprocessing Layers source code.
"""
import tensorflow as tf

class _RandomGenerator(tf.random.Generator):
    """A subclass that allows creation inside distribution strategies.
    This is a temporary solution to allow creating tf.random.Generator inside
    distribution strategies. It will be removed when proper API is in place.
    All replicas will have the same RNG state and generate the same random
    numbers.
    """

    # TODO(b/157995497): Temporarily use primary variable handle inside cross
    # replica context.
    @property
    def state(self):
        """The internal state of the RNG."""
        state_var = self._state_var
        try:
            _ = getattr(state_var, 'handle')
            return state_var
        except ValueError:
            return state_var.values[0]

    def _create_variable(self, *args, **kwargs):
        return tf.Variable(*args, **kwargs)


def make_generator(seed=None):
    if seed:
        return _RandomGenerator.from_seed(seed)
    else:
        return _RandomGenerator.from_non_deterministic_state()