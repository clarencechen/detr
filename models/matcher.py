# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import tensorflow as tf
from scipy.optimize import linear_sum_assignment

@tf.function
def hungarian_matcher(self, costs):
    """ Calculates matching indices using the Hungarian LSAP algorithm for the given collection of cost matrices.

    Params:
        cost: A nested structure containing tensors of shape [batch_size, num_queries, num_queries], which are 
              batched cost matrices between each ordered pair of candidates with indices (output_query, target_entry)
    Returns:
        A tensor of shape [batch_size, 2, num_queries] containing pairs (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
    """
    costs = tf.nest.map_structure(lambda cost: tf.ensure_shape(cost, self.input_shape), costs)
    with tf.device('/job:localhost/device:CPU:0'):
        indices = tf.nest.map_structure(lambda cost: tf.stack([
                     tf.stack(tf.numpy_function(linear_sum_assignment, [c], [tf.int64, tf.int64]), 0)
                    for c in tf.unstack(cost, axis=0)], 0), costs)
        indices = tf.nest.map_structure(lambda index: tf.ensure_shape(index, self.output_shape), indices)
    return indices
