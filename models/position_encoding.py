# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.utils import get_custom_objects


class PositionEmbeddingSine(layers.Layer):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def get_config(self):
        child_config = {
            'num_pos_feats': self.num_pos_feats,
            'temperature': self.temperature,
            'normalize': self.normalize,
            'scale': self.scale
        }
        return super().get_config() + child_config

    def build(self, input_shape):
        self.height, self.width = input_shape[-2:] if K.image_data_format() == 'channels_first' else input_shape[-3:-1]

    def call(self, inputs):
        y_range = tf.range(height, dtype=tf.float32)
        x_range = tf.range(width, dtype=tf.float32)
        y_embed, x_embed = tf.expand_dims(tf.meshgrid(y_range, x_range), 0)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = tf.range(self.num_pos_feats, dtype=tf.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = tf.expand_dims(x_embed, -1) / dim_t
        pos_y = tf.expand_dims(y_embed, -1) / dim_t
        pos_x = tf.reshape(tf.stack([tf.sin(pos_x[:, :, :, 0::2]), tf.cos(pos_x[:, :, :, 1::2])], 4), pos_x.shape[:3] + [-1])
        pos_y = tf.reshape(tf.stack([tf.sin(pos_y[:, :, :, 0::2]), tf.cos(pos_y[:, :, :, 1::2])], 4), pos_y.shape[:3] + [-1])
        pos = tf.concatenate([pos_y, pos_x], 3)
        if K.image_data_format() == 'channels_first':
            pos = tf.transpose(pos, [0, 3, 1, 2])
        return pos


class PositionEmbeddingLearned(layers.Layer):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def get_config(self):
        child_config = {'num_pos_feats': self.num_pos_feats}
        return super().get_config() + child_config

    def build(self, input_shape):
        self.height, self.width = input_shape[-2:] if K.image_data_format() == 'channels_first' else input_shape[-3:-1]
        super(PositionEmbeddingLearned, self).build(input_shape)
        self.row_embed = self.add_weight(
            name='row_embed', shape=(self.height, self.num_pos_feats),
            initializer='glorot_uniform', trainable=True)
        self.col_embed = self.add_weight(
            name='col_embed', shape=(self.width, self.num_pos_feats),
            initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        i, j = tf.range(self.width), tf.range(self.height)
        pos = tf.concatenate([
            tf.stack([self.col_embed] * self.height, 0),
            tf.stack([self.row_embed] * self.width, 1),
        ], dim=-1)
        if K.image_data_format() == 'channels_first':
            pos = tf.permute(pos, [2, 0, 1])
        pos = tf.expand_dims(pos, 0)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding

get_custom_objects().update({
    'PositionEmbeddingSine': PositionEmbeddingSine,
    'PositionEmbeddingLearned': PositionEmbeddingLearned,
})
