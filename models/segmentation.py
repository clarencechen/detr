# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow_addons import layers as tfa_layers


class MaskHeadSmallConv:
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim: int, fpn_dims: List[int], context_dim: int):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = layers.Conv2D(dim, 3, padding='same')
        self.gn1 = tfa_layers.GroupNorm(8)
        self.lay2 = layers.Conv2D(inter_dims[1], 3, padding='same')
        self.gn2 = tfa_layers.GroupNorm(8)
        self.lay3 = layers.Conv2D(inter_dims[2], 3, padding='same')
        self.gn3 = tfa_layers.GroupNorm(8)
        self.lay4 = layers.Conv2D(inter_dims[3], 3, padding='same')
        self.gn4 = tfa_layers.GroupNorm(8)
        self.lay5 = layers.Conv2D(inter_dims[4], 3, padding='same')
        self.gn5 = tfa_layers.GroupNorm(8)
        self.out_lay = layers.Conv2D(1, 3, padding='same')

        self.dim = dim

        self.adapter1 = layers.Conv2D(inter_dims[1], 1)
        self.adapter2 = layers.Conv2D(inter_dims[2], 1)
        self.adapter3 = layers.Conv2D(inter_dims[3], 1)

    def _expand(tensor, length: int):
        return tf.concat([tensor] * int(length), 0)

    def __call__(self, x: tf.Tensor, bbox_mask: tf.Tensor, fpns: List[tf.Tensor]):
        x = tf.concat([tf.stack([x] * bbox_mask.shape[1], 1), bbox_mask], -1)
        x = tf.reshape(x, [-1] + x.shape[-3:])

        x = layers.ReLU()(self.gn1(self.lay1(x)))
        x = layers.ReLU()(self.gn2(self.lay2(x)))

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = _expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        x = cur_fpn + layers.UpSampling2D(size=x.shape[1] // cur_fpn.shape[1])(x)
        x = layers.ReLU()(self.gn3(self.lay3(x)))

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = _expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        x = cur_fpn + layers.UpSampling2D(size=x.shape[1] // cur_fpn.shape[1])(x)
        x = layers.ReLU()(self.gn4(self.lay4(x)))

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = _expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        x = cur_fpn + layers.UpSampling2D(size=x.shape[1] // cur_fpn.shape[1])(x)
        x = layers.ReLU()(self.gn5(self.lay5(x)))

        x = self.out_lay(x)
        return x


@tf.function
def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = tf.sigmoid(inputs)
    numerator = 2 * tf.reduce_sum(inputs * targets, axis=-1)
    denominator = tf.reduce_sum(inputs, axis=-1) + tf.reduce_sum(targets, axis=-1)
    return 1 - (numerator + 1) / (denominator + 1)


@tf.function
def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = tf.sigmoid(inputs)
    ce_loss = losses.binary_crossentropy(inputs, targets, from_logits=True)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = tf.expand_dims(ce_loss, -1) * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return tf.reduce_mean(loss, axis=-1)
