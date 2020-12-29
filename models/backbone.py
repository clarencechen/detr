# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import tensorflow as tf
from tensorflow.keras import layers, applications
from tensorflow.keras import Model

from .position_encoding import build_position_encoding


def _backbone(in_tensor: tf.Tensor, name: str, train_backbone: bool, return_interm_layers: bool):
    """Returns outputs of ResNet backbone with frozen BatchNorm."""

    backbone = getattr(applications, name)(weights='imagenet', include_top=False, input_tensor=in_tensor)

    # Freeze BatchNorm Layers
    for l in backbone.layers:
        if 'bn' in l.name:
            l.trainable = False
        elif not train_backbone or 'conv3' not in name and 'conv4' not in name and 'conv5' not in name:
            l.trainable = False

    if return_interm_layers:
        out_layers = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out']
    else:
        out_layers = ['conv5_block3_out']
    return [backbone.get_layer(name).output for name in out_layers]


def build_backbone(args, in_tensor: tf.Tensor, in_masks: tf.Tensor):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    out_features = _backbone(in_tensor, args.backbone, train_backbone, args.masks)
    out_pos = [tf.cast(position_embedding(t), t.dtype) for t in out_features]
    out_masks = tf.expand_dims(in_masks, -1)
    out_masks = tf.image.resize(out_masks, out_features[-1].shape[-3:-1], method='nearest')
    out_masks = tf.squeeze(out_masks, -1)
    model = Model(inputs=(in_tensor, in_masks), outputs=(out_features, out_pos, out_masks))
    return model, (out_features, out_pos, out_masks)
