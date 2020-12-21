# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import tensorflow as tf
from tensorflow.keras import layers, applications
from tensorflow.keras import Model

from .position_encoding import build_position_encoding


def _backbone(self, in_tensor: tf.Tensor, name: str, train_backbone: bool, return_interm_layers: bool):
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
    return [encoder_model.get_layer(name).output for name in out_layers]


def build_backbone(args, in_tensor: tf.Tensor):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone_outputs = _backbone(in_tensor, args.backbone, train_backbone, return_interm_layers)
    pos_outputs = [tf.cast(position_embedding(t), t.dtype) for t in backbone_outputs]
    model = Model(inputs=in_tensor, outputs=(backbone_outputs, pos_outputs))
    return model
