# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.layers.experimental.preprocessing as pp_layers

from tensorflow.keras.backend import learning_phase
from util.box_ops import box_area_xyxy
from util.rng import make_generator

@tf.function
def crop(image, target, region):
    i, j, h, w = region
    cropped_image = image[..., i:i + h, j:j + w, :]

    if "boxes" in target:
        max_size = tf.tile(tf.shape(image)[-3:-1], [2])
        cropped_boxes = target["boxes"] * max_size - tf.constant([i, j, i, j], dtype=tf.int32)
        cropped_boxes = cropped_boxes / tf.constant([h, w, h, w], dtype=tf.float32)
        target["boxes"] = tf.clip_by_value(cropped_boxes, min=0, max=1)

    if "masks" in target:
        target['masks'] = target['masks'][..., i:i + h, j:j + w, :]

    if "area" in target and ("boxes" in target or "masks" in target):
        # favor boxes selection when recalculating area
        # this is compatible with previous implementation
        if "boxes" in target:
            target["area"] = tf.cast(box_area_xyxy(cropped_boxes), tf.int32)
        else:
            target["area"] = tf.reduce_sum(tf.cast(target['masks'], tf.int32), [-3, -2])

    return cropped_image, target

@tf.function
def resize(image, target, min_size, max_size=None):
    image_shape = tf.shape(image)[-3:-1]
    ratio = min_size / tf.reduce_min(image_shape)
    if max_size and (ratio * tf.reduce_max(image_shape) > max_size):
        ratio = max_size / tf.reduce_max(image_shape)
    new_shape = tf.cast(ratio * image_shape, tf.int32)

    rescaled_image = tf.image.resize(image, new_shape, method='bilinear')

    if "masks" in target:
        rescaled_masks = tf.image.resize(target['masks'], new_shape, method='nearest')
        target['masks'] = rescaled_masks

    if "area" in target:
        rescaled_area = (target["area"] * tf.reduce_prod(new_shape)) / tf.reduce_prod(image_shape)
        target["area"] = rescaled_area

    return rescaled_image, target


class RandomCropExtd(pp_layers.PreprocessingLayer):
    def __init__(self, min_size, max_size, seed=None, **kwargs):
        super(RandomCropExtd, self).__init__(**kwargs)
        self.min_size = min_size
        self.max_size = max_size
        self.rng = make_generator(seed=seed)

    def call(self, img, tgt, training=True):
        if training is None:
            training = learning_phase()

        img_h, img_w = tf.shape(image)[-3], tf.shape(image)[-2]
        h = self.rng.uniform(self.min_size, tf.minimum(img_h, self.max_size), dtype=tf.int32)
        w = self.rng.uniform(self.min_size, tf.minimum(img_w, self.max_size), dtype=tf.int32)
        i = self.rng.uniform(0, img_h - h, dtype=tf.int32)
        j = self.rng.uniform(0, img_w - w, dtype=tf.int32)

        img, tgt = control_flow_util.smart_cond(training, crop(img, tgt, (i, j, h, w)), lambda: img, tgt)
        return img, tgt


class RandomFlipExtd(pp_layers.PreprocessingLayer):
    def __init__(self, horizontal=True, vertical=False, seed=None, **kwargs):
        super(FlipExtd, self).__init__(**kwargs)
        self.horizontal = horizontal
        self.vertical = vertical
        self.rng = make_generator(seed=seed)

    def call(self, img, tgt, training=True):
        if training is None:
            training = learning_phase()

        @tf.function
        def flip(img, tgt):
            seeds = self.rng.make_seeds(2)
            flipped_image = img
            if self.horizontal:
                flipped_image = tf.image.stateless_random_flip_left_right(flipped_image, seeds[:, 0])
            if self.vertical:
                flipped_image = tf.image.stateless_random_flip_up_down(flipped_image, seeds[:, 1])

            if "boxes" in target:
                flipped_boxes, bs = tgt['boxes'], tf.shape(tgt['boxes'])[0]
                if self.horizontal:
                    flips = tf.random.stateless_uniform([bs], seeds[:, 0], 0, 1, dtype=tf.float32)
                    flips = tf.round(flips)
                    left = (1 - flipped_boxes[..., 3]) * flips + (1 - flips) * flipped_boxes[..., 1]
                    right = (1 - flipped_boxes[..., 1]) * flips + (1 - flips) * flipped_boxes[..., 3]
                    flipped_boxes = tf.stack([flipped_boxes[..., 0], left, flipped_boxes[..., 2], right], axis=-1)
                if self.vertical:
                    flips = tf.random.stateless_uniform([bs], seeds[:, 1], 0, 1, dtype=tf.float32)
                    flips = tf.round(flips)
                    top = (1 - flipped_boxes[..., 2]) * flips + (1 - flips) * flipped_boxes[..., 0]
                    btm = (1 - flipped_boxes[..., 0]) * flips + (1 - flips) * flipped_boxes[..., 2]
                    flipped_boxes = tf.stack([top, flipped_boxes[..., 1], btm, flipped_boxes[..., 3]], axis=-1)
                tgt['boxes'] = flipped_boxes

            if "masks" in target:
                flipped_masks = tgt['masks']
                if self.horizontal:
                    flipped_masks = tf.image.stateless_random_flip_left_right(flipped_masks, seeds[:, 0])
                if self.vertical:
                    flipped_masks = tf.image.stateless_random_flip_up_down(flipped_masks, seeds[:, 1])
                tgt['masks'] = flipped_masks
    
            return img, tgt

        img, tgt = tf.cond(training, flip(img, tgt), lambda: img, tgt)
        return img, tgt


class RandomResizeExtd(pp_layers.PreprocessingLayer):
    def __init__(self, sizes, seed=None, max_size=None, **kwargs):
        assert isinstance(sizes, (list, tuple))
        super(RandomResizeExtd, self).__init__(**kwargs)
        self.sizes = sizes
        self.max_idx = len(sizes)
        self.max_size = max_size
        self.rng = make_generator(seed=seed)

    def __call__(self, img, tgt=None, training=True):
        if training is None:
            training = learning_phase()

        idx = self.rng.uniform([], 0, self.max_idx, dtype=tf.int32)
        size = self.sizes[idx]
        img, tgt = tf.cond(training, resize(img, tgt, size, self.max_size), lambda: img, tgt)
        return img, tgt
