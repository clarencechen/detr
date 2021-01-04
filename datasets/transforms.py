# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.layers.experimental.preprocessing as pp_layers

from tensorflow.keras.backend import learning_phase
from util.misc import shallow_update_dict
from util.box_ops import box_area_xyxy
from util.rng import make_generator

@tf.function
def crop(img, tgt, region):
    updates = {}
    i, j, h, w = region
    cropped_img = tf.image.crop_to_bounding_box(img, i, j, h, w)

    if "boxes" in tgt:
        max_size = tf.tile(tf.shape(img)[-3:-1], [2])
        cropped_boxes = tgt["boxes"] * max_size - tf.concat([i, j, i, j])
        cropped_boxes = cropped_boxes / tf.cast(tf.concat([h, w, h, w]), tf.float32)
        updates["boxes"] = tf.clip_by_value(cropped_boxes, min=0, max=1)

    if "masks" in tgt:
        cropped_masks = tf.image.crop_to_bounding_box(tgt['masks'], i, j, h, w)
        updates['masks'] = cropped_masks

    if "area" in tgt and ("boxes" in tgt or "masks" in tgt):
        # favor masks selection when recalculating area
        # this is compatible with semantics in original coco dataset
        # this is also more accurate, especially for panoptic segmentation
        if "masks" in tgt:
            updates["area"] = tf.reduce_sum(tf.cast(tgt['masks'], tf.int32), [-3, -2])
        else:
            updates["area"] = tf.cast(box_area_xyxy(cropped_boxes), tf.int32)

    return cropped_img, shallow_update_dict(tgt, updates)

@tf.function
def resize(img, tgt, min_size, max_size=None):
    updates = {}
    old_shape = tf.shape(img)[-3:-1]
    ratio = tf.cast(min_size, tf.float32) / tf.cast(tf.reduce_min(old_shape), tf.float32)
    if max_size and (ratio * tf.reduce_max(old_shape) > max_size):
        ratio = tf.cast(max_size, tf.float32) / tf.cast(tf.reduce_max(old_shape), tf.float32)
    new_shape = tf.cast(ratio * tf.cast(old_shape, tf.float32), tf.int32)

    new_img = tf.image.resize(img, new_shape, method='bilinear')

    if "masks" in tgt:
        rescaled_masks = tf.image.resize(tgt['masks'], new_shape, method='nearest')
        updates['masks'] = rescaled_masks

    if "area" in tgt:
        rescaled_area = (tgt["area"] * tf.reduce_prod(new_shape)) / tf.reduce_prod(old_shape)
        updates["area"] = rescaled_area

    return new_img, shallow_update_dict(tgt, updates)

@tf.function
def flip(img, tgt, seeds, h=False, v=False):
    updates = {}
    if h:
        flipped_img = tf.image.stateless_random_flip_left_right(img, seeds[:, 0])
    if v:
        flipped_img = tf.image.stateless_random_flip_up_down(img, seeds[:, 1])

    if "boxes" in tgt:
        flipped_boxes, bs = tgt['boxes'], tf.shape(tgt['boxes'])[0]
        if h:
            flips = tf.random.stateless_uniform([bs], seeds[:, 0], 0, 1, dtype=tf.float32)
            flips = tf.round(flips)
            left = (1 - flipped_boxes[..., 3]) * flips + (1 - flips) * flipped_boxes[..., 1]
            right = (1 - flipped_boxes[..., 1]) * flips + (1 - flips) * flipped_boxes[..., 3]
            flipped_boxes = tf.stack([flipped_boxes[..., 0], left, flipped_boxes[..., 2], right], axis=-1)
        if v:
            flips = tf.random.stateless_uniform([bs], seeds[:, 1], 0, 1, dtype=tf.float32)
            flips = tf.round(flips)
            top = (1 - flipped_boxes[..., 2]) * flips + (1 - flips) * flipped_boxes[..., 0]
            btm = (1 - flipped_boxes[..., 0]) * flips + (1 - flips) * flipped_boxes[..., 2]
            flipped_boxes = tf.stack([top, flipped_boxes[..., 1], btm, flipped_boxes[..., 3]], axis=-1)
        updates['boxes'] = flipped_boxes

    if "masks" in tgt:
        flipped_masks = tgt['masks']
        if h:
            flipped_masks = tf.image.stateless_random_flip_left_right(flipped_masks, seeds[:, 0])
        if v:
            flipped_masks = tf.image.stateless_random_flip_up_down(flipped_masks, seeds[:, 1])
        updates['masks'] = flipped_masks

    return flipped_img, shallow_update_dict(tgt, updates)


class RandomCropExtd(pp_layers.PreprocessingLayer):
    def __init__(self, min_size, max_size, seed=None, **kwargs):
        super(RandomCropExtd, self).__init__(**kwargs)
        self.min_size = min_size
        self.max_size = max_size
        self.rng = make_generator(seed=seed)

    def call(self, img, tgt, training):
        if training:
            img_h, img_w = tf.shape(img)[-3], tf.shape(img)[-2]
            h = self.rng.uniform([1], self.min_size, tf.minimum(img_h, self.max_size), dtype=tf.int32)
            w = self.rng.uniform([1], self.min_size, tf.minimum(img_w, self.max_size), dtype=tf.int32)
            i = self.rng.uniform([1], 0, img_h - h, dtype=tf.int32)
            j = self.rng.uniform([1], 0, img_w - w, dtype=tf.int32)
            return crop(img, tgt, (i, j, h, w))
        return img, tgt


class RandomFlipExtd(pp_layers.PreprocessingLayer):
    def __init__(self, horizontal=True, vertical=False, seed=None, **kwargs):
        super(RandomFlipExtd, self).__init__(**kwargs)
        self.horizontal = horizontal
        self.vertical = vertical
        self.rng = make_generator(seed=seed)

    def call(self, img, tgt, training):
        if training:
            seeds = self.rng.make_seeds(2)
            return flip(img, tgt, seeds, self.horizontal, self.vertical)
        return img, tgt


class RandomResizeExtd(pp_layers.PreprocessingLayer):
    def __init__(self, sizes, seed=None, max_size=None, **kwargs):
        assert isinstance(sizes, (list, tuple))
        super(RandomResizeExtd, self).__init__(**kwargs)
        self.sizes = sizes
        self.max_idx = len(sizes)
        self.max_size = max_size
        self.rng = make_generator(seed=seed)

    def call(self, img, tgt, training):
        if training:
            idx = self.rng.uniform([1], 0, self.max_idx, dtype=tf.int32)
            size = tf.gather(self.sizes, idx, axis=0)
            return resize(img, tgt, size, self.max_size)
        return img, tgt


class RandomSelect(pp_layers.PreprocessingLayer):
    def __init__(self, fn, p=0.5, seed=None, **kwargs):
        super(RandomSelect, self).__init__(**kwargs)
        self.p = p
        self.fn = fn
        self.rng = make_generator(seed=seed)

    def call(self, *args, training=None):
        if training:
            select = self.rng.uniform([1], 0, 1, dtype=tf.float32)
            return tf.cond(select < self.p, self.fn(*args, training=training), lambda: args)
        return args
