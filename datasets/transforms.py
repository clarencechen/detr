# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.layers.experimental.preprocessing as pp_layers

from abc import abstractmethod
from tensorflow.keras.backend import learning_phase
from util.misc import shallow_update_dict
from util.box_ops import box_area_xyxy
from util.rng import RandomGenerator


def crop(img, tgt, region):
    updates = {}
    i, j, h, w = region
    new_img = tf.image.crop_to_bounding_box(img, i, j, h, w)

    if "boxes" in tgt:
        max_size = tf.cast(tf.tile(tf.shape(img)[-3:-1], [2]), tf.float32)
        new_boxes = tgt["boxes"] * max_size - tf.cast(tf.stack([i, j, i, j]), tf.float32)
        new_boxes = new_boxes / tf.cast(tf.stack([h, w, h, w]), tf.float32)
        updates["boxes"] = tf.clip_by_value(new_boxes, 0, 1)

    if "masks" in tgt:
        new_masks = tf.image.crop_to_bounding_box(tgt['masks'], i, j, h, w)
        updates['masks'] = new_masks

    if "area" in tgt and ("boxes" in tgt or "masks" in tgt):
        # favor masks selection when recalculating area
        # this is compatible with semantics in original coco dataset
        # this is also more accurate, especially for panoptic segmentation
        if "masks" in tgt:
            updates["area"] = tf.reduce_sum(tf.cast(tgt['masks'], tf.int32), [-3, -2])
        else:
            updates["area"] = tf.cast(box_area_xyxy(new_boxes), tf.int32)

    return new_img, shallow_update_dict(tgt, updates)


def resize(img, tgt, min_size, max_size=None):
    updates = {}
    old_shape = tf.shape(img)[-3:-1]
    if not max_size:
        short, long = tf.reduce_min(old_shape), tf.reduce_max(old_shape)
        max_size = tf.cast((long * min_size) / short, tf.int32) + 1

    new_shape = (max_size, min_size) if old_shape[0] > old_shape[1] else (min_size, max_size)
    new_img = tf.cast(tf.image.resize(img, new_shape, method='bilinear', preserve_aspect_ratio=True), tf.uint8)

    if "masks" in tgt:
        new_masks = tf.image.resize(tgt['masks'], new_shape, method='nearest', preserve_aspect_ratio=True)
        updates['masks'] = new_masks

    if "area" in tgt:
        new_area = (tgt["area"] * tf.reduce_prod(new_shape)) / tf.reduce_prod(old_shape)
        updates["area"] = tf.cast(new_area, tf.int32)

    return new_img, shallow_update_dict(tgt, updates)


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


class RandomAugment(pp_layers.PreprocessingLayer):
    def __init__(self, *args, seed=None, **kwargs):
        super(RandomAugment, self).__init__(*args, **kwargs)
        if seed is None:
            self.rng = RandomGenerator.from_non_deterministic_state()
        else:
            self.rng = RandomGenerator.from_seed(seed)

    @abstractmethod
    def call(self, img, tgt, training):
        raise NotImplementedError


class RandomCropExtd(RandomAugment):
    def __init__(self, min_size, max_size, seed=None, **kwargs):
        super(RandomCropExtd, self).__init__(seed=seed, **kwargs)
        self.min_size = min_size
        self.max_size = max_size


    def call(self, img, tgt, training):
        if training:
            img_h, img_w = tf.shape(img)[-3], tf.shape(img)[-2]
            h = self.rng.uniform([1], self.min_size, tf.minimum(img_h, self.max_size), dtype=tf.int32)[0]
            w = self.rng.uniform([1], self.min_size, tf.minimum(img_w, self.max_size), dtype=tf.int32)[0]
            i = self.rng.uniform([1], 0, img_h - h, dtype=tf.int32)[0]
            j = self.rng.uniform([1], 0, img_w - w, dtype=tf.int32)[0]
            return crop(img, tgt, (i, j, h, w))
        return img, tgt


class RandomFlipExtd(RandomAugment):
    def __init__(self, horizontal=True, vertical=False, seed=None, **kwargs):
        super(RandomFlipExtd, self).__init__(seed=seed, **kwargs)
        self.horizontal = horizontal
        self.vertical = vertical


    def call(self, img, tgt, training):
        if training:
            seeds = self.rng.make_seeds(2)
            return flip(img, tgt, seeds, self.horizontal, self.vertical)
        return img, tgt


class RandomResizeExtd(RandomAugment):
    def __init__(self, sizes, seed=None, max_size=None, **kwargs):
        assert isinstance(sizes, (list, tuple))
        super(RandomResizeExtd, self).__init__(seed=seed, **kwargs)
        self.sizes = sizes
        self.max_idx = len(sizes)
        self.max_size = max_size


    def call(self, img, tgt, training):
        if training:
            idx = self.rng.uniform([1], 0, self.max_idx, dtype=tf.int32)
            size = tf.gather(self.sizes, idx, axis=0)[0]
            return resize(img, tgt, size, self.max_size)
        return img, tgt


class RandomPartialCropResize(RandomCropExtd):
    def __init__(self, scales, min_size, max_size, p=0.5, seed=None, **kwargs):
        assert isinstance(scales, (list, tuple))
        super(RandomPartialCropResize, self).__init__(min_size, max_size, seed=seed, **kwargs)
        self.p = p
        self.scales = scales
        self.max_idx = len(scales)


    def run_transform(self, img, tgt, training):
        idx = self.rng.uniform([1], 0, self.max_idx, dtype=tf.int32)
        scale = tf.gather(self.scales, idx, axis=0)[0]
        crop_img, crop_tgt = super().call(img, tgt, training)
        return resize(crop_img, crop_tgt, scale, None)


    def call(self, img, tgt, training):
        if training:
            select = self.rng.uniform([1], 0, 1, dtype=tf.float32)
            return tf.cond(select < self.p, lambda: self.run_transform(img, tgt, training), lambda: (img, tgt))
        return img, tgt
