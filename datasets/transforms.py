# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.layers.experimental.preprocessing as pp_layers

from tensorflow.keras.backend import learning_phase
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_area_xyxy
from util.rng import make_generator

@tf.function
def crop(image, target, region):
    i, j, h, w = region
    cropped_image = image[..., i:i + h, j:j + w, :]

    fields = ["labels"]

    if "boxes" in target:
        boxes_xyxy = box_cxcywh_to_xyxy(target["boxes"])
        max_size = tf.tile(tf.shape(image)[-3:-1:-1], 2)
        cropped_boxes = boxes_xyxy * max_size - tf.constant([j, i, j, i], dtype=tf.int32)
        cropped_boxes = cropped_boxes / tf.constant([w, h, w, h], dtype=tf.float32)
        target["boxes"] = box_xyxy_to_cxcywh(tf.clip_by_value(cropped_boxes, min=0, max=1))
        
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][..., i:i + h, j:j + w]
        fields.append("masks")

    # recalculate area for boxes or masks that been cropped
    if "boxes" in target or "masks" in target:
        # favor boxes selection when recalculating area
        # this is compatible with previous implementation
        if "boxes" in target:
            target["area"] = tf.cast(box_area_xyxy(cropped_boxes), tf.int32)
        else:
            target["area"] = tf.reduce_sum(tf.cast(target['masks'], tf.int32), [-2, -1])

    return cropped_image, target

@tf.function
def resize(image, target, min_size, max_size=None):

    @tf.function
    def get_size_with_aspect_ratio(image_size, min_size, max_size=None):
        h, w = float(image_size[0]), float(image_size[1])
        if max_size is not None:
            min_original_size = min(h, w)
            max_original_size = max(h, w)
            if max_original_size / min_original_size * size > max_size:
                size = float(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w
        return max(size, int(size * h / w)), max(int(size * w / h), size)

    h, w = tf.shape(image)[-3], tf.shape(image)[-2]
    nh, nw = get_size_with_aspect_ratio((h, w), size, max_size)
    rescaled_image = tf.image.resize(image, [nh, nw], method='bilinear')

    if target is None:
        return rescaled_image, None

    if "masks" in target:
        rescaled_masks = tf.transpose(target['masks'], [0, 2, 3, 1])
        rescaled_masks = tf.image.resize(rescaled_masks, [nh, nw], method='nearest')
        target['masks'] = tf.transpose(rescaled_masks, [0, 3, 1, 2])

    rescaled_area = (target["area"] * nh * nw)/(h * w)
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
            seeds, bs = self.rng.make_seeds(2), tf.shape(tgt['boxes'])[0]
            flipped_image, flipped_masks, flipped_boxes = img, tf.transpose(tgt['masks'], [0, 2, 3, 1]), tgt['boxes']

            if self.horizontal:
                flipped_image = tf.image.stateless_random_flip_left_right(flipped_image, seeds[:, 0])
                flipped_masks = tf.image.stateless_random_flip_left_right(flipped_masks, seeds[:, 0])
                flips = tf.random.stateless_uniform([bs, 1], seeds[:, 0], 0, 1, dtype=tf.float32)
                flips = tf.stack([tf.round(flips)] + 3 * [tf.zeros_like(flips)], axis=-1)
                flipped_boxes = flipped_boxes * (1 - 2 * flips) + flips

            if self.vertical:
                flipped_image = tf.image.stateless_random_flip_up_down(flipped_image, seeds[:, 1])
                flipped_masks = tf.image.stateless_random_flip_up_down(flipped_masks, seeds[:, 1])
                flips = tf.random.stateless_uniform([bs, 1], seeds[:, 1], 0, 1, dtype=tf.float32)
                flips = tf.stack([tf.zeros_like(flips), tf.round(flips)] + 2 * [tf.zeros_like(flips)], axis=-1)
                flipped_boxes = flipped_boxes * (1 - 2 * flips) + flips
                
            img, tgt['masks'], tgt['boxes'] = flipped_img, tf.transpose(flipped_masks, [0, 3, 1, 2]), flipped_boxes
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
