# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import tensorflow as tf

@tf.function
def box_swap_xy(x):
    return tf.stack([x[..., 1], x[..., 0],
                     x[..., 3], x[..., 2]], axis=-1)

@tf.function
def box_area_xyxy(x):
    return (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])

@tf.function
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return tf.stack(b, -1)

@tf.function
def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return tf.stack(b, -1)

# modified from torchvision to also return the union
@tf.function
def box_iou(x1, x2):
    a1, a2 = box_area_xyxy(x1), box_area_xyxy(x2)

    lt = tf.maximum(tf.expand_dims(x1[:, :2], -2), x2[:, :2])  # [N,M,2]
    rb = tf.minimum(tf.expand_dims(x1[:, 2:], -2), x2[:, 2:])  # [N,M,2]

    wh = tf.maximum(rb - lt, 0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = tf.expand_dims(a1, -1) + a2 - inter

    iou = inter / union
    return iou, union

@tf.function
def generalized_box_iou(x1, x2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(x1)
    and M = len(x2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    x1_ok = tf.debugging.assert_greater_equal(x1[:, 2:], x1[:, :2])
    x2_ok = tf.debugging.assert_greater_equal(x2[:, 2:], x2[:, :2])
    with tf.control_dependencies([x1_ok, x2_ok]):
        iou, union = box_iou(x1, x2)

        lt = tf.minimum(tf.expand_dims(x1[:, :2], -2), x2[:, :2])
        rb = tf.maximum(tf.expand_dims(x1[:, 2:], -2), x2[:, 2:])

        wh = tf.maximum(rb - lt, 0)  # [N,M,2]
        hull = wh[:, :, 0] * wh[:, :, 1]

        return iou - (hull - union) / hull

@tf.function
def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if tf.size(masks) == 0:
        return tf.zeros((0, 4), dtype=tf.float)

    h, w = masks.shape[-2:]

    y = tf.range(0, h, dtype=tf.float)
    x = tf.range(0, w, dtype=tf.float)
    y, x = tf.meshgrid(y, x)

    x_mask = masks * tf.expand_dims(x, 0)
    x_max = tf.reduce_max(x_mask, [-1, -2])
    x_min = tf.reduce_min(tf.where(masks, x_mask, 1e8), [-1, -2])

    y_mask = masks * tf.expand_dims(y, 0)
    y_max = tf.reduce_max(y_max, [-1, -2])
    y_min = tf.reduce_min(tf.where(masks, y_mask, 1e8), [-1, -2])

    return tf.stack([x_min, y_min, x_max, y_max], 1)
