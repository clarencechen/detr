# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Dict

from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.applications import imagenet_utils
from util.box_ops import box_xyxy_to_cxcywh, box_swap_xy
from util.misc import shallow_update_dict

import datasets.transforms as T


class CocoDataset:
    # Implementation of Dataset Pipeline
    def __init__(self, ds_name, image_set, source, seed=None, return_masks=True, max_size=1333, num_queries=100):
        self.load_string = ds_name

        if image_set not in ('train', 'val', 'validation'):
            raise ValueError(f'unknown {image_set}')
        self.image_set = image_set

        self.source = source
        self.seed = seed
        self.target_key = 'panoptic_objects' if 'panoptic' in ds_name else 'objects'
        self.return_masks = return_masks
        self.max_size = max_size
        self.num_queries = num_queries

        self.random_flip = T.RandomFlipExtd(seed=seed)
        self.crop_cond = T.RandomPartialCropResize(scales=[400, 500, 600], min_size=384, max_size=600, seed=seed)
        self.resize_scales = T.RandomResizeExtd([480 + 32 * i for i in range(11)], seed=seed)

    def transforms(self, img, tgt):
        training = (self.image_set == 'train')

        img, tgt = self.random_flip(img, tgt, training=training)
        img, tgt = self.crop_cond(img, tgt, training=training)
        img, tgt = self.resize_scales(img, tgt, training=training)
        if not training:
            img, tgt = T.resize(img, tgt, 800, max_size=self.max_size)
        return imagenet_utils.preprocess_input(img, mode='caffe'), tgt


    def format_box_masks(self, img, tgt):
        updates = {}
        updates['size'] = tf.shape(img)[-3:-1]
        if 'masks' in tgt:
            masks_bnhw = tf.transpose(tgt['masks'], [0, 3, 1, 2])
            updates['masks'] = masks_bnhw
        if 'boxes' in tgt:
            boxes_cycxhw = box_xyxy_to_cxcywh(tgt['boxes'])
            updates['boxes'] = box_swap_xy(boxes_cycxhw)

        img_pad_mask = tf.ones(updates['size'], dtype=tf.int32)
        return img, img_pad_mask, shallow_update_dict(tgt, updates)


    def decode_example(self, ex):
        img = ex['image']
        targets = {
            'image_id': ex['image/id'],
            'area': ex[self.target_key]['area'],
            'boxes': ex[self.target_key]['bbox'],
            'labels': ex[self.target_key]['label'],
            'orig_size': tf.shape(img)[-3:-1]
        }
        if self.return_masks and 'panoptic_image' in ex:
            seg_map_raw = tf.cast(ex['panoptic_image'], tf.int32)
            seg_map_id = seg_map_raw[..., 0] + 256 * seg_map_raw[..., 1] + 65536 * seg_map_raw[..., 2]
            seg_map_id = tf.expand_dims(seg_map_id, [-1])
            seg_id_list = tf.expand_dims(tf.expand_dims(ex[self.target_key]['id'], -2), -3)
            targets['masks'] = tf.cast(seg_id_map == seg_id_list, tf.int32)
        if targets['boxes'].dtype == tf.int32:
            h, w = targets['orig_size'][0], targets['orig_size'][1]
            boxes_left_top = targets['boxes'][..., 0:2]
            box_offset = tf.concat([tf.zeros_like(boxes_left_top), boxes_left_top], -1)
            adj_boxes = tf.cast(targets['boxes'] + box_offset, tf.float32) / tf.cast(tf.stack([w, h, w, h]), tf.float32)
            targets['boxes'] = box_swap_xy(adj_boxes)
        return img, targets


    def deserialize_example(self, serialized_ex):
        serialized_tensor = tf.io.parse_tensor(serialized_ex, out_type=tf.string)
        serialized_seq = tf.unstack(serialized_tensor, num=8, axis=0)
        obj_array = tf.nest.pack_sequence_as({
            'image':tf.TensorSpec(shape=(), dtype=tf.string),
            'panoptic_image':tf.TensorSpec(shape=(), dtype=tf.string),
            'image/id': tf.TensorSpec(shape=(), dtype=tf.string),
            self.target_key: {
                'area': tf.TensorSpec(shape=(), dtype=tf.string),
                'bbox': tf.TensorSpec(shape=(), dtype=tf.string),
                'id': tf.TensorSpec(shape=(), dtype=tf.string),
                'is_crowd': tf.TensorSpec(shape=(), dtype=tf.string),
                'label': tf.TensorSpec(shape=(), dtype=tf.string),
            }
        }, serialized_seq)
        obj_array['image/id'] = tf.io.parse_tensor(obj_array['image/id'], out_type=tf.int32)
        for k in obj_array[self.target_key].keys():
            obj_array[self.target_key][k] = tf.io.parse_tensor(
                obj_array[self.target_key][k], out_type=tf.bool if k == 'is_crowd' else tf.int32
            )
        obj_array['image'] = tf.image.decode_jpeg(obj_array['image'], channels=3)
        if self.return_masks and 'panoptic_image' in obj_array:
            obj_array['panoptic_image'] = tf.image.decode_png(obj_array['panoptic_image'], channels=3)
        return obj_array


    def dataset(self, batch_size, num_workers):
        if self.source == 'tfrecord':
            num_shards = 16 if self.image_set == 'train' else 4
            ds = tf.data.TFRecordDataset([f'gs://coco-dataset-tfrecord/{self.load_string}_{self.image_set}_part_{i}.zip' for i in range(num_shards)],
                                         compression_type='GZIP', num_parallel_reads=num_shards)
        else:
            decoder_dict = {'image': tfds.decode.SkipDecoding()}
            if self.return_masks:
                decoder_dict['panoptic_image'] = tfds.decode.SkipDecoding()
            ds = tfds.load(self.load_string, split=self.image_set, batch_size=1,
                           shuffle_files=(self.image_set == 'train'),
                           decoders=decoder_dict, read_config=tfds.ReadConfig(
                            try_autocache=False, shuffle_seed=(self.seed + 1), skip_prefetch=True
                           ), try_gcs=True)
        if self.image_set == 'train':
            ds = ds.shuffle(seed=self.seed, buffer_size=0x4000)
        if self.source == 'tfrecord':
            ds = ds.map(self.deserialize_example, num_parallel_calls=num_workers)
        ds = ds.map(self.decode_example, num_parallel_calls=num_workers)
        ds = ds.map(self.transforms, num_parallel_calls=num_workers)
        ds = ds.map(self.format_box_masks, num_parallel_calls=num_workers)

        tgt_padded_shape_dict = {
            'image_id': [],
            'area': [self.num_queries],
            'boxes': [self.num_queries, 4],
            'labels': [self.num_queries],
            'orig_size': [2]
        }
        if self.return_masks:
            tgt_padded_shape_dict['masks'] = [self.num_queries, self.max_size, self.max_size]
        ds = ds.padded_batch(batch_size, padded_shapes=(
                             [self.max_size, self.max_size, 3],
                             [self.max_size, self.max_size],
                             tgt_padded_shape_dict), drop_remainder=True)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds


def build(ds_name, image_set, args, seed):
    return CocoDataset(ds_name, image_set, args.source, seed=seed, return_masks=args.masks, max_size=args.max_size, num_queries=args.num_queries)
