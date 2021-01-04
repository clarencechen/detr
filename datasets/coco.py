# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.layers.experimental.preprocessing as pp_layers

from tensorflow.data.experimental import AUTOTUNE
from util.box_ops import box_xyxy_to_cxcywh, box_swap_xy

import datasets.transforms as T


class CocoDataset:
    # Implementation of Dataset Pipeline
    def __init__(self, ds_name, image_set, source, seed=None, return_masks=True, max_size=1333):
        self.load_string = ds_name + ('_panoptic' if return_masks else '')
        self.image_set = image_set
        self.source = source
        self.seed = seed
        self.target_key = 'panoptic_objects' if return_masks else 'objects'
        self.transforms = self.make_coco_transforms(image_set)
        self.return_masks = return_masks
        self.max_size = max_size


    def make_coco_transforms(self, image_set):
        normalize = pp_layers.Normalization(dtype=tf.float32, mean=[0.485, 0.456, 0.406], variance=[0.229, 0.224, 0.225])
        rescale = pp_layers.Rescaling(scale=1./255)
        resize_800 = tf.keras.layers.Lambda(lambda image, target: T.resize(image, target, 800, max_size=self.max_size))

        random_flip = T.RandomFlipExtd(seed=self.seed)
        crop = T.RandomCropExtd(min_size=384, max_size=600, seed=self.seed)
        resize_crop = T.RandomResizeExtd([400, 500, 600], seed=self.seed)
        crop_cond = T.RandomSelect(tf.keras.Sequential(resize_crop, crop), seed=self.seed)
        resize_scales = T.RandomResizeExtd([480 + 32 * i for i in range(11)], seed=self.seed)

        def train_transform(image, targets):
            image, targets = crop_cond(random_flip(image, targets))
            image, targets = resize_scales(image, targets)
            return normalize(rescale(image)), targets

        def val_transform(image, targets):
            image, targets = resize_800(image, targets)
            return normalize(rescale(image)), targets

        if image_set == 'train':
            return train_transform
        elif image_set == 'validation':
            return val_transform
        raise ValueError(f'unknown {image_set}')


    def format_box_masks(self, img, tgt):
        masks_bnhw = tf.transpose(tgt['masks'], [0, 3, 1, 2])
        bbox_cycxhw = box_xyxy_to_cxcywh(tgt['boxes'])
        tgt['boxes'] = box_swap_xy(bbox_cycxhw)
        tgt['masks'] = masks_bnhw
        img_pad_mask = tf.ones(tf.shape(img)[-3:-1], dtype=tf.int32)
        return img, img_pad_mask, tgt


    def decode_example(self, ex):
        img = tf.io.decode_img(ex['image'], channels=3)
        targets = {
            'image_id': ex['image/id'],
            'area': ex[self.target_key]['area'],
            'boxes': ex[self.target_key]['bbox'],
            'labels': ex[self.target_key]['label']
            'orig_size': tf.shape(img)[-3:-1]
        }
        if self.return_masks:
            seg_map_raw = tf.cast(tf.io.decode_img(ex['panoptic_image'], channels=3), tf.int32)
            seg_map_id = seg_map_raw[..., 0] + 256 * seg_map_raw[..., 1] + 65536 * seg_map_raw[..., 2]
            seg_map_id = tf.expand_dims(seg_map_id, [-1])
            seg_id_list = tf.expand_dims(tf.expand_dims(ex[self.target_key]['id'], -2), -3)
            targets['masks'] = tf.cast(seg_id_map == seg_id_list, tf.int32)
        if targets['boxes'].dtype == tf.int32:
            h, w = targets['orig_size'][0], targets['orig_size'][1]
            box_offset = tf.stack([0, 0, targets['boxes'][..., 0], targets['boxes'][..., 1]], axis=-1)
            adj_boxes = targets['boxes'] + box_offset) / tf.constant([w, h, w, h], dtype=tf.float32)
            targets['boxes'] = box_swap_xy(adj_boxes)
        return img, targets


    def deserialize_example(self, serialized_ex):
        serialized_tensor = tf.io.parse_tensor(serialized_ex, out_type=tf.string)
        serialized_seq = tf.unstack(serialized_tensor, num=8, axis=0)
        obj_array = tf.nest.pack_sequence_as({
            'image':tf.TensorSpec(shape=(), dtype=tf.string),
            'panoptic_image':tf.TensorSpec(shape=(), dtype=tf.string),
            'image/id': tf.TensorSpec(shape=(), dtype=tf.string),
            'panoptic_objects': {
                'area': tf.TensorSpec(shape=(), dtype=tf.string),
                'bbox': tf.TensorSpec(shape=(), dtype=tf.string),
                'id': tf.TensorSpec(shape=(), dtype=tf.string),
                'is_crowd': tf.TensorSpec(shape=(), dtype=tf.string),
                'label': tf.TensorSpec(shape=(), dtype=tf.string),
            }
        }, serialized_seq)
        obj_array['image/id'] = tf.io.parse_tensor(obj_array['image/id'], out_type=tf.int32)
        for k in obj_array['panoptic_objects'].keys():
            obj_array['panoptic_objects'][k] = tf.io.parse_tensor(
                obj_array['panoptic_objects'][k], out_type=tf.bool if k == 'is_crowd' else tf.int32
            )
        return obj_array


    def dataset(self, batch_size, num_workers):
        if self.source == 'tfrecord':
            ds = tf.data.TFRecordDataset([f'gs://coco-dataset-tfrecord/{self.load_string}_{self.image_set}_part_{i}.zip' for i in range(self.num_shards)],
                                         compression_type='GZIP', num_parallel_reads=AUTOTUNE)
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
            ds = ds.shuffle(seed=self.seed, buffer_size=AUTOTUNE)
        if self.source == 'tfrecord':
            ds = ds.map(self.deserialize_example, num_parallel_calls=num_workers)
        ds = ds.map(self.decode_example, num_parallel_calls=num_workers)
        if self.transforms is not None:
            ds = ds.map(self.transforms, num_parallel_calls=num_workers)
        ds = ds.map(self.format_box_masks, num_parallel_calls=num_workers)
        ds = ds.padded_batch(batch_size, padded_shapes=(
                             [self.max_size, self.max_size, 3], [self.max_size, self.max_size, 3], [None]
                             ), drop_remainder=True)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds


def build(ds_name, image_set, args, seed):
    return CocoDataset(ds_name, image_set, args.source, seed=seed, return_masks=args.masks, max_size=args.max_size)
