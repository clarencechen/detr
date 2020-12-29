# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .coco import build as build_coco


def build_dataset(image_set, args, seed):
    if args.dataset_file == 'coco':
        if args.masks:
            raise ValueError('coco 2014 does not support panoptic segmentation')
        return build_coco('coco_2014', image_set, args, seed)
    if args.dataset_file == 'coco_panoptic':
        return build_coco('coco_2017', image_set, args, seed)
    raise ValueError(f'dataset {args.dataset_file} not supported')
