# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Tuple, Iterable

import tensorflow as tf

import util.misc as utils
from utis.metric_logger import MetricLogger, SmoothedValue
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: Tuple[tf.keras.Model], criterion: tf.keras.Model,
                    data_iter: Iterable, optimizer: Tuple[tf.train.Optimizer],
                    strategy: tf.distribute.Strategy, epoch: int, max_norm: float = 0):
    backbone, detector = model
    b_optimizer, d_optimizer = optimizer
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    train_summary_writer = tf.summary.create_file_writer()
    print_freq = 10

    @tf.function
    def train_step(inputs, masks, targets, weight_dict={}):
        with tf.GradientTape() as tape:
            loss_dict = criterion(detector(backbone(inputs), masks), targets)
            losses = tf.reduce_sum([loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict])

        b_vars, d_vars = backbone.trainable_variables detector.trainable_variables
        b_grad, d_grad = tape.gradient(loss, b_vars), tape.gradient(loss, d_vars)
        b_grad = [tf.clip_by_norm(grad, max_norm) for grad in b_grad]
        d_grad = [tf.clip_by_norm(grad, max_norm) for grad in d_grad]
        d_optimizer.apply_gradients(zip(d_grad, d_vars))
        b_optimizer.apply_gradients(zip(b_grad, b_vars))
        return loss_dict

    for data in metric_logger.log_every(data_iter, print_freq, header=f'Epoch: [{epoch}]'):
        loss_dict = strategy.run(train_step, args=data, kwargs={'weight_dict': criterion.weight_dict})

        # reduce losses over all GPUs for logging purposes
        ldict_reduced, ldict_reduced_raw, ldict_reduced_scld = utils.reduce_dict(strategy, loss_dict, criterion.weight_dict)
        loss_value = tf.reduce_sum(ldict_reduced_scld.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            for k, v in ldict_reduced.items():
                tf.print(k, ": ", v)
            sys.exit(1)

        step = d_optimizer.iterations
        if step % print_freq == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar('train/lr', d_optimizer.lr(step), step=step)
                for (k, v), (k_u, v_u) in zip(ldict_reduced_scld.items(), ldict_reduced_raw.items()):
                    tf.summary.scalar(f'train/losses/scaled/{k}', v, step=step)
                    tf.summary.scalar(f'train/losses/unscaled/{k_u}', v_u, step=step)
                tf.summary.scalar(f'train/metrics/class_error', ldict_reduced['class_error'], step=step)

        metric_logger.update(class_error=ldict_reduced['class_error'])
        metric_logger.update(lr=d_optimizer.lr(step))

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model, criterion, postprocessors, data_iter, strategy, output_dir):
    backbone, detector = model
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    train_summary_writer = tf.summary.create_file_writer()
    print_freq = 10
    coco_evaluator = panoptic_evaluator = None

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())

    for data in metric_logger.log_every(data_iter, print_freq, header='Test:'):
        loss_dict = strategy.run((lambda inputs, masks, targets: criterion(detector(backbone(inputs), masks), targets)),
                                 args=data)

        # reduce losses over all GPUs for logging purposes
        ldict_reduced, ldict_reduced_raw, ldict_reduced_scld = utils.reduce_dict(strategy, loss_dict, criterion.weight_dict)
        loss_value = tf.reduce_sum(ldict_reduced_scld.values())

        step = d_optimizer.iterations
        if step % print_freq == 0:
            with train_summary_writer.as_default():
                for (k, v), (k_u, v_u) in zip(ldict_reduced_scld.items(), ldict_reduced_raw.items()):
                    tf.summary.scalar(f'test/losses/scaled/{k}', v, step=step)
                    tf.summary.scalar(f'test/losses/unscaled/{k_u}', v_u, step=step)
                tf.summary.scalar(f'test/metrics/class_error', ldict_reduced['class_error'], step=step)

        metric_logger.update(class_error=ldict_reduced['class_error'])

        orig_target_sizes = tf.stack([t["orig_size"] for t in targets], axis=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = tf.stack([t["size"] for t in targets], axis=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].as_string(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = tf.strings.as_string(target["image_id"])
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
