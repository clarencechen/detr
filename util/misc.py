# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
from typing import Optional, List

import tensorflow as tf


def reduce_dict(strategy: tf.distribute.Strategy, input_dict: dict, weight_dict: dict, average: bool = True):
    """
    Args:
        strategy: strategy object indicating the devices/threads to reduce over
        input_dict: all the values to be reduced
        weight_dict: scalar weights for each of the values to be reduces
        average: whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    reduce_op = tf.distribute.ReduceOp.MEAN if average else tf.distribute.ReduceOp.SUM
    # sort the keys so that they are consistent across processes
    reduced_dict = {}
    for k in sorted(input_dict.keys()):
        reduced_dict[k] = strategy.reduce(reduce_op, input_dict[k], axis=None)
    dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in reduced_dict.items()}
    dict_reduced_scaled = {k: v * weight_dict[k] for k, v in reduced_dict.items() if k in weight_dict}
    return reduced_dict, dict_reduced_unscaled, dict_reduced_scaled


def gather_dict(strategy: tf.distribute.Strategy, input_dict: dict, axis: int = 0):
    """
    Args:
        strategy: strategy object indicating the devices/threads to gather over
        input_dict: all the values to be gathered
    Gather the values in the dictionary from all processes across the specified
    axis so that the output dict has the results concatenated from all process.
    Returns a dict with the same fields as input_dict, after gathering.
    """
    # sort the keys so that they are consistent across processes
    gather_dict = {}
    for k in sorted(input_dict.keys()):
        gather_dict[k] = strategy.gather(input_dict[k], axis)
    return gather_dict


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def find_strategy_single_worker(args):
    if args.use_tpu:
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
        except ValueError:
            tpu = None
    else:
        tpu = None
    if tpu:
        # TPUStrategy for distributed training
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    elif args.num_gpus > 1:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
            gpu_name_list = [gpu.name for gpu in gpus]
            print(f"{len(args.num_gpus)} Physical GPUs requested, {len(gpus)} Physical GPUs available")
            strategy = tf.distribute.MirroredStrategy(devices=gpu_name_list[:min(args.num_gpus, len(gpus))])
        else:
            strategy = tf.distribute.get_strategy()
    return strategy


def get_rank():
    return 1


@tf.function
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if tf.size(target) == 0:
        return [tf.zeros([], dtype=tf.float64)]
    batch_size = target.shape[0]
    remaining_axes = list(range(2, tf.rank(output)))
    pred = tf.transpose(tf.sort(output, 1)[:, :max(topk)], [1, 0] + remaining_axes)
    correct = pred == tf.expand_dims(tf.reshape(target, (1, -1)), remaining_axes)

    res = []
    for k in topk:
        correct_k = tf.reduce_sum(tf.cast(tf.reshape(correct[:k], -1), tf.float64), 0)
        res.append(correct_k * (100.0 / batch_size))
    return res
