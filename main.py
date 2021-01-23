# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse, time, datetime, os, json

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='ResNet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--max_size', default=1280, type=int,
                        help="Maximum resolution to pad images to for training")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1.0, type=float)
    parser.add_argument('--dice_loss_coef', default=1.0, type=float)
    parser.add_argument('--bbox_loss_coef', default=5.0, type=float)
    parser.add_argument('--giou_loss_coef', default=2.0, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco_panoptic')
    parser.add_argument('--source', default='tfrecord')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--use_tpu', action='store_false',
                        help='use available tpu pods for training')
    parser.add_argument('--num_gpus', default=1, type=int,
                        help='number of available gpu devices to train on')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    strategy = utils.find_strategy_single_worker(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    backbone, detector, cost_calc, criterion, postprocessors = build_model(args, strategy)

    detector.summary()
    backbone.summary()

    # if args.frozen_weights is not None:
    #    backbone = backbone.load(args.frozen_weights, map_location='cpu')

    total_batch_size = args.batch_size * strategy.num_replicas_in_sync
    dataset_train = build_dataset(image_set='train', args=args, seed=seed)
    dataset_val = build_dataset(image_set='validation', args=args, seed=seed)

    data_loader_train = dataset_train.dataset(batch_size=total_batch_size, num_workers=args.num_workers)
    data_loader_val = dataset_val.dataset(batch_size=total_batch_size, num_workers=args.num_workers)
    data_iter_train = iter(strategy.experimental_distribute_dataset(data_loader_train))
    data_iter_val = iter(strategy.experimental_distribute_dataset(data_loader_val))

    NUM_TRAIN_EXS = 118287
    steps_per_drop = args.lr_drop * (NUM_TRAIN_EXS // total_batch_size)
    b_sched = tf.keras.optimizers.schedules.ExponentialDecay(args.lr_backbone, steps_per_drop, 0.1, staircase=True)
    d_sched = tf.keras.optimizers.schedules.ExponentialDecay(args.lr, steps_per_drop, 0.1, staircase=True)
    b_opt = tfa.optimizers.AdamW(learning_rate=b_sched, weight_decay=args.weight_decay)
    d_opt = tfa.optimizers.AdamW(learning_rate=d_sched, weight_decay=args.weight_decay)
    optimizer = (b_opt, d_opt)
    model = (backbone, detector)

    if args.resume:
        if args.resume.startswith('https'):
            raise NotImplementedError('Loading hub checkpoints is not supported yet.')
        else:
            checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
            manager = tf.train.CheckpointManager(checkpoint, args.resume, max_to_keep=1)
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            args.start_epoch = int(manager.latest_checkpoint.split('-')[-1])
            del manager, checkpoint
        else:
            raise FileNotFoundError(f'Could not find checkpoint at {args.resume}')

    if args.eval:
        test_stats, _ = evaluate(model, cost_calc, criterion, postprocessors,
                                 data_iter_val, strategy, args.output_dir)
        return

    print("Start training")
    start_time = time.time()
    if args.output_dir:
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint')
        save_checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        save_manager = tf.train.CheckpointManager(save_checkpoint, checkpoint_path, max_to_keep=1)

        tensorboard_path = os.path.join(args.output_dir, 'tensorboard')
        summary_writer = tf.summary.create_file_writer(tensorboard_path)
    else:
        summary_writer = None

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, cost_calc, criterion, data_iter_train, optimizer,
            strategy, epoch, summary_writer, args.clip_max_norm)
        if args.output_dir:
            save_manager.save(checkpoint_number=epoch + 1)
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                save_checkpoint.save(checkpoint_path + f'-{epoch:04}')

        test_stats, coco_evaluator = evaluate(
            model, cost_calc, criterion, postprocessors,
            data_iter_val, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch}

        if args.output_dir:
            log_path = os.path.join(args.output_di, 'log.txt')
            with tf.io.gfile.GFile(log_path, mode='a') as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                eval_path = os.path.join(args.output_di, 'eval')
                if not tf.io.gfile.exists(eval_path):
                    tf.io.gfile.mkdir(eval_path)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   eval_path / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir and not tf.io.gfile.exists(args.output_dir):
        tf.io.gfile.makedirs(args.output_dir)
    main(args)
