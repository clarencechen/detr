# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
from typing import List

import tensorflow as tf
from tensorflow.keras import layers, losses, Sequential, Model

from util import box_ops
from util.misc import accuracy

from .backbone import build_backbone
from .matcher import build_matcher
from .postprocess import (PostProcess, PostProcessPanoptic)
from .segmentation import (MaskHeadSmallConv, dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class DETR():
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, batch_size, num_classes, num_queries, training, aux_loss=False, panoptic_seg_head=False, freeze_detr=False):
        """ Initializes the model.
        Parameters:
            transformer: keras functional layer graph of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            panoptic_seg_head: True if the panoptic segmentation mask head (attention map and FPN) is to be used.
        """
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = layers.Dense(num_classes + 1)
        self.bbox_embed = Sequential([layers.Dense(hidden_dim, activation='relu')] * 2 + [layers.Dense(4, activation='sigmoid')])
        self.query_embed = layers.Embedding(num_queries, hidden_dim)(tf.stack([tf.range(num_queries, dtype=tf.int32)] * batch_size, 0))
        self.input_proj = layers.Conv2D(hidden_dim, kernel_size=1)
        self.input_reshape, self.pos_reshape = layers.Reshape((-1, hidden_dim)), layers.Reshape((-1, hidden_dim))
        self.mask_reshape = layers.Reshape((-1,))
        self.aux_loss = aux_loss
        self.panoptic_seg_head = panoptic_seg_head
        self.is_training = training

        if panoptic_seg_head:
            self.bbox_attention = layers.MultiHeadAttention(n_heads, hidden_dim // nheads, dropout=0)
            self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)

    def build(self, features, pos, masks):
        """ The output Keras model expects three inputs, which consists of:
               - features: list of batched features from backbone, each of shape [batch_size x F_H x F_W x num_channels]
               - pos: list of position embeddings, each of shape [1 x H x W x num_pos_feats]
               - masks: batched binary masks, of shape [batch_size x F_H x F_W], containing 1 on non-padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        src_proj, flat_masks = self.input_reshape(self.input_proj(features[-1])), self.mask_reshape(masks)
        hs, memory = self.transformer(src_proj, flat_masks, self.query_embed, self.pos_reshape(pos[-1]), training=self.is_training)

        outputs_class, outputs_coord = [self.class_embed(h) for h in hs], [self.bbox_embed(h) for h in hs]

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.panoptic_seg_head:
            # FIXME h_boxes takes the last one computed, keep this in mind
            bbox_pad_mask = tf.expand_dims(masks, 1)
            _, bbox_mask = self.bbox_attention(query=hs[-1], key=memory, training=self.is_training,
                                               attention_mask=bbox_pad_mask, return_attention_scores=True)
            bbox_mask = layers.Permute((2, 3, 4, 1))(bbox_mask)

            seg_masks = self.mask_head(src_proj, bbox_mask, [features[2], features[1], features[0]])
            out["pred_masks"] = tf.reshape(seg_masks, [-1, self.num_queries, seg_masks.shape[-3], seg_masks.shape[-2]])

        return Model(inputs=(features, pos, masks), outputs=out)

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(layers.Layer):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, indices_shape, num_classes, matcher, weight_dict, eos_coef, losses, **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super(SetCriterion, self).__init__(**kwargs)
        self.indices_shape = indices_shape
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.loss_list = losses

        self.key_list = ['area']
        for loss in losses:
            if loss == 'cardinality':
                pass
            elif loss == 'class_error':
                if 'labels' not in self.key_list:
                    self.key_list += ['labels']
            else:
                self.key_list += [loss]

    @tf.function
    def loss_labels(self, outputs, targets, valid_idxs, num_boxes, **kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [batch_size, num_queries]
        """
        assert 'pred_logits' in outputs
        
        pred_logits = outputs["pred_logits"]
        target_classes = tf.where(targets['area'] > 0, targets['labels'], self.num_classes)

        loss_ce = losses.sparse_categorical_crossentropy(target_classes, pred_logits, from_logits=True)
        sample_weight = tf.where(target_classes == self.num_classes, self.eos_coef, 1)

        loss_dict = {'loss_ce': tf.nn.compute_average_loss(loss_ce, sample_weight=sample_weight)}
        return loss_dict

    @tf.function
    def loss_class_error(self, outputs, targets, valid_idxs, num_boxes, log=True, **kwargs):
        """TargetbBox classification error
        targets dicts must contain the key "labels" containing a tensor of dim [batch_size, num_queries]
        """
        assert 'pred_logits' in outputs
        if not log:
            return {}

        src_logits = tf.gather_nd(outputs['pred_logits'], valid_idxs)
        target_classes_o = tf.gather_nd(targets['labels'], valid_idxs)

        loss_dict = {'class_error': tf.nn.compute_average_loss(100 - accuracy(src_logits, target_classes_o)[0])}
        return loss_dict

    @tf.function
    def loss_cardinality(self, outputs, targets, valid_idxs, num_boxes, **kwargs):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        assert 'pred_logits' in outputs

        pred_logits = outputs['pred_logits']
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        pred_card = tf.math.count_nonzero((tf.argmax(pred_logits, axis=-1) != self.num_classes), axis=-1)
        target_card = tf.math.count_nonzero(targets['area'] > 0, axis=-1)

        loss_dict = {'cardinality_error': tf.nn.compute_average_loss(tf.abs(pred_card - target_card))}
        return loss_dict

    @tf.function
    def loss_boxes(self, outputs, targets, valid_idxs, num_boxes, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "bbox" containing a tensor of dim [batch_size, num_queries, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        src_boxes = tf.gather_nd(outputs['pred_boxes'], valid_idxs)
        target_boxes = tf.gather_nd(targets['boxes'], valid_idxs)

        loss_bbox = tf.reduce_mean(tf.abs(src_boxes -target_boxes), axis=-1)
        loss_giou = 1 - tf.linalg.diag_part(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        loss_dict = {
            'loss_bbox': tf.reduce_sum(loss_bbox) / num_boxes,
            'loss_giou': tf.reduce_sum(loss_giou) / num_boxes
        }
        return loss_dict

    @tf.function
    def loss_masks(self, outputs, targets, valid_idxs, num_boxes, **kwargs):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [batch_size, num_queries, h, w]
        """
        assert "pred_masks" in outputs

        # TODO mask spatial padding regions for mask losses
        src_masks = tf.gather_nd(outputs['pred_masks'], valid_idxs)
        target_masks = tf.gather_nd(targets['masks'], valid_idxs)

        # upsample predictions to the target size
        src_masks = tf.image.resize(tf.expand_dims(src_masks, -1), size=target_masks.shape[1:3])[..., 0]
        src_masks, target_masks = self._flatten_mask(src_masks), self._flatten_mask(target_masks)

        loss_dict = {
            'loss_mask': tf.reduce_sum(sigmoid_focal_loss(src_masks, target_masks)) / num_boxes,
            'loss_dice': tf.reduce_sum(dice_loss(src_masks, target_masks)) / num_boxes
        }
        return loss_dict
    @staticmethod
    @tf.function
    def _flatten_mask(x: tf.Tensor):
        return tf.reshape(x, [tf.shape(x)[0], -1])

    @tf.function
    def _get_tgt_src_permutation(self, outputs: tf.Tensor, targets: tf.Tensor, indices: tf.Tensor):
        # retrieve permutation of each output and target tensor of shape [batch_size, num_queries, ...] in batch following indices
        indices = tf.ensure_shape(indices, self.indices_shape)
        matched_outputs = {k: tf.gather(outputs[k], indices[:, 0], axis=1, batch_dims=1) for k in outputs.keys()}
        matched_targets = {k: tf.gather(targets[k], indices[:, 1], axis=1, batch_dims=1) for k in self.key_list}
        return matched_outputs, matched_targets

    @tf.function
    def get_loss(self, loss, outputs, targets, valid_idxs, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'class_error': self.loss_class_error,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, valid_idxs, num_boxes, **kwargs)

    @tf.function
    def call(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: dict of tensors, such that targets[k].shape[0] == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        match_indices = self.matcher(outputs_without_aux, targets)

        # Gather the targets matched with the outputs of the last layer
        matched_outputs, matched_targets = self._get_tgt_src_permutation(outputs_without_aux, targets, match_indices)
        valid_indices = tf.where(matched_targets['area'] > 0)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = tf.cast(tf.shape(valid_indices)[0], tf.float32)
        ctx = tf.distribute.get_replica_context()
        if ctx is not None:
            num_boxes = tf.maximum(1.0, ctx.all_reduce("SUM", num_boxes))

        # Compute all the requested losses
        loss_dict = {}
        for loss in self.loss_list:
            loss_dict.update(self.get_loss(loss, matched_outputs, matched_targets, valid_indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(aux_outputs, targets)
                aux_outputs, aux_targets = self._get_tgt_src_permutation(aux_outputs, targets, aux_indices)
                aux_valid_indices = tf.where(aux_targets['area'] > 0)
                for loss in self.loss_list:
                    kwargs = {}
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if loss == 'class_error':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, aux_targets, aux_valid_indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    loss_dict.update(l_dict)

        return loss_dict


def build(args, strategy):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250

    weight_dict = {'loss_ce': 1.0, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    with strategy.scope():
        input_layer = layers.Input((args.max_size, args.max_size, 3))
        masks_input_layer = layers.Input((args.max_size, args.max_size))

        backbone, out_tensors = build_backbone(args, input_layer, masks_input_layer)
        out_feats, out_pos, out_masks = out_tensors
        detr_feats = [layers.Input(o.shape[1:]) for o in out_feats]
        detr_pos = [layers.Input(o.shape[1:]) for o in out_pos]
        detr_masks = layers.Input(out_masks.shape[1:])

        detector = DETR(backbone,
            build_transformer(args),
            batch_size=args.batch_size,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            panoptic_seg_head=args.masks,
            freeze_detr=(args.frozen_weights is not None),
            training=(not args.eval)
        ).build(detr_feats, detr_pos, detr_masks)
        matcher = build_matcher(args)

        losses = ['labels', 'boxes', 'cardinality']
        if args.masks:
            losses += ["masks"]
        criterion = SetCriterion(indices_shape=[args.batch_size, 2, args.num_queries],
                                 num_classes=num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses)

        postprocessors = {'detr': PostProcess()}
        if args.masks:
            postprocessors['iou_types'] = ('bbox', 'segm')
            if args.dataset_file == "coco_panoptic":
                is_thing_map = {i: i <= 90 for i in range(201)}
                postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)
        else:
            postprocessors['iou_types'] = tuple('bbox')

    return backbone, detector, criterion, postprocessors
