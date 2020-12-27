# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import tensorflow as tf
from tensorflow.keras import layers, losses, Sequential, Model

from util import box_ops
from util.misc import accuracy

from .backbone import build_backbone
from .matcher import build_matcher
from .postprocess import (PostProcess, PostProcessPanoptic,
                            PostProcessSegm)
from .segmentation import (MHAttentionMap, MaskHeadSmallConv,
                            dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class DETR():
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, training, aux_loss=False, panoptic_seg_head=False, freeze_detr=False):
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
        self.query_embed = layers.Embedding(num_queries, hidden_dim)
        self.input_proj = layers.Conv2D(hidden_dim, kernel_size=1)
        self.aux_loss = aux_loss
        self.panoptic_seg_head = panoptic_seg_head
        self.is_training = training

        if panoptic_seg_head:
            self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
            self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)

    def build(self, features, pos, masks):
        """ The output Keras model expects two inputs, which consists of:
               - features: list of batched features from backbone, each of shape [batch_size x F_H x F_W x num_channels]
               - pos: position embeddings, of shape [1 x H x W x num_pos_feats]
               - masks: batched binary masks, of shape [batch_size x H x W], containing 1 on non-padded pixels

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
        src_proj = self.input_proj(features[-1])
        hs, memory = self.transformer(src_proj, masks, self.query_embed.weights[0], pos[-1], training=self.is_training)

        outputs_class, outputs_coord = list(zip([[self.class_embed(h), self.bbox_embed(h)] for h in hs]))

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.panoptic_seg_head:
            # FIXME h_boxes takes the last one computed, keep this in mind
            bbox_mask = self.bbox_attention(hs[-1], memory, training=self.is_training, mask=masks)

            seg_masks = self.mask_head(src_proj, bbox_mask, [features[2], features[1], features[0]])
            outputs_seg_masks = tf.reshape(seg_masks, [-1, self.num_queries, seg_masks.shape[-3], seg_masks.shape[-2]])
            out["pred_masks"] = outputs_seg_masks

        return Model(inputs=(features, pos, masks), outputs=out)

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(Model):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super(SetCriterion, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

    @tf.function
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        
        src_logits = outputs["pred_logits"]
        target_classes_o = targets["labels"]

        target_classes = tf.fill(src_logits.shape[:2], self.num_classes, dtype=tf.int64)
        target_classes = _fill_src_permutation(target_classes, indices, target_classes_o)

        loss_ce = losses.sparse_categorical_crossentropy(src_logits, target_classes)
        sample_weight = tf.where(target_classes == self.num_classes, self.eos_coef, 1)
        loss_ce = tf.nn.compute_average_loss(loss_ce, sample_weight=sample_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    @tf.function
    def loss_class_error(self, outputs, targets, indices, num_boxes, log=True):
        """TargetbBox classification error
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        if not log:
            return {}

        permuted_logits = _get_src_permutation(outputs['pred_logits'], indices)
        target_classes_o = targets['labels']

        losses = {'class_error': tf.nn.compute_average_loss(100 - accuracy(permuted_logits, target_classes_o)[0])}
        return losses

    @tf.function
    def loss_cardinality(self, outputs, targets, indices, num_boxes, lengths=None):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = tf.reduce_sum((tf.argmax(pred_logits, axis=-1) != self.num_classes), axis=-1)
        card_err = tf.abs(card_pred - lengths)
        losses = {'cardinality_error': tf.nn.compute_average_loss(card_err)}
        return losses

    @tf.function
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "bbox" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        src_boxes = _get_src_permutation(outputs['pred_boxes'], indices)
        target_boxes = targets['boxes']

        loss_bbox = tf.reduce_mean(tf.abs(src_boxes -target_boxes), axis=-1)

        losses = {}
        losses['loss_bbox'] = tf.nn.compute_average_loss(loss_bbox) / num_boxes

        loss_giou = 1 - tf.linalg.diag_part(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = tf.nn.compute_average_loss(loss_giou) / num_boxes
        return losses

    @tf.function
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_masks = _get_src_permutation(outputs['pred_masks'], indices)
        # TODO use valid to mask invalid areas due to padding in loss
        # target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = targets['masks']

        # upsample predictions to the target size
        src_masks = tf.image.resize(tf.expand_dims(src_masks, -1), size=target_masks.shape[1:3])[..., 0]
        src_masks, target_masks = _flatten_mask(src_masks), _flatten_mask(target_masks)

        losses = {
            "loss_mask": tf.nn.compute_average_loss(sigmoid_focal_loss(src_masks, target_masks, num_boxes)),
            "loss_dice": tf.nn.compute_average_loss(dice_loss(src_masks, target_masks, num_boxes)),
        }
        return losses

    @tf.function
    def _flatten_mask(x: tf.Tensor):
        return tf.reshape(x, [x.shape[0], -1])

    @tf.function
    def _get_tgt_permutation(targets: tf.Tensor, indices: List[tf.Tensor], lengths: tf.Tensor):
        # retrieve permutation of each target tensor of shape [num__valid_targets, ...] in batch following indices
        return tf.concat([tf.gather(tgt, idx, axis=0) for tgt, (_, idx) in zip(tf.split(targets, lengths, axis=0), indices)], axis=0)

    @tf.function
    def _get_src_permutation(outputs: tf.Tensor, indices: List[tf.Tensor]):
        # retrieve permutation of each output tensor of shape [batch_size, num_queries, ...] in batch following indices
        return tf.concat([tf.gather(outputs[i, ...], src, axis=0) for i, (src, _) in enumerate(indices)], axis=0)

    @tf.function
    def _fill_src_permutation(template: tf.Tensor, indices: List[tf.Tensor], outputs: tf.Tensor):
        batch_query_idx = tf.concat([tf.stack([i * tf.ones_like(src, dtype=tf.int64), src], axis=1) for i, (src, _) in enumerate(indices)], axis=0)
        return tf.tensor_scatter_nd_update(template, batch_query_idx, outputs)

    @tf.function
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'class_error': self.loss_class_error,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    @tf.function
    def call(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: dict of tensors, such that targets[k].shape[0] == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Gather valid targets from padded batch
        valid_entries = tf.where(targets['area'] > 0)
        valid_targets = {k: tf.gather_nd(v, valid_entries) for k, v in targets.items() if k != 'area'}
        tgt_lengths = tf.reduce_sum(tf.cast(targets['area'] > 0, tf.int32), axis=-1)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, valid_targets, tgt_lengths)

        # Gather the targets matched with the outputs of the last layer
        matched_targets = {k: _get_tgt_permutation(v, indices, tgt_lengths) for k, v in valid_targets.items() if k != 'area'}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = tf.cast(tf.reduce_sum(tgt_lengths), dtype=tf.float32)
        ctx = tf.distribute.get_replica_context()
        if ctx is not None:
            num_boxes = tf.maximum(1, ctx.all_reduce("MEAN", num_boxes, axis=None))

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, matched_targets, indices, tgt_lengths, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(aux_outputs, valid_targets, tgt_lengths)
                aux_targets = {k: _get_tgt_permutation(v, indices, tgt_lengths) for k, v in valid_targets.items() if k != 'area'}
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'class_error':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, aux_targets, aux_indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


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

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
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

        backbone = build_backbone(args, input_layer, masks_input_layer)
        detector = DETR(
            build_transformer(args),
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            panoptic_seg_head=args.masks,
            freeze_detr=(args.frozen_weights is not None),
            training=(not args.eval)
        ).build(backbone.outputs[0], backbone.outputs[1], masks_input_layer)
        matcher = build_matcher(args)

        losses = ['labels', 'boxes', 'cardinality']
        if args.masks:
            losses += ["masks"]
        criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses)

        postprocessors = {'detr': PostProcess()}
        if args.masks:
            postprocessors['iou_types'] = ('bbox', 'segm')
            args.dataset_file == "coco_panoptic":
                is_thing_map = {i: i <= 90 for i in range(201)}
                postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)
        else:
            postprocessors['iou_types'] = tuple('bbox')

    return backbone, detector, criterion, postprocessors
