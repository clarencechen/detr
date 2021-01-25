# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR criterion class.
"""
import tensorflow as tf
from tensorflow.keras import layers, losses

from .segmentation import (dice_loss, sigmoid_focal_loss)

from util import box_ops
from util.misc import accuracy


class HungarianCost(layers.Layer):
    """This class computes hungarian cost matrices between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, batch_size: int, num_queries: int,
                 cost_class: float = 1, cost_bbox: float = 1,
                 cost_giou: float = 1, **kwargs):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super(HungarianCost, self).__init__(**kwargs)
        self.cost_shape = [batch_size, num_queries, num_queries]
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @tf.function
    def call(self, outputs, targets):
        costs = {}
        principal_outputs = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        costs['main_outputs'] = self.match_costs(principal_outputs, targets)
        if 'aux_outputs' in outputs:
            costs['aux_outputs'] = [self.match_costs(aux_out, targets) for aux_out in outputs['aux_outputs']]
        costs = tf.nest.map_structure(lambda c: tf.ensure_shape(c, self.cost_shape), costs)

        return costs

    @tf.function
    def match_costs(self, outputs, targets):
        """ Calculates cost matrices for the Hungarian Matching algorithm.

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a dict of targets containing key-value pairs:
                 "labels": Tensor of dim [batch_size, num_queries] (where num_queries is the padded length of 
                           objects in the target batch) containing the class labels
                 "boxes": Tensor of dim [batch_size, num_queries, 4] containing the target box coordinates
        Returns:
            A tensor of shape [batch_size, num_queries, num_queries] containing batched cost matrices between each
            ordered pair of candidates with indices (output_query, target_entry)
        """

        out_shape = tf.shape(outputs["pred_logits"])

        # We flatten to compute the cost matrices in a batch
        out_prob = tf.nn.softmax(outputs["pred_logits"], -1)  # [batch_size, num_queries, num_classes]
        out_boxes = tf.expand_dims(outputs["pred_boxes"], -2)  # [batch_size, num_queries, 1, 4]
        tgt_boxes = tf.expand_dims(targets["boxes"], -3)  # [batch_size, 1, num_targets, 4]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -tf.gather(out_prob, targets["labels"], axis=2, batch_dims=1) # [batch_size, num_queries, num_targets]

        # Compute the L1 cost between all pairs of boxes
        cost_bbox = tf.reduce_mean(tf.abs(out_boxes - tgt_boxes), -1)

        # Compute the giou cost betwen all pairs of boxes
        cost_giou = -box_ops.generalized_box_iou(
                     box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"]),
                     box_ops.box_cxcywh_to_xyxy(targets["boxes"])
                    )

        # Final cost matrix with shape [batch_size, num_queries, num_targets]
        cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        return tf.where(tf.expand_dims(targets["area"] > 0, -2), cost, 1e8)


class SetCriterion(layers.Layer):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, weight_dict, eos_coef, losses, **kwargs):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super(SetCriterion, self).__init__(**kwargs)
        self.num_classes = num_classes
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
    def call(self, outputs, indices, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             indices: dict of tensors of shape [batch_size, 2, num_queries]
             targets: dict of tensors, such that targets[k].shape[0] == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Gather the targets matched with the outputs of the last layer
        outputs_without_aux = {k: v for k, v in outputs.items() if k not in 'aux_outputs'}
        matched_outputs, matched_targets = self._get_tgt_src_permutation(outputs_without_aux, targets, indices['main_outputs'])
        valid_tgts = tf.where(matched_targets['area'] > 0)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_tgts = tf.cast(tf.shape(valid_tgts)[0], tf.float32)
        ctx = tf.distribute.get_replica_context()
        if ctx is not None:
            num_tgts = tf.maximum(1.0, ctx.all_reduce("SUM", num_tgts))

        # Compute all the requested losses
        loss_dict = {}
        for loss in self.loss_list:
            loss_dict.update(self.get_loss(loss, matched_outputs, matched_targets, valid_tgts, num_tgts))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_outputs, aux_targets = self._get_tgt_src_permutation(aux_outputs, targets, indices['aux_outputs'][i])
                aux_valid_tgts = tf.where(aux_targets['area'] > 0)
                for loss in self.loss_list:
                    kwargs = {}
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if loss == 'class_error':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, aux_targets, aux_valid_tgts, num_tgts, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    loss_dict.update(l_dict)

        return loss_dict
