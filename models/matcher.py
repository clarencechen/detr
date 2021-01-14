# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import tensorflow as tf
from scipy.optimize import linear_sum_assignment

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher:
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @tf.function
    def __call__(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a dict of targets containing key-value pairs:
                 "labels": Tensor of dim [batch_size, num_queries] (where num_queries is the padded length of 
                           objects in the target batch) containing the class labels
                 "boxes": Tensor of dim [batch_size, num_queries, 4] containing the target box coordinates

        Returns:
            A tensor of shape [batch_size, 2, num_queries] containing pairs (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
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
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(outputs["pred_boxes"]), box_cxcywh_to_xyxy(targets["boxes"]))

        # Final cost matrix with shape [batch_size, num_queries, num_targets]
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = tf.where(tf.expand_dims(targets["area"] > 0, -2), C, 1e8)

        with tf.device('/CPU:0'):
            indices = tf.stack([tf.stack(
                        tf.numpy_function(linear_sum_assignment, [c[0]], [tf.int32, tf.int32]),
                      0) for c in tf.split(C, tf.ones(tf.shape(C)[0], tf.int32), axis=0)], 0)

        return indices


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
