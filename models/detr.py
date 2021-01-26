# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model class and builder.
"""
from typing import List

import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

from .backbone import build_backbone
from .criterion import (HungarianCost, SetCriterion)
from .postprocess import (PostProcess, PostProcessPanoptic)
from .segmentation import (MaskHeadSmallConv)
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
        cost_calc = HungarianCost(args.batch_size, args.num_queries,
                                  cost_class=args.set_cost_class,
                                  cost_bbox=args.set_cost_bbox,
                                  cost_giou=args.set_cost_giou)
        losses = ['labels', 'boxes', 'cardinality']
        if args.masks:
            losses += ["masks"]
        criterion = SetCriterion(num_classes=num_classes, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses)

        postprocessors = {'detr': PostProcess()}
        if args.masks:
            postprocessors['iou_types'] = ('bbox', 'segm')
            if args.dataset_file == "coco_panoptic":
                is_thing_map = {i: i <= 90 for i in range(201)}
                postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)
        else:
            postprocessors['iou_types'] = tuple('bbox')

    return backbone, detector, cost_calc, criterion, postprocessors
