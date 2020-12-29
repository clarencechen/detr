import io
from collections import defaultdict
from typing import List, Optional

import tensorflow as tf
from PIL import Image

import util.box_ops as box_ops

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass


class PostProcess:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, outputs, processed_sizes, target_sizes=None):
        """ This module converts the model's output into the format expected by the coco api
        Parameters:
            outputs: raw outputs of the model
            processed_sizes: This is a tensor of dimension [batch_size x 2] of sizes of the images that were passed to
                             the model, ie the size after data augmentation but before batching.
            target_sizes: This is a tensor of dimension [batch_size x 2] corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert out_logits.shape[0] == target_sizes.shape[0]
        assert target_sizes.shape[0] == processed_sizes.shape[0]
        assert target_sizes.shape[1] == 2

        prob = tf.nn.softmax(out_logits, -1)
        scores, labels = tf.reduce_max(prob[..., :-1], -1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        scale_fct = tf.stack([target_sizes[:, 1], target_sizes[:, 0], target_sizes[:, 1], target_sizes[:, 0]], 1)
        boxes = boxes * tf.expand_dims(scale_fct, 1)

        results = [{'scores': scores[i], 'labels': labels[i], 'boxes': boxes[i]} for i in range(target_sizes.shape[0])]

        if 'pred_masks' in outputs:
            max_size = tf.reduce_max(processed_sizes, 0)
            outputs_masks = tf.transpose(outputs['pred_masks'], [0, 2, 3, 1])
            outputs_masks = tf.sigmoid(outputs_masks) > self.threshold

            for i in range(len(results)):
                img_h, img_w = processed_sizes[i][0], processed_sizes[i][1]
                out_masks_slice = outputs_masks[i, :img_h, :img_w, :]
                results[i]["masks"] = tf.transpose(tf.image.resize(out_masks_slice, size=target_sizes[i], method="nearest"), [2, 0, 1])

        return results


class PostProcessPanoptic:
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a tensor of dimension [batch_size x 2] of sizes of the images that were passed to
                             the model, ie the size after data augmentation but before batching.
            target_sizes: This is a tensor of dimension [batch_size x 2] corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert processed_sizes.shape[0] == target_sizes.shape[0]
        out_logits, raw_masks, raw_boxes = outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
        assert out_logits.shape[0] == raw_masks.shape[0] == target_sizes.shape[0]
        preds = []

        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            cur_probs = tf.nn.softmax(cur_logits, -1)
            scores, labels = tf.reduce_max(cur_probs, -1), tf.argmax(cur_probs, -1)
            keep = (labels == (outputs["pred_logits"].shape[-1] - 1)) & (scores > self.threshold)
            cur_scores, cur_classes = tf.gather_nd(scores, keep), tf.gather_nd(labels, keep)
            cur_masks = tf.gather_nd(cur_masks, keep)
            cur_masks = tf.squeeze(tf.image.resize(tf.expand_dims(cur_masks, -1), size, method="bilinear"), -1)
            cur_boxes = box_ops.box_cxcywh_to_xyxy(tf.gather_nd(cur_boxes, keep))

            h, w = cur_masks.shape[-2], cur_masks.shape[-1]
            assert cur_boxes.shape[0] == cur_classes.shape[0]

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k in range(cur_classes.shape[0]):
                if not self.is_thing_map[label[k]]:
                    stuff_equiv_classes[label[k]].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = tf.nn.softmax(masks, 0)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = tf.zeros((h, w), dtype=tf.int32)
                else:
                    m_id = tf.argmax(m_id, 0)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id = tf.where(m_id == eq_id, equiv[0], m_id)

                seg_img = id2rgb(tf.reshape(m_id, (h, w)).numpy())
                seg_img = tf.image.resize(seg_image, size=target_size, method="nearest")

                m_id = tf.image.resize(m_id, size=target_size, method="nearest")

                area = tf.reduce_sum(tf.expand_dims(m_id, -1) == tf.expand_dims(scores, [-3, -2]), [-3, -2])
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if tf.size(cur_classes) > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = tf.where(area <= 4)
                    if tf.reduce_any(filtered_small):
                        cur_scores = tf.gather_nd(cur_scores, ~filtered_small)
                        cur_classes = tf.gather_nd(cur_classes, ~filtered_small)
                        cur_masks = tf.gather_nd(cur_masks, ~filtered_small)
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = tf.ones(1, dtype=tf.int32)

            segments_info = []
            for i in range(area.shape[0]):
                cat = cur_classes[i]
                segments_info.append({"id": i, "isthing": self.is_thing_map[cat], "category_id": cat, "area": area[i]})
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
            preds.append(predictions)
        return preds
