from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from utils.cython_bbox import bbox_overlaps

try:
    import cPickle as pickle
except ImportError:
    import pickle


# Just return the ground truth boxes for a single image
def compute_target(memory_size, gt_boxes, feat_stride):
    factor_h = (memory_size[0] - 1.) * feat_stride
    factor_w = (memory_size[1] - 1.) * feat_stride
    num_gt = gt_boxes.shape[0]

    x1 = gt_boxes[:, [0]] / factor_w
    y1 = gt_boxes[:, [1]] / factor_h
    x2 = gt_boxes[:, [2]] / factor_w
    y2 = gt_boxes[:, [3]] / factor_h

    rois = np.hstack((y1, x1, y2, x2))
    batch_ids = np.zeros((num_gt), dtype=np.int32)
    # overlap to regions of interest
    roi_overlaps = np.ones((num_gt), dtype=np.float32)
    labels = np.array(gt_boxes[:, 4], dtype=np.int32)

    return rois, batch_ids, roi_overlaps, labels


# Also return the reverse index of rois
def compute_target_memory(memory_size, rois, feat_stride):
    """

    :param memory_size: [H/16, W/16], shape of memory
    :param rois: [N, 5], for (batch_id, x1, y1, x2, y2)
    :param labels: [N,], roi labels
    :param feat_stride: 16
    :return:
    """
    minus_h = memory_size[0] - 1.
    minus_w = memory_size[1] - 1.
    num_roi = rois.shape[0]
    assert np.all(rois[:, 0] == 0), 'only support single image per batch.'

    x1 = rois[:, [1]] / feat_stride
    y1 = rois[:, [2]] / feat_stride
    x2 = rois[:, [3]] / feat_stride
    y2 = rois[:, [4]] / feat_stride

    # h, w, h, w
    n_rois = np.hstack((y1, x1, y2, x2))
    n_rois[:, 0::2] /= minus_h
    n_rois[:, 1::2] /= minus_w
    batch_ids = np.zeros(num_roi, dtype=np.int32)

    # h, w, h, w
    inv_rois = np.empty_like(n_rois)
    inv_rois[:, 0:2] = 0.
    inv_rois[:, 2] = minus_h
    inv_rois[:, 3] = minus_w
    inv_rois[:, 0::2] -= y1
    inv_rois[:, 1::2] -= x1

    # normalize coordinates
    inv_rois[:, 0::2] /= np.maximum(y2 - y1, cfg.EPS)
    inv_rois[:, 1::2] /= np.maximum(x2 - x1, cfg.EPS)

    inv_batch_ids = np.arange(num_roi, dtype=np.int32)

    return n_rois, batch_ids, inv_rois, inv_batch_ids


def compute_rel_rois(num_rel, rois, relations):
    """
    union subject boxes and object boxes given a set of rois and relations
    """
    rel_rois = np.zeros([num_rel, 5])
    for i, rel in enumerate(relations):
        sub_im_i = rois[rel[0], 0]
        obj_im_i = rois[rel[1], 0]
        assert(sub_im_i == obj_im_i)
        rel_rois[i, 0] = sub_im_i

        sub_roi = rois[rel[0], 1:]
        obj_roi = rois[rel[1], 1:]
        union_roi = [np.minimum(sub_roi[0], obj_roi[0]),
                    np.minimum(sub_roi[1], obj_roi[1]),
                    np.maximum(sub_roi[2], obj_roi[2]),
                    np.maximum(sub_roi[3], obj_roi[3])]
        rel_rois[i, 1:] = union_roi

    return rel_rois


# Update weights for the target
def update_weights(labels, cls_prob):
    num_gt = labels.shape[0]
    index = np.arange(num_gt)
    cls_score = cls_prob[index, labels]
    big_ones = cls_score >= 1. - cfg.MEM.BETA
    # Focus on the hard examples
    weights = 1. - cls_score
    weights[big_ones] = cfg.MEM.BETA
    weights /= np.maximum(np.sum(weights), cfg.EPS)

    return weights
