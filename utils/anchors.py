# -*- encoding: utf-8 -*-
# @TIME    : 2019/10/13 9:40
# @Author  : 成昭炜

import numpy as np
import torch

from config import cfg
from utils.compute_overlap import compute_overlap


class AnchorConfig:

    def __init__(self, sizes, strides, ratios, scales, pyramid_levels):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales
        self.pyramid_levels = pyramid_levels

    def num_anchors(self):
        # return len(self.ratios)*len(self.scales)
        return len(self.ratios)


def guess_shapes(image_shape, pyramid_levels):
    """
    返回特征金字塔对应层特征图的尺寸
    :param image_shape: 图片尺寸，一个二元组(height, width)
    :param pyramid_levels: 特征金字塔列表
    :return: [np.array([h1, w1]), np.array([h2, w2])]
    """
    image_shape = np.array(image_shape)
    image_shapes = [(image_shape + 2**x-1)//(2**x) for x in pyramid_levels]
    return image_shapes


# AnchorConfig.default = AnchorConfig(
#     sizes=[32, 64, 128],
#     strides=[8, 16, 32],
#     # ratios=np.array([0.261, 0.571, 0.655, 0.947, 0.978, 1.22, 1.622, 1.627, 3.299], np.float32),
#     ratios=np.array([0.5, 1, 2], np.float32),
#     scales=np.array([2**0, 2**(1.0/3.0), 2**(2.0/3.0)], dtype=np.float32),
#     pyramid_levels=[3, 4, 5]
# )


def parse_anchor_parameter():
    ratio = np.array(list(map(float, cfg.ANCHOR.RATIOS.split(" "))), np.float32)
    scales = np.array(list(map(lambda x: 2**(float(x)/3.0), cfg.ANCHOR.SCALES.split(" "))), np.float32)

    sizes = list(map(int, cfg.ANCHOR.SIZES.split(" ")))
    strides = list(map(int, cfg.ANCHOR.STRIDES.split(" ")))
    pyramid_levels = list(map(int, cfg.ANCHOR.PYRAMID_LEVELS.split(" ")))

    return AnchorConfig(sizes, strides, ratio, scales, pyramid_levels)


AnchorConfig.default = parse_anchor_parameter()


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    :param base_size:
    :param ratios:
    :param scales:
    :return: (xmin, ymin, xmax, ymax)
    """
    if ratios is None:
        ratios = AnchorConfig.default.ratios

    if scales is None:
        scales = AnchorConfig.default.scales

    num_anchors = len(ratios)*len(scales)

    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size*np.tile(scales, (2, len(ratios))).T

    areas = anchors[:, 2]*anchors[:, 3]

    anchors[:, 2] = np.sqrt(areas/np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2]*np.repeat(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2]*0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3]*0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    """
    :param shape: 二元组[height, width]
    :param stride:
    :param anchors:
    :return:
    """
    shift_y, shift_x = np.mgrid[0:shape[0], 0:shape[1]]
    shift_x = (shift_x+0.5)*stride
    shift_y = (shift_y+0.5)*stride
    shift = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel())).transpose()

    anchors_num = anchors.shape[0]
    shifts_num = shift.shape[0]

    # (shifts_num, anchors_num, 4)
    anchors = anchors.reshape((1, anchors_num, 4)) + shift.reshape((1, shifts_num, 4)).transpose((1, 0, 2))

    anchors = anchors.reshape((anchors_num*shifts_num, 4))

    return anchors.astype(np.float32)


def anchors_for_shape(image_shape, anchor_params=None, shapes_callback=None):
    """

    :param image_shape: 图片的尺寸，一个三元组[height, width, channel]
    :param anchor_params: 一个AnchorConfig对象
    :param shapes_callback:返回特征图尺寸的函数
    :return:
    """

    if anchor_params is None:
        anchor_params = parse_anchor_parameter()

    pyramid_levels = anchor_params.pyramid_levels

    if shapes_callback is None:
        shapes_callback = guess_shapes

    image_shapes = shapes_callback(image_shape[:2], pyramid_levels)

    # 所有特征图的anchor
    all_anchors = np.zeros((0, 4))
    for idx in range(len(pyramid_levels)):
        # 生成匹配特征图的anchors
        anchors = generate_anchors(anchor_params.sizes[idx], anchor_params.ratios, anchor_params.scales)
        # 生成当前特征图所有位置的anchor
        shifted_anchors = shift(image_shapes[idx], anchor_params.strides[idx], anchors)

        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
    return all_anchors


def anchors_for_feature(feature_sizes, anchor_params=None):
    """

    :param feature_sizes: 这个是特征图的尺寸
    :param anchor_params:
    :return:
    """
    if anchor_params is None:
        anchor_params = parse_anchor_parameter()

    pyramid_levels = anchor_params.pyramid_levels

    all_anchors = np.zeros((0, 4))

    for idx in range(len(pyramid_levels)):
        anchors = generate_anchors(anchor_params.sizes[idx], anchor_params.ratios, anchor_params.scales)
        shifted_anchors = shift(feature_sizes[idx], anchor_params.strides[idx], anchors)

        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
    # print("all_anchors: ", all_anchors.shape)
    return all_anchors


def generate_shifted_anchors(image_shape, shapes_callback=guess_shapes):
    """
    :param image_shape: 是一个3元组(height, width, channel)
    :param shapes_callback:返回对应特征图尺寸的函数
    :return:
    """
    anchor_params = parse_anchor_parameter()
    return anchors_for_shape(image_shape, anchor_params, shapes_callback=shapes_callback)


def get_gt_indices(anchors, annotations, positive_overlap=0.5, negative_overlap=0.4):

    overlaps = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
    # 获得每个anchor对应的最大gt的idx
    best_overlap_each_anchor_indices = np.argmax(overlaps, axis=1)
    best_overlap_each_anchor = overlaps[np.arange(overlaps.shape[0]), best_overlap_each_anchor_indices]

    # print(np.max(best_overlap_each_anchor))
    positive_indices = best_overlap_each_anchor > positive_overlap
    ignore_indices = (best_overlap_each_anchor > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, best_overlap_each_anchor_indices


def bbox_transform(anchors, gt_bboxes):

    mean = np.array([0, 0, 0, 0])
    std = np.array([0.2, 0.2, 0.2, 0.2])

    anchors_width = anchors[:, 2] - anchors[:, 0]
    anchors_height = anchors[:, 3] - anchors[:, 1]

    target_dx1 = (gt_bboxes[:, 0] - anchors[:, 0]) / anchors_width
    target_dy1 = (gt_bboxes[:, 1] - anchors[:, 1]) / anchors_height
    target_dx2 = (gt_bboxes[:, 2] - anchors[:, 2]) / anchors_width
    target_dy2 = (gt_bboxes[:, 3] - anchors[:, 3]) / anchors_height

    targets = np.stack((target_dx1, target_dy1, target_dx2, target_dy2))
    targets = targets.T

    targets = (targets - mean) /std

    return targets


# def bbox_transform(anchors, gt_bboxes, prior_scaling=None):
#     if prior_scaling is None:
#         prior_scaling = [0.1, 0.1, 0.2, 0.2]
#
#     anchor_widths = anchors[:, 2] - anchors[:, 0]
#     anchor_heights = anchors[:, 3] - anchors[:, 1]
#
#     anchor_cx = (anchors[:, 2] + anchors[:, 0])/2.0
#     anchor_cy = (anchors[:, 3] + anchors[:, 1])/2.0
#
#     bbox_cx = (gt_bboxes[:, 2] + gt_bboxes[:, 0])/2.0
#     bbox_cy = (gt_bboxes[:, 3] + gt_bboxes[:, 1])/2.0
#
#     bbox_widths = gt_bboxes[:, 2] - gt_bboxes[:, 0]
#     bbox_heights = gt_bboxes[:, 3] - gt_bboxes[:, 1]
#
#     target_cx = (bbox_cx - anchor_cx)/anchor_widths/prior_scaling[0]
#     target_cy = (bbox_cy - anchor_cy)/anchor_heights/prior_scaling[1]
#
#     target_width = np.log(bbox_widths/anchor_widths)/prior_scaling[2]
#     target_height = np.log(bbox_heights/anchor_heights)/prior_scaling[3]
#
#     target = np.stack((target_cx, target_cy, target_width, target_height), axis=-1)

    # return target


def bbox_transform_inv(boxes, deltas, mean=None, std=None):

    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0]*std[0] + mean[0])*width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1]*std[1] + mean[1])*height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2]*std[2] + mean[2])*width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3]*std[3] + mean[3])*height

    pred_boxes = torch.stack([x1, y1, x2, y2], dim=2)

    return pred_boxes


# def bbox_transform_inv(anchors, deltas, prior_scaling=None):
#     """
#
#     :param anchors:
#     :param deltas: [batch_size, all_anchors, num_values]
#     :param prior_scaling:
#     :return:
#     """
#     # print("anchors: ", anchors.shape)c
#     # print("deltas: ", deltas.shape)
#     if prior_scaling is None:
#         prior_scaling = [0.1, 0.1, 0.2, 0.2]
#
#     anchors_cx = (anchors[:, :, 2] + anchors[:, :, 0])/2.0
#     anchors_cy = (anchors[:, :, 3] + anchors[:, :, 1])/2.0
#
#     anchors_weights = anchors[:, :, 2] - anchors[:, :, 0]
#     anchors_heights = anchors[:, :, 3] - anchors[:, :, 1]
#
#     pred_cx = anchors_weights*prior_scaling[0]*deltas[:, :, 0] + anchors_cx
#     pred_cy = anchors_heights*prior_scaling[1]*deltas[:, :, 1] + anchors_cy
#     pred_w = torch.exp(deltas[:, :, 2]*prior_scaling[2])*anchors_weights
#     pred_h = torch.exp(deltas[:, :, 3]*prior_scaling[3])*anchors_heights
#
#     x_min = pred_cx - pred_w/2.0
#     y_min = pred_cy - pred_h/2.0
#     x_max = pred_cx + pred_w/2.0
#     y_max = pred_cy + pred_h/2.0
#
#     return torch.stack([x_min, y_min, x_max, y_max], dim=2)


def anchor_targets_bbox(anchors,
                        batch,
                        num_classes,
                        negative_overlap=0.4,
                        positive_overlap=0.5,
                        multi_labels=None):

    batch_size = len(batch)

    regression = np.zeros((batch_size, anchors.shape[0], 4+1), dtype=np.float32)
    classification = np.zeros((batch_size, anchors.shape[0], num_classes + 1), dtype=np.float32)

    if multi_labels is not None:
        is_multi = True
    else:
        is_multi = False

    for idx, (image, annotations) in enumerate(batch):

        if annotations.shape[0]:
            bbox = annotations[:, 1:5]
            positive_indices, ignore_indices, best_overlap_indices = get_gt_indices(anchors, bbox,
                                                                                    positive_overlap, negative_overlap)
            classification[idx, ignore_indices, -1] = -1
            classification[idx, positive_indices, -1] = 1
            classification[idx, positive_indices,
                           annotations[:, 0][best_overlap_indices[positive_indices]].astype(np.int)] = 1

            if is_multi:
                p_idx = best_overlap_indices[positive_indices]
                anchor_indices = np.where(positive_indices)[0]
                labels = multi_labels[idx]
                for i, a_idx in enumerate(anchor_indices):
                    classification[idx, a_idx, labels[p_idx[i]]] = 1

            regression[idx, ignore_indices, -1] = -1
            regression[idx, positive_indices, -1] = 1
            # 有点浪费计算资源
            regression[idx, :, :-1] = bbox_transform(anchors, bbox[best_overlap_indices, :])

        # 去掉图片外的标注
        if image.shape:
            width = image.shape[2]
            height = image.shape[1]
            anchors_center = np.vstack([(anchors[:, 0]+anchors[:, 2])/2.0,
                                        (anchors[:, 1]+anchors[:, 3])/2.0]).T
            indices = np.logical_or(anchors_center[:, 0] >= width,
                                    anchors_center[:, 1] >= height)
            classification[idx, indices, -1] = -1
            regression[idx, indices, -1] = -1

    return regression, classification




