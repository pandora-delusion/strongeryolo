# -*- encoding: utf-8 -*-
# @TIME    : 2019/10/14 13:17
# @Author  : 成昭炜

import torch
import torch.nn.functional as F
import torchvision
import numpy as np

from model.darknet53 import Darknet
import utils.anchors as anchors
from utils.compute_overlap import compute_overlap
from torch.autograd import Variable

import math


class Anchors(torch.nn.Module):

    def __init__(self, anchor_config):

        super(Anchors, self).__init__()
        self.sizes = anchor_config.sizes
        self.strides = anchor_config.strides
        self.ratios = anchor_config.ratios
        self.scales = anchor_config.scales

        self.num_anchors = anchor_config.num_anchors()

    def forward(self, feature):
        """
        :param feature: ([batch_size, num_channels, height, width], ...)
        :return:
        """
        lists = []
        for f in feature:
            lists.append(f.shape[2:])
        all_anchors = torch.from_numpy(anchors.anchors_for_feature(lists))
        # [batch, all_anchors, 4]
        all_anchors = all_anchors.unsqueeze(0).repeat(feature[0].shape[0], 1, 1)
        if feature[0].is_cuda:
            all_anchors = all_anchors.cuda().type(torch.float32)
        all_anchors = Variable(all_anchors, requires_grad=False)
        return all_anchors


class UpSample(torch.nn.Module):

    def __init__(self, mode="nearest"):
        super(UpSample, self).__init__()
        # self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode=self.mode)
        return x


class Reshape(torch.nn.Module):

    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x, shape):
        return x.contiguous().view(*shape)


class Permute(torch.nn.Module):

    def __init__(self, *shape):
        super(Permute, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.permute(*self.shape)


class Detection(torch.nn.Module):

    def __init__(self, backbone, num_anchors, num_classes, pyramid_number):
        super(Detection, self).__init__()
        self.pyramid_number = pyramid_number
        self.num_anchors = num_anchors
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_values = 4

        self.c5 = torch.nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.u5 = UpSample()
        self.r5 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.c4 = torch.nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.u4 = UpSample()
        self.r4 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.c3 = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.r3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.c6 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True)
        # self.c7 = torch.nn.ReLU(inplace=False)
        # self.r7 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True)

        self.classifications, self.regressions = self.task(256, 256)
        self.reshape = Reshape()

    def _initial_pyramid(self):
        for m in [self.c5, self.r5, self.c4, self.r4, self.c3, self.r3, self.c6]:
            # torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.normal_(m.weight, 0.0, 0.01)
            torch.nn.init.constant_(m.bias, 0)

    def init(self, probability=0.01):
        val = -math.log((1-probability)/probability)

        def init_(m):

            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.01)
                # torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        self.classifications.apply(init_)
        self.regressions.apply(init_)

        for cls in self.classifications:
            temp = cls[-3]
            torch.nn.init.constant_(temp.bias, val)

        self._initial_pyramid()

    def regression_task(self, pyramid_feature_size,
                        regression_feature_size=256):
        modules = []
        in_channels = pyramid_feature_size
        for i in range(4):
            modules.append(torch.nn.Conv2d(in_channels,
                                           regression_feature_size,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1))
            modules.append(torch.nn.ReLU(inplace=False))
            in_channels = regression_feature_size
        modules.append(torch.nn.Conv2d(in_channels,
                                       self.num_values*self.num_anchors,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1))
        modules.append(Permute(0, 2, 3, 1))
        return torch.nn.Sequential(*modules)

    def classification_task(self, pyramid_feature_size,
                            classification_feature_size=256):
        modules = []
        in_channels = pyramid_feature_size
        for i in range(4):
            modules.append(torch.nn.Conv2d(in_channels,
                                           classification_feature_size,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1))
            modules.append(torch.nn.ReLU(inplace=False))
            in_channels = classification_feature_size
        modules.append(torch.nn.Conv2d(in_channels,
                                       self.num_classes*self.num_anchors,
                                       kernel_size=3,
                                       stride=1, padding=1))
        modules.append(Permute(0, 2, 3, 1))
        modules.append(torch.nn.Sigmoid())
        return torch.nn.Sequential(*modules)

    def task(self, regression_feature_size, classification_feature_size):
        pyramids = self.pyramid_number

        classifications = []
        regressions = []
        for f in range(pyramids):
            classifications.append(self.classification_task(256, classification_feature_size))
            regressions.append(self.regression_task(256, regression_feature_size))
        return torch.nn.ModuleList(classifications), torch.nn.ModuleList(regressions)

    def forward(self, x):
        # 52 26 13
        f3, f4, f5 = self.backbone(x)

        f5 = self.c5(f5)
        temp = f4.shape[2:]
        f5_ = self.u5(f5, temp)
        f5 = self.r5(f5)

        f4 = self.c4(f4)
        f4 = f5_ + f4
        temp = f3.shape[2:]
        f4_ = self.u4(f4, temp)
        f4 = self.r4(f4)

        f3 = self.c3(f3)
        f3 = f4_ + f3
        f3 = self.r3(f3)

        f6 = self.c6(f5)
        # u7 = self.c7(f6)
        # f7 = self.r7(u7)
        features = [f3, f4, f5, f6]
        # features = [f3, f4, f5, f6]

        results = []

        for cls, f in zip(self.classifications, features):
            f = cls(f)
            shape = f.shape
            results.append(self.reshape(f, (shape[0], -1, self.num_classes)))

        cls = torch.cat(results, dim=1)

        results = []
        for reg, f in zip(self.regressions, features):
            f = reg(f)
            shape = f.shape
            results.append(self.reshape(f, (shape[0], -1, self.num_values)))

        reg = torch.cat(results, dim=1)

        # cls: [batch_size, all_anchors, num_classes]
        # reg: [batch_size, all_anchors, num_values]
        # features [(batch_size, channels, height, width), ...]
        return cls, reg, features


class Model(torch.nn.Module):

    def __init__(self, model, loss):
        super(Model, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, data, target):
        c, r, _ = self.model(data)
        cls, reg = target
        return self.loss((cls, reg), (c, r))

    def get_model(self):
        return self.model


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


class ClipBoxes(torch.nn.Module):

    def forward(self, bbox, shapes):

        res = torch.zeros_like(bbox)

        for i in range(bbox.shape[0]):
            res[i, :, 0] = torch.clamp(bbox[i, :, 0], 0, shapes[i, 1])
            res[i, :, 1] = torch.clamp(bbox[i, :, 1], 0, shapes[i, 0])
            res[i, :, 2] = torch.clamp(bbox[i, :, 2], 0, shapes[i, 1])
            res[i, :, 3] = torch.clamp(bbox[i, :, 3], 0, shapes[i, 0])

        return res


class Filter(torch.nn.Module):

    def __init__(self, nms=True,
                 nms_threshold=0.5, score_threshold=0.05,
                 class_specific_filter=False,
                 max_detections=300):
        super(Filter, self).__init__()
        self.nms = nms
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.class_specific_filter = class_specific_filter
        self.max_detections = max_detections

    def forward(self, box, cls):
        """
        fool!!!
        :param box: [batch_size, all_channels, num_values]
        :param cls: [batch_size, all_channels, num_classes]
        :return:
        """
        batch_size = box.shape[0]
        select_boxes = -1*torch.ones(batch_size, self.max_detections, box.shape[-1])
        select_labels = -1*torch.ones(batch_size, self.max_detections, 1)
        select_scores = -1*torch.ones(batch_size, self.max_detections, 1)

        select_boxes = select_boxes.cuda()
        select_labels = select_labels.cuda()
        select_scores = select_scores.cuda()

        for i in range(batch_size):
            boxes, scores, labels = self.filter_detection(box[i], cls[i])
            length = boxes.shape[0]
            select_boxes[i, :length, :] = boxes
            select_scores[i, :length, :] = scores
            select_labels[i, :length, :] = labels
        return select_boxes, select_scores, select_labels

    def filter_detection(self, boxes, classification):

        def _filter_detection(scores):

            select_mask = scores > self.score_threshold
            select_indexes = select_mask.nonzero()[:, 0]

            filtered_boxes = boxes[select_mask]
            filtered_scores = scores[select_mask]

            num_indices = torchvision.ops.nms(filtered_boxes,
                                              filtered_scores,
                                              iou_threshold=self.nms_threshold)
            select_indexes = select_indexes[num_indices][:self.max_detections]

            return select_indexes

        if not self.class_specific_filter:
            scores, labels = classification.max(dim=1)
            select_indexes = _filter_detection(scores)
            select_scores = scores[select_indexes]
            select_labels = labels[select_indexes]

        else:
            select_scores = []
            select_labels = []
            select_indexes = []
            for c in range(int(classification.shape[1])):
                scores = classification[:, c]
                select_idx = _filter_detection(scores)
                temp = scores[select_idx]
                select_scores.append(temp)
                select_indexes.append(select_idx)
                select_labels.append(torch.ones_like(temp, dtype=torch.int64)*c)

            select_scores = torch.cat(select_scores, 0)
            select_labels = torch.cat(select_labels, 0)
            select_indexes = torch.cat(select_indexes, 0)

        _, indexes = torch.topk(select_scores, k=min(self.max_detections, select_scores.shape[0]))
        indices = select_indexes[indexes]
        select_boxes = torch.index_select(boxes, 0, indices)
        select_labels = torch.index_select(select_labels, 0, indexes)
        select_scores = torch.index_select(select_scores, 0, indexes)
        select_labels.unsqueeze_(1)
        select_scores.unsqueeze_(1)

        select_labels = select_labels.int()

        return select_boxes, select_scores, select_labels


class Prediction(torch.nn.Module):

    def __init__(self, model):
        super(Prediction, self).__init__()
        self.model = model
        self.shifted_anchors = Anchors(anchors.parse_anchor_parameter())
        self.clip = ClipBoxes()
        self.filter = Filter()

    def forward(self, images, sizes):

        cls, reg, feature = self.model(images)

        # cls: [batch_size, all_anchors, num_classes]
        # reg: [batch_size, all_anchors, num_values]
        # feature: [(batch_size, height, width, channels), ...]
        # image: (batch_size, channels, height, width)

        # [batch, all_anchors, num_values]
        shifted_anchors = self.shifted_anchors(feature)
        bbox = anchors.bbox_transform_inv(shifted_anchors, reg)
        # [batch_size, all_anchors, num_classes]
        bbox = self.clip(bbox, sizes)
        # print("bbox: ", bbox.shape)
        selected = self.filter(bbox, cls)
        return selected


def evaluate_batch(pred, target, score, iou_threshold):

    pred_result = {}
    pred_annotation = {}
    pred_size = pred.shape[0]
    detected_annotations = []
    for i in range(pred_size):
        pred_label = int(pred[i, 0])
        if pred_label not in pred_result:
            pred_result[pred_label] = []

        # record [score, true or false]
        s_ = float(score[i, 0])

        box = np.expand_dims(pred[i, 1:5].astype(np.float64), axis=0)
        overlap = compute_overlap(box, target[:, 1:5].astype(np.float64))
        assigned_target = np.argmax(overlap, axis=1)
        max_overlap = overlap[0, assigned_target]

        if max_overlap >= iou_threshold and assigned_target not in detected_annotations:
            record = (s_, 1)
            detected_annotations.append(assigned_target)
        else:
            record = (s_, 0)
        pred_result[pred_label].append(record)

    for j in range(target.shape[0]):
        target_label = int(target[j, 0])
        if target_label not in pred_annotation:
            pred_annotation[target_label] = 0
        pred_annotation[target_label] += 1
    return pred_result, pred_annotation


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap







