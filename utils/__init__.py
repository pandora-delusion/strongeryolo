# -*- encoding: utf-8 -*-
# @TIME    : 2019/10/11 14:06
# @Author  : 成昭炜

import torch
from torch.autograd import Variable
import numpy as np
import torch.optim.lr_scheduler as F
import utils.visualization as vis
import model
import os
import sys
sys.path.append(os.path.abspath(".."))

from utils.compute_overlap import compute_overlap


class Trainer(object):

    def __init__(self, m, loss, lr, weight_decay, idx2cls, data_loader,
                 draw_freq=10, save_freq=1, iou_threshold=0.5):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        device_number = torch.cuda.device_count()
        print("device number: {}".format(device_number))
        m_ = model.Model(m, loss)
        prediction_model = model.Prediction(m)
        if device_number > 1:
            m = torch.nn.DataParallel(m_).cuda(self.device)
            prediction_model = torch.nn.DataParallel(prediction_model).cuda(self.device)
        self.model = m
        self.prediction_model = prediction_model
        # self.loss = loss
        #
        self.optimizer = torch.optim.Adam(params=self.model.module.parameters(),
                                          lr=lr, weight_decay=weight_decay)
        # self.optimizer = torch.optim.SGD(params=self.model.module.parameters(),
        #                                  lr=lr, weight_decay=weight_decay, momentum=0.9)
        self.scheduler = F.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.1, patience=2, verbose=True)
        self.draw_freq = draw_freq
        self.save_freq = save_freq

        self.iou_threshold = iou_threshold
        self.idx2cls = idx2cls
        self.data_loader = data_loader

    def _loop(self, data_loader):
        loop_loss = []
        idx = 0
        time = 3600*12
        for idx, (data, cls, reg) in enumerate(data_loader):
            if idx and idx % time == 0:
                self.save(idx)
            self.optimizer.zero_grad()
            data, cls, reg = Variable(data.cuda()), \
                             Variable(cls.cuda(), requires_grad=False),\
                             Variable(reg.cuda(), requires_grad=False)
            loss_t, p_t = self.model(data, (cls, reg))
            positive = torch.sum(p_t)
            if positive < 1.0:
                positive = 1.0
            loss_f = torch.sum(loss_t)/positive
            loss_data = float(loss_f.cpu().data.numpy())
            loop_loss.append(loss_data)
            print("batch: {} loss: {}".format(idx, loss_data))
            loss_f.backward()
            self.optimizer.step()
        avg_loss = sum(loop_loss)/(idx+1)
        return avg_loss

    def train(self, data_loader):
        self.model.train()
        # self.loss.train()
        return self._loop(data_loader)

    def _evaluate(self, loader):
        false_positives = {}
        true_positives = {}
        confidence = {}
        annotations = {}

        for idx, (images, scales, targets, sizes, _) in enumerate(loader):
            images = Variable(images.cuda(), requires_grad=False)
            sizes = Variable(sizes.cuda(), requires_grad=False)
            with torch.no_grad():
                boxes, scores, labels = self.prediction_model(images, sizes)

            mask = scores != -1
            indexes = mask.nonzero()
            for i in range(boxes.shape[0]):
                indice = indexes[indexes[:, 0] == i][:, 1]
                b = torch.index_select(boxes[i], dim=0, index=indice)
                l = torch.index_select(labels[i], dim=0, index=indice)
                value = torch.cat([l, b], dim=1).detach().cpu().numpy()

                value[:, 1:5] /= scales[i]
                s = torch.index_select(scores[i], dim=0, index=indice).cpu().numpy()

                pred_result, pred_ann = model.evaluate_batch(value, targets[i], s, self.iou_threshold)

                for m in pred_result:
                    if m not in false_positives:
                        false_positives[m] = []
                    if m not in true_positives:
                        true_positives[m] = []
                    if m not in confidence:
                        confidence[m] = []

                    pred_batch = pred_result[m]

                    for pred in pred_batch:
                        confidence[m].append(pred[0])
                        if pred[1] == 1:
                            true_positives[m].append(1)
                            false_positives[m].append(0)
                        else:
                            true_positives[m].append(0)
                            false_positives[m].append(1)

                for n in pred_ann:
                    if n not in annotations:
                        annotations[n] = 0
                    annotations[n] += pred_ann[n]

        average_precisions = {}
        for idx in confidence:
            conf = np.array(confidence[idx])
            false = np.array(false_positives[idx])
            true = np.array(true_positives[idx])

            indices = np.argsort(-conf)
            false = false[indices]
            true = true[indices]

            false = np.cumsum(false)
            true = np.cumsum(true)

            temp = annotations.get(idx, 0)

            if temp == 0:
                recall = np.zeros_like(true)
            else:
                recall = true / annotations[idx]
            precision = true / np.maximum(true + false, np.finfo(np.float64).eps)

            average_precisions[idx] = model.compute_ap(recall, precision)
        return average_precisions, annotations

    def validate(self, data_loader, epoch):
        self.model.eval()
        self.prediction_model.eval()
        total_instances = []
        precisions = []
        ap, ann = self._evaluate(data_loader)
        log_folder = "../log/"
        log_path = log_folder + "model_epoch_{}.txt".format(epoch)
        log = open(log_path, "w")
        for label in ann:
            number = ap.get(label, 0)
            records = "{: .0f} instances of class ".format(ann[label]) + \
                      self.idx2cls(label) + " with average precision: {:.8f}".format(number)
            print(records)
            log.write(records + "\n")
            total_instances.append(ann[label])
            precisions.append(number)
        temp = sum(x > 0 for x in total_instances)
        if temp == 0:
            mean_ap = 0.0
        else:
            mean_ap = sum(precisions)/temp
        print("mean ap: {:.8f}".format(mean_ap))
        log.write("mean ap: {:.8f}".format(mean_ap))
        log.close()
        return mean_ap

    def save(self, epoch):
        model_folder = "../checkpoints/"
        model_path = model_folder + "model_epoch_{}.pth".format(epoch)
        state = self.model.module.get_model().state_dict()
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        torch.save(state, model_path)

    def loop(self, epochs, train_data, val_data):

        # losses = []
        for ep in range(1, epochs+1):
            loss = self.train(train_data)
            map = self.validate(val_data, ep)
            self.save(ep)

    def handle_multi_labels(self, boxes, labels, scores):
        overlap = compute_overlap(boxes, boxes)
        boxes_number = boxes.shape[0]

        fathers = []
        for i in range(0, boxes_number):
            for j in range(i, boxes_number):
                father, _ = self.data_loader.judge_similar(labels[i, 0], labels[j, 0])
                if father != -1 and overlap[i, j] > 0.5:
                    fathers.append(father)

        fathers = set(fathers)
        sons = list(set(range(boxes_number))-fathers)
        sons.sort()
        boxes = boxes[sons]
        scores = scores[sons]
        labels = labels[sons]

        return boxes, labels, scores

    def predict(self, image_generator, idx2cls):
        import csv

        csvfile = open("submission.csv", "w")
        writer = csv.writer(csvfile)
        writer.writerow(["ImageId", "PredictionString"])
        # id_batch, image_batch, ratios, sizes
        for ids, images, ratios, origin_sizes, sizes in image_generator:

            with torch.no_grad():
                boxes, scores, labels = self.prediction_model(images, sizes)

            mask = scores != -1
            batch_number = boxes.shape[0]
            indexes = mask.nonzero()
            for i in range(batch_number):

                print(ids[i])
                strings = []
                indice = indexes[indexes[:, 0] == i][:, 1]
                b = torch.index_select(boxes[i], dim=0, index=indice)
                l = torch.index_select(labels[i], dim=0, index=indice)
                b = b.detach().cpu().numpy()
                l = l.detach().cpu().numpy()
                s = torch.index_select(scores[i], dim=0, index=indice).cpu().numpy()
                b /= ratios[i]

                # b, l, s = self.handle_multi_labels(b.astype(np.float64), l, s)

                height, width = origin_sizes[i, :]

                b[:, 0:3:2] /= width
                b[:, 1:4:2] /= height

                b = np.clip(b, 0.0, 1.0)

                for j in range(b.shape[0]):
                    label = int(l[j, 0])
                    label = idx2cls[label]
                    strings.append("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}"
                                   .format(label, s[j, 0], float(b[j, 0]),
                                           float(b[j, 1]), float(b[j, 2]), float(b[j, 3])))
                predict_string = " ".join(strings)
                writer.writerow([ids[i], predict_string])

        csvfile.close()
