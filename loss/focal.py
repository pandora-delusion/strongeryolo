# -*- encoding: utf-8 -*-
# @TIME    : 2019/10/17 19:34
# @Author  : 成昭炜

import torch
import numpy as np
import utils.visualization as vis


class FocalLoss(torch.nn.Module):

    def __init__(self, alphas=0.25, gamma=2.0, sigma=3.0):
        super(FocalLoss, self).__init__()
        self.alphas = alphas
        self.gamma = gamma
        self.sigma = sigma
        self.sigma_squared = self.sigma**2

    def _smooth_l1(self, true, pred):

        results = pred
        targets = true[:, :, :-1]
        states = true[:, :, -1]

        mask = states == 1
        results = results[mask]
        targets = targets[mask]

        difference = torch.abs(results - targets)
        loss = torch.where(torch.lt(difference, 1.0/self.sigma_squared),
                           0.5*self.sigma_squared*torch.pow(difference, 2),
                           difference-0.5/self.sigma_squared)

        # positive = results.shape[0]
        # if positive < 1.0:
        #     positive = 1.0
        return loss

    def _focal(self, true, pred):
        labels = true[:, :, :-1]
        states = true[:, :, -1]
        predictions = pred

        mask = states != -1
        labels = labels[mask]
        predictions = predictions[mask]

        alphas_factor = torch.ones_like(labels, requires_grad=False)*self.alphas
        alphas_factor = torch.where(labels == 1, alphas_factor, 1.0 - alphas_factor)
        focal_weight = torch.where(labels == 1, 1.0-predictions, predictions)
        focal_weight = alphas_factor*focal_weight**self.gamma
        cls_loss = focal_weight*torch.nn.functional.binary_cross_entropy(predictions, labels, reduction="none")

        # positive_mask = states == 1
        # positive = torch.masked_select(states, positive_mask)
        # positive = positive.shape[0]
        # if positive < 1.0:
        #     positive = 1.0
        # print("positive: ", positive)
        return cls_loss

    # def __call__(self, true, pred):
    #     cls_true, reg_true = true
    #     cls_pred, reg_pred = pred
    #     cls_loss = self._focal(cls_true, cls_pred)
    #     reg_loss = self._smooth_l1(reg_true, reg_pred)
    #     return cls_loss + reg_loss

    def forward(self, true, pred):
        cls_true, reg_true = true
        cls_pred, reg_pred = pred
        cls_loss = self._focal(cls_true, cls_pred)
        reg_loss = self._smooth_l1(reg_true, reg_pred)

        states = reg_true[:, :, -1]
        positive_mask = states == 1
        positive = positive_mask.nonzero().shape[0]
        loss = torch.sum(cls_loss) + torch.sum(reg_loss)
        loss.unsqueeze_(0)
        positive = torch.tensor(positive, dtype=torch.float32, requires_grad=False).cuda()
        positive.unsqueeze_(0)

        return loss, positive
