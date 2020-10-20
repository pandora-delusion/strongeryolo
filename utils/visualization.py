# -*- encoding: utf-8 -*-
# @TIME    : 2019/10/12 16:17
# @Author  : 成昭炜

import matplotlib.pyplot as plt
import random
import numpy as np
import datasets.voc as VOC


def plt_bboxes(img, bboxes, loader, figsize=(10, 10), linewidth=1.5):

    fig = plt.figure(figsize=figsize)
    plt.imshow(img)

    colors = dict()
    for i in range(bboxes.shape[0]):
        cls_id = int(bboxes[i][0])
        if cls_id >= 0:
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())

            xmin = int(bboxes[i, 1])
            ymin = int(bboxes[i, 2])
            xmax = int(bboxes[i, 3])
            ymax = int(bboxes[i, 4])

            rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False,
                                 edgecolor=colors[cls_id], linewidth=linewidth)
            plt.gca().add_patch(rect)
            # class_name = IDX2CLS[cls_id]
            class_name = loader.get_label(cls_id)
            plt.gca().text(xmin, ymin-2, "{:s}".format(class_name),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()


def show_train_info(losses_train):
    batches = np.arange(len(losses_train))
    losses_train = np.array(losses_train)
    plt.plot(batches, losses_train)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


def show_acc_info(maps):
    epoches = np.arange(len(maps))
    maps = np.array(maps)
    plt.plot(epoches, maps)
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()


def show_cls_distribution(cls_positive, cls_negative, cls_hard, cls_easy):

    plt.figure(figsize=(16, 8), dpi=80)

    plots = [221, 222, 223, 224]
    titles = ["positive", "negative", "hard", "easy"]
    classes = [cls_positive, cls_negative, cls_hard, cls_easy]

    for plot, title, cls in zip(plots, titles, classes):

        plt.subplot(plot)
        if np.size(cls) <= 0:
            continue
        a = np.max(cls)
        b = np.min(cls)
        nums = (a-b)//1e-5
        # print(title)
        if nums == 0:
            continue
        plt.hist(cls, 100)
        plt.xticks(np.arange(b, a+1, step=0.0001))
        plt.xlabel("value")
        plt.ylabel("num")

    plt.show()
