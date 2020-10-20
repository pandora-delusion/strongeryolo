# -*- encoding: utf-8 -*-
# @TIME    : 2019/10/11 14:06
# @Author  : 成昭炜

from utils.augment import train_augment, RandomAffine, Resize, val_augment, stupid_process
from utils.visualization import plt_bboxes
import datasets.oid as dataset
from utils.anchors import generate_shifted_anchors, anchor_targets_bbox
from model.darknet53 import Darknet
from model.mobilenetv2 import MobileNet
import model
from loss import FocalLoss
from utils import Trainer
from config import cfg

import sys
import os
import torchvision
import torch
import numpy as np
import PIL
sys.path.append(os.path.abspath(".."))

root_path = os.path.abspath("..")


def merge_data(train_sets, is_multi_label=False):
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

    def _merge_data(batch):
        # C x H x W
        max_shape = tuple(max(img.shape[d] for img, _ in batch) for d in range(3))

        image_batch = []
        labels_batch = None
        if is_multi_label:
            labels_batch = []
        for image_idx, (img, anns) in enumerate(batch):
            temp = torch.zeros(max_shape, dtype=torch.float32)
            temp[:, :img.shape[1], :img.shape[2]] = img
            temp = normalize(temp)
            image_batch.append(temp)
            if is_multi_label:
                labels_batch.append(train_sets.handle_multi_label(anns))

        image_batch = torch.stack(image_batch, 0)
        anchors = generate_shifted_anchors(max_shape[1:])
        rgr, cls = anchor_targets_bbox(anchors, batch, len(train_sets.class2idx),
                                       multi_labels=labels_batch)
        reg = torch.from_numpy(rgr)
        cls = torch.from_numpy(cls)

        # [batch_size, channel, width, height]
        # [batch_size, all_anchors, num_classes+1]
        # [batch_size, all_anchors, num_values+1]
        return image_batch, cls, reg
    return _merge_data


def handle_val():
    normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])

    def _handle_val(batch):

        batch_size = len(batch)

        max_shape = tuple(max(img.shape[d] for img, _ in batch) for d in range(3))

        image_batch = []
        ratios = []
        targets = []
        sizes = np.zeros((batch_size, 2), dtype=np.float32)
        origins = []

        for image_idx, var in enumerate(batch):
            img, (scales, target, orgin) = var
            temp = torch.zeros(max_shape, dtype=torch.float32)
            temp[:, :img.shape[1], :img.shape[2]] = img
            temp = normalize(temp)
            image_batch.append(temp)
            ratios.append(scales)
            origins.append(orgin)
            targets.append(target)
            sizes[image_idx, :] += [img.shape[1], img.shape[2]]

        image_batch = torch.stack(image_batch, 0)
        sizes = torch.from_numpy(sizes)

        return image_batch, ratios, targets, sizes, origins
    return _handle_val


def handle_test():
    normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])

    def _handle_test(batch):

        batch_size = len(batch)

        max_shape = tuple(max(img.shape[d] for _, img, _ in batch) for d in range(3))

        image_batch = []
        origin_sizes = np.zeros((batch_size, 2), dtype=np.float32)
        id_batch = []
        ratios = []
        sizes = np.zeros((batch_size, 2), dtype=np.float32)

        for image_idx, var in enumerate(batch):
            file_id, image, info = var
            temp = torch.zeros(max_shape, dtype=torch.float32)
            temp[:, :image.shape[1], :image.shape[2]] = image
            sizes[image_idx, :] += [image.shape[1], image.shape[2]]
            id_batch.append(file_id)
            temp = normalize(temp)
            image_batch.append(temp)
            scale, _, origin = info
            ratios.append(scale)
            origin_sizes[image_idx, :] += [origin.size[1], origin.size[0]]
        image_batch = torch.stack(image_batch, 0)
        sizes = torch.from_numpy(sizes)

        return id_batch, image_batch, ratios, origin_sizes, sizes

    return _handle_test


os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TRAIN.CUDA_VISIBLE_DEVICES

GPU_NUMBER = len(cfg.TRAIN.CUDA_VISIBLE_DEVICES.split(","))

BATCH_SIZE = 8
VAL_SIZE = 51

train_augments = train_augment(
    torchvision.transforms.ColorJitter(
        brightness=(0.9, 1.1),
        contrast=(0.9, 1.1),
        saturation=(0.95, 1.05),
        hue=(-0.1, 0.1),
    ),

    None,

    RandomAffine(
        min_rotation=-0.1,
        max_rotation=0.1,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        min_shear=-0.1,
        max_shear=0.1,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_chance=0.5,
        flip_y_chance=0.5,
    ),

    stupid_process(),
    Resize()
)

validation_augment = val_augment(stupid_process(), None, Resize())

test_augment = val_augment(stupid_process(), None, Resize())

# sets = VOCDetection(os.path.join(root_path, "data"),
#                     image_set="train", transforms=train_augments)
# #
# vals = VOCDetection(os.path.join(root_path, "data"), image_set="val", transforms=validation_augment)
#

sets = dataset.TrainLoader(os.path.join(root_path, "data/oid"),
                           help_file=os.path.join(root_path, "data/train_info.txt"),
                           transforms=train_augments)
vals = dataset.ImageLoader(os.path.join(root_path, "data/oid"), "validation",
                           transforms=validation_augment)

tests = dataset.TestLoader(os.path.join(root_path, "data/oid"),
                           transform=test_augment)

valer = torch.utils.data.DataLoader(vals, batch_size=VAL_SIZE,
                                    shuffle=False,
                                    num_workers=VAL_SIZE,
                                    collate_fn=handle_val())

loader = torch.utils.data.DataLoader(sets,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     collate_fn=merge_data(sets, is_multi_label=True),
                                     num_workers=BATCH_SIZE,
                                     drop_last=False)

tester = torch.utils.data.DataLoader(tests,
                                     batch_size=VAL_SIZE,
                                     shuffle=False,
                                     collate_fn=handle_test(),
                                     num_workers=VAL_SIZE)

# for idx, (images, cls, reg) in enumerate(loader):
#
#     num = cls.shape[0]*cls.shape[1]
#     print("all examples: ", num)
#
#     positive = cls[cls[:, :, -1] == 1]
#     print("positive ratio: ", positive.shape[0]/num)
#     negative = cls[cls[:, :, -1] == 0]
#     print("negative: ", negative.shape[0]/num)

# net = MobileNet()
# net.load_state_dict(torch.load("../weights/mobilenet_v2-b0353104.pth"))
#
net = Darknet()
# net.load_darknet_weights("../weights/darknet53.conv.74")
net = model.Detection(net, 9, len(sets.idx2class), 4)
net.load_state_dict(torch.load("../checkpoints/model_epoch_2.pth"))
# net.init()
print(net)

# net = net.cuda()
# predict = model.Prediction(net).cuda()
loss = FocalLoss()
trainer = Trainer(net, loss, 1e-6, 1e-5, vals.get_label, vals)
# trainer.predict(tester, vals.idx2class)
# trainer.save("test")
# trainer.validate(valer, "001")
# trainer.loop_debug(loader, valer)
trainer.loop(1, loader, valer)


# from utils.visualization import plt_bboxes
#
#
# valer = torch.utils.data.DataLoader(vals, batch_size=1,
#                                     shuffle=False,
#                                     num_workers=1,
#                                     collate_fn=handle_val())
#
#
# for idx, (images, ratios, targets, sizes, origins) in enumerate(valer):
#     images = images.cuda()
#
#     with torch.no_grad():
#         boxes, scores, labels = predict(images, sizes)
#
#     mask = scores != -1
#     indexes = mask.nonzero()
#     indice = indexes[indexes[:, 0] == 0][:, 1]
#     b = torch.index_select(boxes[0], dim=0, index=indice)
#     l = torch.index_select(labels[0], dim=0, index=indice)
#     value = torch.cat([l, b], dim=1).detach().cpu().numpy()
#
#     value[:, 1:5] /= ratios[0]
#     plt_bboxes(origins[0], value, vals)
#     if idx > 100:
#         break




