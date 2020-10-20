# -*- encoding: utf-8 -*-
# @TIME    : 2019/10/11 17:19
# @Author  : 成昭炜

import numpy as np
from PIL import Image
import cv2
import torchvision


class TransformParameters:

    def __init__(self, fill_mode="nearest",
                 interpolation="linear",
                 cval=0,
                 relative_translation=True):
        # 边界填充模式
        self.fill_mode = fill_mode
        self.cval = cval
        # 插值模式
        self.interpolation = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == "constant":
            return cv2.BORDER_CONSTANT
        elif self.fill_mode == "nearest":
            return cv2.BORDER_REPLICATE
        elif self.fill_mode == "reflect":
            return cv2.BORDER_REFLECT_101
        elif self.fill_mode == "wrap":
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        # 最邻近插值
        if self.interpolation == "nearest":
            return cv2.INTER_NEAREST
        # 线性插值
        elif self.interpolation == "linear":
            return cv2.INTER_LINEAR
        # 三次样条插值
        elif self.interpolation == "cubic":
            return cv2.INTER_CUBIC
        # 区域插值
        elif self.interpolation == "area":
            return cv2.INTER_AREA
        # lanczos插值
        elif self.interpolation == "lanczos4":
            return cv2.INTER_LANCZOS4


def random_vector(mini, maxi, default):
    mini = np.array(mini)
    maxi = np.array(maxi)
    assert mini.shape == maxi.shape
    return default(mini, maxi)


def rotation(angle):
    """
    构造一个二维旋转矩阵
    :param angle:
    :return:
    """
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])


def translation(trans):
    """
    构造一个二维平移矩阵
    :param trans:
    :return:
    """
    return np.array([
        [1, 0, trans[0]],
        [0, 1, trans[1]],
        [0, 0, 1]
    ])


def shear(angle):
    """
    构造一个二维shear矩阵
    :param angle:
    :return:
    """
    return np.array([
        [1, -np.sin(angle), 0],
        [0, np.cos(angle), 0],
        [0, 0, 1]
    ])


def scaling(factor):
    """
    构造二维缩放矩阵
    :param factor:
    :return:
    """
    return np.array([
        [factor[0], 0, 0],
        [0, factor[1], 0],
        [0, 0, 1]
    ])


def flip(flip_x, flip_y):
    _x = int(flip_x)
    _y = int(flip_y)
    return np.array([
        [(-1)**_x, 0, _x],
        [0, (-1)**_y, _y],
        [0, 0, 1]
    ])


class RandomAffine(object):

    default = np.random

    def __init__(self, min_rotation=-0.1,
                 max_rotation=0.1,
                 min_translation=(-0.1, -0.1),
                 max_translation=(0.1, 0.1),
                 min_shear=-0.1,
                 max_shear=0.1,
                 min_scaling=(0.9, 0.9),
                 max_scaling=(1.1, 1.1),
                 flip_x_chance=0.5,
                 flip_y_chance=0.5):

        self.rotation = (min_rotation, max_rotation)
        self.min_translation = min_translation
        self.max_translation = max_translation
        self.shear = (min_shear, max_shear)
        self.min_scaling = min_scaling
        self.max_scaling = max_scaling
        self.flip_x_chance = flip_x_chance
        self.flip_y_chance = flip_y_chance

        self.transform_parameters = TransformParameters()

    @staticmethod
    def get_params(rot, min_translation, max_translation, sh,
                   min_scaling, max_scaling, flip_x_chance, flip_y_chance):
        rot = RandomAffine.default.uniform(rot[0], rot[1])
        trans = random_vector(min_translation, max_translation, RandomAffine.default.uniform)
        sh = RandomAffine.default.uniform(sh[0], sh[1])
        scale = random_vector(min_scaling, max_scaling, RandomAffine.default.uniform)
        fl = (RandomAffine.default.uniform(0, 1) < flip_x_chance,
              RandomAffine.default.uniform(0, 1) < flip_y_chance)

        return rot, trans, sh, scale, fl

    @staticmethod
    def transform_coordinate(transform, aabb):
        xmin, ymin, xmax, ymax = aabb

        points = transform.dot([
            [xmin, xmax, xmin, xmax],
            [ymin, ymax, ymax, ymin],
            [1, 1, 1, 1],
        ])

        min_corner = points[0:2, :].min(axis=1)
        max_corner = points[0:2, :].max(axis=1)

        return [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]

    def __call__(self, img, target):
        rot, trans, sh, scale, fl = self.get_params(self.rotation,
                                                    self.min_translation,
                                                    self.max_translation,
                                                    self.shear,
                                                    self.min_scaling,
                                                    self.max_scaling,
                                                    self.flip_x_chance,
                                                    self.flip_y_chance)

        random_rotation = rotation(rot)
        random_translation = translation(trans)
        random_shear = shear(sh)
        random_scaling = scaling(scale)
        random_flip = flip(fl[0], fl[1])

        transform = np.linalg.multi_dot([
            random_rotation,
            random_translation,
            random_shear,
            random_scaling,
            random_flip
        ])

        height = img.shape[0]
        width = img.shape[1]

        transform[0, 2] *= width
        transform[1, 2] *= height

        params = self.transform_parameters

        img = cv2.warpAffine(img, transform[:2, :], dsize=(width, height), flags=params.cvInterpolation(),
                             borderMode=params.cvBorderMode(),
                             borderValue=params.cval)

        for idx in range(target.shape[0]):
            target[idx, 1:5] = self.transform_coordinate(transform, target[idx, 1:5])

        return img, target


class Resize(object):

    def __init__(self, min_side=400, max_side=400):
        self.min_side = min_side
        self.max_side = max_side

    def __call__(self, img, target=None):
        width = img.shape[1]
        height = img.shape[0]

        small_side = min(width, height)

        scale = self.min_side/small_side

        large_side = max(width, height)

        if large_side*scale > self.max_side:
            scale = self.max_side/large_side

        img = cv2.resize(img, None, fx=scale, fy=scale)

        if target is not None:
            target[:, 1:5] *= scale

        return img, target, scale


def img2matrix(img):
    return np.asarray(img.convert("RGB"))


def matrix2img(matrix):
    return Image.fromarray(matrix, "RGB")


def normalizer_image(x):
    x = x.astype(np.float32)
    x /= 255.0
    return x


def stupid_process():
    toTensor = torchvision.transforms.ToTensor()

    def _process(image):
        image = matrix2img(image)
        image = toTensor(image)
        return image
    return _process


def train_augment(visual_effect, handle_target, transform, preprocess, resize):

    def _train_augment(image, target):

        image = visual_effect(image)
        image = img2matrix(image)
        if handle_target:
            target = handle_target(target)
        image, target = transform(image, target)
        image, target, _ = resize(image, target)

        image = preprocess(image)
        return image, target

    return _train_augment


def val_augment(preprocess, handle_target, resize):

    def _val_augment(image, target):
        orgin = image
        image = img2matrix(image)
        if handle_target:
            target = handle_target(target)
        image, _, scale = resize(image, None)
        image = preprocess(image)
        return image, (scale, target, orgin)
    return _val_augment


def simple_augment():

    train_augment = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])

    val_augment = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])

    return train_augment, val_augment



















