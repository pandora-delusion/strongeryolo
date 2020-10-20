# -*- encoding: utf-8 -*-
# @TIME    : 2019/10/12 14:37
# @Author  : 成昭炜

import numpy as np


def preprocess_image(x, mode="caffe"):
    x = x.astype(np.float32)

    if mode == "tf":
        x /= 127.5
        x -= 1.0
    elif mode == "caffe":
        x[..., 2] -= 103.939
        x[..., 1] -= 116.779
        x[..., 0] -= 123.68
    return x
