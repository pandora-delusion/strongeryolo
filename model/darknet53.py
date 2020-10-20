# -*- encoding: utf-8 -*-
# @TIME    : 2019/10/14 13:59
# @Author  : 成昭炜

import torch
import numpy as np

import model


def downsample_block(name, in_channels, out_channels, bias=False):
    modules = torch.nn.Sequential()
    downsample = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                                 padding=1, bias=bias)
    modules.add_module(f"downSample_{name}", downsample)
    modules.add_module(f"batchNorm_{name}", torch.nn.BatchNorm2d(out_channels,
                                                                 momentum=0.9, eps=1e-5))
    modules.add_module(f"leaky_{name}", torch.nn.LeakyReLU(0.1))

    return modules


def base_block(name, in_channels, out_channels, kernel_size, bias=False):
    modules = torch.nn.Sequential()

    padding = (kernel_size - 1) // 2
    conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                           bias=bias)
    modules.add_module(f"conv_{name}", conv)
    modules.add_module(f"batchNorm_{name}", torch.nn.BatchNorm2d(out_channels, momentum=0.9,
                                                                 eps=1e-5))
    modules.add_module(f"leaky_{name}", torch.nn.LeakyReLU(0.1))
    return modules


class Residual(torch.nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):

        super(Residual, self).__init__()
        self.block1 = base_block(1, in_channels, out_channels, kernel_size=1, bias=bias)
        self.block2 = base_block(2, out_channels, out_channels*2, kernel_size=3, bias=bias)

        self.sample = None
        if in_channels != out_channels*2:
            self.sample = torch.nn.Conv2d(in_channels, out_channels*2, kernel_size=1, bias=False)

    def forward(self, x):
        res = self.block1(x)
        res = self.block2(res)
        if self.sample is not None:
            x = self.sample(x)
        return x + res


class Darknet(torch.nn.Module):

    def __init__(self):

        super(Darknet, self).__init__()
        self.block1 = base_block(1, in_channels=3, out_channels=32, kernel_size=3)
        self.downSample1 = downsample_block(2, in_channels=32, out_channels=64)
        self.block2 = self.build_block(1, 64, 32)
        self.downSample2 = downsample_block(3, in_channels=64, out_channels=128)
        self.block3 = self.build_block(2, 128, 64)
        self.downSample3 = downsample_block(4, in_channels=128, out_channels=256)
        self.block4 = self.build_block(8, 256, 128)
        self.downSample4 = downsample_block(5, in_channels=256, out_channels=512)
        self.block5 = self.build_block(8, 512, 256)
        self.downSample5 = downsample_block(6, in_channels=512, out_channels=1024)
        self.block6 = self.build_block(4, 1024, 512)

    @staticmethod
    def build_block(number, in_channels, out_channels):

        groups = []
        for i in range(number):
            groups.append(Residual(in_channels, out_channels))

        return torch.nn.Sequential(*groups)

    def forward(self, x):
        results = []
        res = self.block1(x)
        res = self.downSample1(res)
        res = self.block2(res)
        res = self.downSample2(res)
        res = self.block3(res)
        res = self.downSample3(res)
        res = self.block4(res)
        results.append(res)
        res = self.downSample4(res)
        res = self.block5(res)
        results.append(res)
        res = self.downSample5(res)
        res = self.block6(res)
        results.append(res)
        return results

    def load_darknet_weights(self, weights_path):

        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            # header_info = header
            # seen = header[3]
            weights = np.fromfile(f, dtype=np.float32)
            print(len(weights))

        ptr = 0
        stack = []
        for i, m in enumerate(self.modules()):
            if isinstance(m, torch.nn.Conv2d):
                stack.append(m)
            elif isinstance(m, torch.nn.BatchNorm2d):
                # 加载BN层的偏置和权重 运行时均值和运行时标准差
                num_b = m.bias.numel()
                bn_b = torch.from_numpy(weights[ptr:ptr+num_b]).view_as(m.bias)
                m.bias.data.copy_(bn_b)
                ptr += num_b
                bn_w = torch.from_numpy(weights[ptr:ptr+num_b]).view_as(m.weight)
                m.weight.data.copy_(bn_w)
                ptr += num_b
                bn_rm = torch.from_numpy(weights[ptr:ptr+num_b]).view_as(m.running_mean)
                m.running_mean.data.copy_(bn_rm)
                ptr += num_b
                bn_rv = torch.from_numpy(weights[ptr:ptr+num_b]).view_as(m.running_var)
                m.running_var.data.copy_(bn_rv)
                ptr += num_b

                conv = stack.pop(-1)
                num_w = conv.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr+num_w]).view_as(conv.weight)
                conv.weight.data.copy_(conv_w)
                ptr += num_w

        print(ptr)
