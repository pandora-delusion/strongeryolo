# -*- encoding: utf-8 -*-
# @TIME    : 2019/10/13 9:49
# @Author  : 成昭炜

from __future__ import division

from easydict import EasyDict
import sys
import os

root_path = os.path.abspath("..")
sys.path.append(root_path)
__C = EasyDict()
cfg = __C

__C.FILE = EasyDict()
__C.FILE.BASE_PATH = root_path
__C.FILE.CLASS_HIERARCHY_FILE = "data/challenge-2019-label500-hierarchy.json"
__C.FILE.TRAIN_BBOX_DIR = "data/labels/train/"
__C.FILE.VAL_BBOX_DIR = "data/labels/validation/"
__C.FILE.TRAIN_BBOX_FILE = "data/challenge-2019-train-detection-bbox.csv"
__C.FILE.VAL_BBOX_FILE = "data/challenge-2019-validation-detection-bbox.csv"
__C.FILE.CLASS_DESCRIPTION = "data/challenge-2019-classes-description-500.csv"
__C.FILE.BBOX_DIR = "data/labels/"
__C.FILE.IMAGES_TRAIN_DIR = "data/oid/train/"
__C.FILE.IMAGES_VALIDATION_DIR = "data/oid/validation/"
__C.FILE.TRAIN_INFO = "data/train_info.txt"
__C.FILE.SNAPSHOT_PATH = "checkpoint/"

__C.BBOX = EasyDict()
__C.BBOX.CSV_IMAGEID = "ImageID"
__C.BBOX.CSV_SOURCE = "Source"
__C.BBOX.CSV_LABELNAME = "LabelName"
__C.BBOX.CSV_CONFIDENCE = "Confidence"
__C.BBOX.CSV_XMIN = "XMin"
__C.BBOX.CSV_XMAX = "XMax"
__C.BBOX.CSV_YMIN = "YMin"
__C.BBOX.CSV_YMAX = "YMax"
__C.BBOX.CSV_ISOCCLUDED = "IsOccluded"
__C.BBOX.CSV_ISTRUNCATES = "IsTruncated"
__C.BBOX.CSV_ISGROUPOF = "IsGroupOf"
__C.BBOX.CSV_ISDEPICTION = "IsDepiction"
__C.BBOX.CSV_ISINSIDE = "IsInside"

__C.ANCHOR = EasyDict()
__C.ANCHOR.RATIOS = "0.5 1 2"
__C.ANCHOR.SCALES = "0 1 2"
__C.ANCHOR.SIZES = "32 64 128 256"
__C.ANCHOR.STRIDES = "8 16 32 64"
__C.ANCHOR.PYRAMID_LEVELS = "3 4 5 6"

__C.TRAIN = EasyDict()
__C.TRAIN.CUDA_VISIBLE_DEVICES = "0,1,2"
