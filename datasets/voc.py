# -*- encoding: utf-8 -*-
# @TIME    : 2019/10/12 14:37
# @Author  : 成昭炜

import numpy as np
import torchvision
import os
from PIL import Image
import collections
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


CLASSES = ["car", "bus", "bicycle", "motorbike", "aeroplane", "boat", "train", "chair", "sofa",
           "diningtable", "tvmonitor", "bottle", "pottedplant", "cat", "dog", "cow", "horse", "sheep",
           "bird", "person"]

CLS2IDX = dict([(cls, idx) for cls, idx in zip(CLASSES, range(len(CLASSES)))])
IDX2CLS = dict([(idx, cls) for cls, idx in zip(CLASSES, range(len(CLASSES)))])

BBOX = "bndbox"

DATASET_YEAR_DICT = {
    "2007": {
        "base_dir": "VOCdevkit/VOC2007",
    },
    "2012": {
        "base_dir": "VOCdevkit/VOC2012",
    }
}


def handle_voc_target(target):

    objects = target["annotation"]["object"]
    if isinstance(objects, dict):
        objects = [objects]

    values = np.zeros((len(objects), 5), dtype=np.float32)

    for idx, obj in enumerate(objects):
        xmin = int(obj[BBOX]["xmin"])
        ymin = int(obj[BBOX]["ymin"])
        xmax = int(obj[BBOX]["xmax"])
        ymax = int(obj[BBOX]["ymax"])
        values[idx, :] += np.array([CLS2IDX[obj["name"]], xmin, ymin, xmax, ymax])

    return values


class VOCDetection(torchvision.datasets.VisionDataset):

    def __init__(self,
                 root,
                 image_set="train",
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(VOCDetection, self).__init__(root, transforms, transform, target_transform)
        self.years = ["2007", "2012"]

        base_dirs = [DATASET_YEAR_DICT[year]["base_dir"] for year in self.years]
        voc_roots = [os.path.join(self.root, base_dir) for base_dir in base_dirs]
        image_dirs = [os.path.join(voc_root, "JPEGImages") for voc_root in voc_roots]
        annotation_dirs = [os.path.join(voc_root, "Annotations") for voc_root in voc_roots]

        if not os.path.isdir(voc_roots[0]) or \
           not os.path.isdir(voc_roots[1]):
            raise RuntimeError("Dataset not found or corrupted.")

        splits_dirs = [os.path.join(voc_root, "ImageSets/Main") for voc_root in voc_roots]
        split_fs = [os.path.join(splits_dir, image_set.rstrip("\n")+".txt")
                    for splits_dir in splits_dirs]

        if not os.path.exists(split_fs[0]) or not os.path.exists(split_fs[1]):
            raise ValueError("Wrong image_set entered!")

        with open(os.path.join(split_fs[0]), "r") as f:
            file2007_names = [x.strip() for x in f.readlines()]

        with open(os.path.join(split_fs[1]), "r") as f:
            file2012_names = [x.strip() for x in f.readlines()]

        self.images = []
        self.annotations = []
        for x in file2007_names:
            self.images.append(os.path.join(image_dirs[0], x + ".jpg"))
            self.annotations.append(os.path.join(annotation_dirs[0], x + ".xml"))

        for x in file2012_names:
            self.images.append(os.path.join(image_dirs[1], x + ".jpg"))
            self.annotations.append(os.path.join(annotation_dirs[1], x + ".xml"))

        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(".."))

    root_path = os.path.abspath("..")
    val = VOCDetection(os.path.join(root_path, "data"), image_set="train")
    # print(len(val))

    import torchvision
    sets1 = torchvision.datasets.VOCDetection(os.path.join(root_path, "data"),
                                              image_set="train",
                                              year="2012")
    sets2 = torchvision.datasets.VOCDetection(os.path.join(root_path, "data"),
                                              image_set="train",
                                              year="2007")
    print(len(sets1) + len(sets2))


