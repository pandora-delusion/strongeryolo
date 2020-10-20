#-*-coding:utf-8-*-

import torch.utils.data as tdata
import os
from PIL import Image
import numpy as np
import pandas as pd
from config import cfg
import json


def read_descriptions():
    data = pd.read_csv(os.path.join(cfg.FILE.BASE_PATH,
                                    cfg.FILE.CLASS_DESCRIPTION))
    class2Name = {}
    class2Name["/m/061hd_"] = "Infant bed"
    for _, row in data.iterrows():
        class2Name[row["/m/061hd_"]] = row["Infant bed"]
    return class2Name


def load_class(classes):
    class2idx = dict(((cls, idx) for idx, cls in enumerate(classes)))
    return class2idx


def analyse_hierarchy_old(json_dir):
    json_file = open(os.path.join(cfg.FILE.BASE_PATH, json_dir))
    json_dict = json.load(json_file)
    labelName = cfg.BBOX.CSV_LABELNAME
    subcategory = "Subcategory"
    classes = [json_dict[labelName]]

    def _handle_list(json_list):
        for json_dict in json_list:
            classes.append(json_dict[labelName])

        for js_dict in json_list:
            sub = js_dict.get(subcategory, None)
            if sub:
                _handle_list(sub)

    _handle_list(json_dict[subcategory])
    return classes


def old_idx_class_map():
    classes = analyse_hierarchy_old(cfg.FILE.CLASS_HIERARCHY_FILE)
    class2idx = load_class(classes)
    return class2idx


def old2new(new):
    o2n = {}
    old = old_idx_class_map()
    for cls in new:
        o2n[old[cls]] = new[cls]
    return o2n


def analyse_hierarchy(json_dir, class2idx):
    json_file = open(os.path.join(cfg.FILE.BASE_PATH, json_dir))
    json_dict = json.load(json_file)
    lableName = cfg.BBOX.CSV_LABELNAME
    subcategory = "Subcategory"
    hierarchy = {}

    stack = []

    def _analyse_hierarchy(json_dict):
        cls = json_dict[lableName]
        if subcategory in json_dict:
            if cls in class2idx:
                stack.append(cls)
            for json_obj in json_dict[subcategory]:
                _analyse_hierarchy(json_obj)

            if cls in class2idx:
                stack.pop(-1)

        if cls in class2idx:
            temp = []
            temp.extend(stack)
            if cls in hierarchy:
                hierarchy[cls].extend(temp)
            else:
                hierarchy[cls] = temp
    _analyse_hierarchy(json_dict)
    return hierarchy


def get_multi_labels_hierarchy_and_classes():
    class2idx, class2Name = handle_class()
    hi = analyse_hierarchy(cfg.FILE.CLASS_HIERARCHY_FILE, class2idx)
    class2idx = dict(((key, idx) for idx, key in enumerate(hi.keys())))
    idx2class = dict(((idx, key) for idx, key in enumerate(hi.keys())))
    hidx = {}
    for key in hi:
        hidx[class2idx[key]] = [class2idx[x] for x in hi[key]]
    return class2idx, idx2class, class2Name, hidx


def handle_class():
    class2Name = read_descriptions()
    classes = class2Name.keys()
    class2idx = load_class(classes)
    return class2idx, class2Name


def multi_label_handler(annotations, hierarchy):
    size = annotations.shape[0]
    labels = []
    for i in range(size):
        temp = int(annotations[i, 0])
        label = [temp]
        label.extend(hierarchy[temp])
        label = np.array(label, dtype=int)
        labels.append(label)
    return labels


class TestLoader(tdata.Dataset):

    @staticmethod
    def load_image_item(file):
        return Image.open(file).convert("RGB")

    def __init__(self, root, transform=None):
        super(TestLoader, self).__init__()

        self.root = root
        self.images_dir = os.path.join(root, "test")
        self.transform = transform

        if not os.path.exists(self.images_dir):
            raise OSError("...")

        count = 0
        with os.scandir(self.images_dir) as scanner:
            for _ in scanner:
                count += 1
        numbers = count

        self.numbers = numbers

    def __len__(self):
        return self.numbers

    def search_file(self, item):
        select_file = None
        with os.scandir(self.images_dir) as scanner:
            for idx, entry in enumerate(scanner):
                if idx == item:
                    select_file = entry.name
        if not select_file:
            raise RuntimeError("search image failed!")
        return select_file.replace(".jpg", "")

    def load_data(self, idx):
        file_info = self.search_file(idx)
        image_file = os.path.join(self.images_dir, "{}.jpg".format(file_info))
        image = self.load_image_item(image_file)
        return file_info, image

    def __getitem__(self, item):
        file_info, image = self.load_data(item)
        info = None

        if self.transform is not None:
            image, info = self.transform(image, None)

        return file_info, image, info


class ImageLoader(tdata.Dataset):

    def get_length(self):
        raise NotImplementedError

    def __init__(self,
                 root,
                 dataset="train",
                 transforms=None,
                 help_file=None):
        super(ImageLoader, self).__init__()

        self.root = root
        self.dataset = dataset
        self.transforms = transforms

        self.images_dir = os.path.join(root, dataset)
        self.ann_dir = os.path.join(root, "labels/{}".format(dataset))
        if not os.path.exists(self.root) or not os.path.exists(self.ann_dir):
            raise OSError("the folder {} or {} is not exist!".format(self.images_dir,
                                                                     self.ann_dir))
        self.help_file = help_file

        if not self.help_file:
            count = 0
            with os.scandir(self.ann_dir) as scanner:
                for _ in scanner:
                    count += 1
            numbers = count
        else:
            numbers = self.get_length()
        self.numbers = numbers

        self.class2idx, self.idx2class, self.class2name, self.label_hierarchy = self.load_classes()

    def judge_similar(self, a, b):
        """
        返回顺序：父 子
        :param a:
        :param b:
        :return:
        """
        a_h = self.label_hierarchy[a]
        b_h = self.label_hierarchy[b]
        if b in a_h:
            return b, a
        if a in b_h:
            return a, b
        return -1, -1

    def __len__(self):
        return self.numbers

    def __getitem__(self, item):
        image, annotation = self.load_data(item)

        if self.transforms is not None:
            image, annotation = self.transforms(image, annotation)

        return image, annotation

    def search_file(self, item):
        select_file = None
        with os.scandir(self.ann_dir) as scanner:
            for idx, entry in enumerate(scanner):
                if idx == item:
                    select_file = entry.name
        if not select_file:
            raise RuntimeError("search image failed!")
        return select_file.replace(".txt", "")

    def handle_multi_label(self, anns):
        return multi_label_handler(anns, self.label_hierarchy)

    @staticmethod
    def load_image_item(file):
        return Image.open(file).convert("RGB")

    @staticmethod
    def load_annotations_item(file):
        return np.loadtxt(file, ndmin=2, dtype=np.float32)

    def load_classes(self):
        cls2idx, idx2cls, cls2Name, hi = get_multi_labels_hierarchy_and_classes()
        return cls2idx, idx2cls, cls2Name, hi

    @staticmethod
    def handle_annotations(image, annotations):
        width, height = image.size
        annotations[:, 1:3] *= width
        annotations[:, 3:] *= height

        # [x, x1, x2, y1, y2] -> [x, x1, y1, x2, y2]
        bbox = np.zeros((annotations.shape[0], 5), dtype=np.float32)
        bbox[:, 1] += annotations[:, 1]
        bbox[:, 4] += annotations[:, 4]
        bbox[:, 2] += annotations[:, 3]
        bbox[:, 3] += annotations[:, 2]
        # bbox[:, 1::3] += annotations[:, 1::3]
        # bbox[:, 2:4] += annotations[:, 3:1:-1]
        bbox[:, 0] += annotations[:, 0]
        return bbox

    def load_data(self, idx):
        file_info = self.search_file(idx)
        image_file = os.path.join(self.images_dir, "{}.jpg".format(file_info))
        ann_file = os.path.join(self.ann_dir, "{}.txt".format(file_info))
        image = self.load_image_item(image_file)
        annotations = self.load_annotations_item(ann_file)[:, 0:5]
        annotations = self.handle_annotations(image, annotations)
        return image, annotations

    def get_label(self, idx):
        return self.class2name[self.idx2class[idx]]


class TrainLoader(ImageLoader):
    """
    因为在处理oid数据集时因为处理过于庞大的数据集使得oid训练集被分割为多个子数据集，为了
    解决由此带来的图片标注加载和由于处理类别层次结构而犯下的错误，我不得不使用该类来加载
    训练集数据。而验证集和测试集并不存在上述问题，请仍然使用父类ImageLoader。在未来，
    如果有时间，请重新生成训练集标注文件。
    """
    def __init__(self,
                 root,
                 transforms=None,
                 help_file=None):
        self._idx_dict = {}
        super(TrainLoader, self).__init__(root, "train", transforms, help_file)

    def __len__(self):
        return self.numbers

    def get_length(self):
        file = open(self.help_file)
        lines = file.readlines()

        acc = 0
        for line in lines:
            idx, length = line.split(" ")
            length.replace("\\n", "")
            length = int(length)
            self._idx_dict[acc] = idx
            acc += length
        file.close()
        return acc

    def search_file(self, item):
        idxs = list(self._idx_dict.keys())
        # print(self._idx_dict)
        idxs.sort(reverse=True)
        select_key = None
        relative_idx = 0
        i = None
        for i in idxs:
            if item >= i:
                select_key = self._idx_dict[i]
                relative_idx = item - i
                break
        images_dir = "train_{}".format(select_key)
        select_file = None
        with os.scandir(os.path.join(self.ann_dir, images_dir)) as scanner:
            for idx, entry in enumerate(scanner):
                if idx == relative_idx:
                    select_file = entry.name
                    break
        if not select_file:
            raise RuntimeError("search image failed, can not find {} group files in"
                               " file {}".format(item, self._idx_dict[i]))
        select_file = "{}/{}".format(images_dir, select_file.replace(".txt", ""))
        return select_file

    def fix_label_error(self, annotations):
        for i in range(annotations.shape[0]):
            annotations[i, 0] = self.o2n[int(annotations[i, 0])]

    def load_data(self, idx):
        image, anns = super(TrainLoader, self).load_data(idx)
        self.fix_label_error(anns)
        # labels_hierarchy = multi_label_handler(anns, self.label_hierarchy)
        return image, anns

    def load_classes(self):
        cls2idx, idx2cls, cls2Name, hi = super(TrainLoader, self).load_classes()
        self.o2n = old2new(cls2idx)

        return cls2idx, idx2cls, cls2Name, hi


if __name__ == "__main__":

    from config import cfg
    from utils.visualization import plt_bboxes
    root_path = cfg.FILE.BASE_PATH
    trains = TrainLoader(os.path.join(root_path, "data/oid"),
                         help_file=os.path.join(root_path, "data/train_info.txt"))
    vals = ImageLoader(os.path.join(root_path, "data/oid"), "validation")
    print(vals.idx2class)
    print(trains.class2name[trains.idx2class[47]])
    # for i in range(20):
    #     image, anns = trains[i]
    #     plt_bboxes(image, anns, trains)
