#-*-coding:utf-8-*-

import numpy as np


def iou(box, clusters):
    x = np.minimum(box[0], clusters[:, 0])
    y = np.minimum(box[1], clusters[:, 1])
    if np.count_nonzero(x) > 0 or np.count_nonzero(y) > 0:
        raise ValueError("Box has no area")

    intersection = x*y
    box_area = box[0]*box[1]
    cluster_area = clusters[:, 0]*clusters[:, 1]

    i = intersection/(box_area+cluster_area-intersection)
    return i


def handle_boxes(boxes):
    """
    (x1, y1, x2, y2) -> (width, height)
    :param boxes:
    :return: (number, 2)
    """
    values = np.zeros((boxes.shape[0], 2))
    values[:, 0] = np.abs(boxes[:, 2] - boxes[:, 0])
    values[:, 1] = np.abs(boxes[:, 3] - boxes[:, 1])
    return values


def kmeans(boxes, k, dist=np.median):
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows, ))

    np.random.seed()

    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            try:
                distances[row] = 1 - iou(boxes[row], clusters)
            except ValueError:
                distances[row] = 1

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


if __name__ == "__main__":
    import sys
    import torchvision
    import os

    from datasets.voc import handle_voc_target, VOCDetection
    sys.path.append(os.path.abspath(".."))
    root_path = os.path.abspath("..")
    sets = VOCDetection(os.path.join(root_path, "data"), image_set="train",
                        target_transform=handle_voc_target)
    boxes = []
    for _, target in sets:
        boxes.append(target[:, 1:5])
    boxes = np.concatenate(boxes, 0)
    boxes = handle_boxes(boxes)
    out = kmeans(boxes, k=9)
    print("Boxes: \n {}".format(out))

    ratios = np.around(out[:, 0]/out[:, 1], decimals=3).tolist()
    print("Ratios: \n {}".format(sorted(ratios)))
