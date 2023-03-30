import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
import json
from PIL import Image
import cv2
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
import os

class get_extract_data(data.Dataset):
    def __init__(self, imagelist, dir_, json_file, transform, train=True):
        with open(json_file, 'r') as load_f:
            self.json_dict = json.load(load_f)
        self.imagelist = imagelist
        self.dir_ = dir_
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        image_id = os.path.join(str(self.dir_) + '/', str(self.json_dict[str(index)]['img_id']))
        image = Image.open(image_id)
        image = image.convert('RGB')
        if self.train:
            labs925 = np.array(self.json_dict[str(index)]['labels_925'])
            labs1k = np.array(self.json_dict[str(index)]['labels_1006'])
            labs81 = np.array(self.json_dict[str(index)]['labels_81'])
            if self.transform:
                image = self.transform(image)
            return image_id, image, labs1k, labs925, labs81
        else:
            labs1k = np.array(self.json_dict[str(index)]['labels_1006'])
            labs81 = np.array(self.json_dict[str(index)]['labels_81'])
            if self.transform:
                image = self.transform(image)
            return image_id, image, labs1k, labs81

    def __len__(self):
        return len(self.json_dict)


def get_seen_unseen_classes(file_tag1k, file_tag81):
    with open(file_tag1k, "r") as file:
        tag1k = np.array(file.read().splitlines())
    with open(file_tag81, "r") as file:
        tag81 = np.array(file.read().splitlines())
    seen_cls_idx = np.array(
        [i for i in range(len(tag1k)) if tag1k[i] not in tag81])
    unseen_cls_idx = np.array(
        [i for i in range(len(tag1k)) if tag1k[i] in tag81])
    return seen_cls_idx, unseen_cls_idx, tag1k, tag81


def compute_AP(predictions, labels):
    num_class = predictions.shape[1]
    ap = np.zeros(num_class)
    for idx_cls in range(num_class):
        prediction = np.squeeze(predictions[:, idx_cls])
        label = np.squeeze(labels[:, idx_cls])
        mask = np.abs(label) == 1
        if np.sum(label > 0) == 0:
            continue
        binary_label = np.clip(label[mask], 0, 1)
        ap[idx_cls] = average_precision_score(binary_label, prediction[mask])  # AP(prediction,label,names)
    return ap


def compute_F1(pre, lab, mode_F1):
    if mode_F1 == 'overall':
        print('evaluation overall!! cannot decompose into classes F1 score')
        mask = pre == 1
        TP = np.sum(lab[mask] == 1)
        p = TP / np.sum(mask)
        r = TP / np.sum(lab == 1)
        f1 = 2 * p * r / (p + r)

    #        p_2,r_2,f1_2=compute_F1_fast0tag(predictions,labels)
    else:
        num_class = pre.shape[1]
        print('evaluation per classes')
        f1 = np.zeros(num_class)
        p = np.zeros(num_class)
        r = np.zeros(num_class)
        for idx_cls in range(num_class):
            prediction = np.squeeze(pre[:, idx_cls])
            label = np.squeeze(lab[:, idx_cls])
            if np.sum(label > 0) == 0:
                continue
            binary_label = np.clip(label, 0, 1)
            f1[idx_cls] = f1_score(binary_label, prediction)  # AP(prediction,label,names)
            p[idx_cls] = precision_score(binary_label, prediction)
            r[idx_cls] = recall_score(binary_label, prediction)
    return f1, p, r


def evaluate(predictions, labels, is_exclude=False):
    pre = predictions.copy()
    lab = labels.copy()
    if is_exclude:
        mask = np.sum(labels == 1, 1) > 0

        print("Total test samples: {} Total samples with positive labels: {}".format(pre.shape[0],
                                                                                     np.sum(mask)))

        pre = pre[mask]
        lab = lab[mask]
    else:
        print('no exclude')

    assert pre.shape == lab.shape, 'invalid shape'

    return compute_AP(pre, lab), pre, lab


def evaluate_k(k, predictions, labels, is_exclude=False, mode_F1='overall'):
    pre = predictions.copy()
    lab = labels.copy()
    if is_exclude:
        mask = np.sum(lab == 1, 1) > 0
        print("Total test samples: {} Total samples with positive labels: {}".format(pre.shape[0], np.sum(mask)))
        pre = pre[mask]
        lab = lab[mask]
    else:
        print('no exclude')

    idx = np.argsort(pre, axis=1)
    for i in range(pre.shape[0]):
        pre[i][idx[i][-k:]] = 1
        pre[i][idx[i][:-k]] = 0
        # pre[i][idx[i][:k]] = 1
        # pre[i][idx[i][k:]] = 0

    assert pre.shape == lab.shape, 'invalid shape'

    return compute_F1(pre, lab, mode_F1)

