from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits
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


# plot the visualization of the feature map
def load_data(data_path):
    return np.load(data_path, allow_pickle=True).item()


def tSNE(datas, name):
    color_dict = {'blue': 6, 'green': 15, 'yellow': 28, 'red': 44, 'black': 52, 'gray': 60, 'gold': 67, 'pink': 71,
                  'purple': 76, 'orange': 80}

    data = np.array(datas['data'])
    target = np.array(datas['target'])
    tsne = TSNE(n_components=2, n_iter=500)
    data_tsne = tsne.fit_transform(data)
    x, y = data_tsne[:, 0], data_tsne[:, 1]
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)

    # current_axes = plt.axes()
    # current_axes.xaxis.set_visible(False)
    # current_axes.yaxis.set_visible(False)

    color_target = [list(color_dict.keys())[list(color_dict.values()).index(c)] for c in target]
    plt.scatter(x, y, c=color_target, s=10)
    plt.savefig(name, dpi=1000)
    plt.show()



def visualize_grid_attention_v2(img_path, save_path, attention_mask, ratio=1, cmap="jet", save_image=False,
                             save_original_image=False, quality=200):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    print("load image from: ", img_path)
    
    
    
    
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    # plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    # plt.imshow(img, alpha=1)
    # plt.axis('off')

    # normalize the attention map

    normed_mask = np.maximum(attention_mask, 0)
    # normed_mask = cv2.resize(normed_mask, (img_h, img_w))
    normed_mask = normed_mask / normed_mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')


    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)

    if save_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
        img_with_attention_save_path = os.path.join(save_path, img_name)
        
        # pre-process and save image
        print("save image to: " + save_path + " as " + img_name)
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_with_attention_save_path, dpi=quality)

    if save_original_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # save original image file
        print("save original image at the same time")
        img_name = img_path.split('/')[-1].split('.')[0] + "_original.jpg"
        original_image_save_path = os.path.join(save_path, img_name)
        img.save(original_image_save_path, quality=quality)



if __name__ == "__main__":
    # iris = load_iris()
    # digits = load_digits()
    # plt.scatter(np.array([0, 1, 3]), np.array([1, 2, 2]), c=['green', 'blue', 'gold'], s=10)
    # plt.savefig("test.png", dpi=1000)
    # plt.show()
    # tSNE(digits)
    # tsne_vgg = load_data("tsne_vgg.npy")
    # tsne_last = load_data("tsne_pred.npy")
    # tSNE(tsne_vgg, "tsne-vgg")
    # tSNE(tsne_last, "tsne-last")

    img_path="./example.jpg"
    save_path="test"
    attention_mask = np.random.randn(7,300)
    visualize_grid_attention_v2(img_path,
                            save_path=save_path,
                            attention_mask=attention_mask,
                            save_image=True,
                            save_original_image=True,
                            quality=100)

