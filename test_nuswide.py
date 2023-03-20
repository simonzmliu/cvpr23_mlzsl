from collections import OrderedDict

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from sklearn.preprocessing import normalize
import torch.nn.functional as F
import numpy as np
import random
import argparse
import time
import random
import os
import csv
import tqdm
import warnings
from models.Channel_MLZSL import build_GroupChannel
from config import opt
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import json
import util
import pickle
import pandas as pd


torch.multiprocessing.set_sharing_strategy('file_system')

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

dim_feature = [196, opt.nclass_all]
dim_w2v = 300

# with open(os.path.join(opt.dataset_path, "unseen_classes.pickle"), 'rb') as fp:
#     unseen_classes = pickle.load(fp)

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
        labs1k = np.array(self.json_dict[str(index)]['labels_1006'])
        labs81 = np.array(self.json_dict[str(index)]['labels_81'])
        if self.transform:
            image = self.transform(image)
        if self.train:
            # print(labs1k.shape, labs81.shape)
            return image_id, image, labs1k, labs81
        else:
            return image_id, image, labs1k, labs81

    def __len__(self):
        return len(self.json_dict)





def test():
    net = build_GroupChannel(224, name='vgg19', state='test')
    if opt.cuda:
        net = net.cuda()
        cudnn.benchmark = True
        print(net)

    if opt.multigpu:
        net_dict = torch.load(opt.trained_model)
        new_net_dict = OrderedDict()
        for k, v in net_dict.items():
            name = k[7:]
            new_net_dict[name] = v
        net.load_state_dict(new_net_dict)
    else:
        net.load_state_dict(torch.load(opt.trained_model))

    net.eval()

    print('Finished loading model!')

    src = "data/NUS_WIDE"
    file_tag1k = os.path.join(src, 'NUS_WID_Tags', 'TagList1k.txt')
    file_tag81 = os.path.join(src, 'Concepts81.txt')
    seen_cls_idx, unseen_cls_idx, tag1k, tag81 = util.get_seen_unseen_classes(file_tag1k, file_tag81)
    test_imagelist = 'data/NUS_WIDE/ImageList/TestImagelist.txt'
    att_path = os.path.join(src, 'word_embedding', 'NUS_WIDE_pretrained_w2v_glove-wiki-gigaword-300')
    src_att = pickle.load(open(att_path, 'rb'))
    print("attributes are combined in this order-> seen+unseen", 'r')
    unseen_vecs = src_att[1]
    all_vecs = torch.from_numpy(normalize(np.concatenate((src_att[0][seen_cls_idx], src_att[1]), axis=0)))

    classes = {}
    unseen_classes = tag81.tolist()
    classes_array = np.concatenate((tag1k[seen_cls_idx], tag81), axis=0)
    for i in range(len(classes_array)):
        classes[i] = classes_array[i]

    nus_test = get_extract_data(test_imagelist,
                                     dir_=opt.image_dir,
                                     json_file=os.path.join(opt.output_dir, opt.test_json + '2.json'),
                                     transform=transforms.Compose([transforms.Resize([224, 224]),
                                                                #    transforms.RandomHorizontalFlip(),
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                        (0.2023, 0.1994, 0.2010))]),
                                                                                        train=False)

    data_loader = data.DataLoader(nus_test, opt.batch_size * 2, num_workers=12)

    num_images = len(data_loader)
    predictions = []
    predictions_gzsl = []
    labels81 = []
    labels1k = []
    with torch.no_grad():
        # for i in range(len(nus_test)):
        for i, (image_id, images, labs1k, labs81) in enumerate(data_loader):
            # image_id, images, labs1k, labs81 = nus_test.__getitem__(i)
            # images = Variable(images.unsqueeze(0))
            # if i > 10:
            #     break
            # load train data
            if opt.cuda:
                images = images.cuda()
            else:
                images = images
                labs81 = labs81
                labs1k = labs1k

            # forward
            out = net(images)

            b, g, c = out.shape

            for z in range(b):
                np_output = out[z].cpu().detach().numpy()
                x = np_output.reshape(opt.wordvec_dim, opt.num_rows)
                logit = np.dot(x.T, unseen_vecs.T)
                logit_gzsl = np.dot(x.T, all_vecs.T)
                logit = np.max(logit, axis=0)
                logit_gzsl = np.max(logit_gzsl, axis=0)
                predictions.append(logit[np.newaxis, :])
                predictions_gzsl.append(logit_gzsl[np.newaxis, :])
                labels81.append(labs81[z][np.newaxis, :])
                labels1k.append(labs1k[z][np.newaxis, :])
            print('im_detect: {:d}/{:d}'.format(i + 1, int(num_images)), end='\r')

    predictions = np.concatenate(predictions)
    predictions_gzsl = np.concatenate(predictions_gzsl)
    labels81 = np.concatenate(labels81)
    labels1k = np.concatenate(labels1k)


    # idxs, dists = get_knns(labels81, predictions)
    # precision_3, recall_3, F1_3 = calc_F1(labels1k.numpy(), idxs, 3, relevant_inds=unseen_cls_idx,
    #                                       num_classes=len(labels81))
    # print("Top-{}: precision {:.2f}, recall {:.2f}, F1 {:.2f}".format(3, precision_3, recall_3, F1_3))


    # zsl
    F1_3_tst, P_3_tst, R_3_tst = util.evaluate_k(3, predictions, labels81)
    F1_5_tst, P_5_tst, R_5_tst = util.evaluate_k(5, predictions, labels81)
    ap_tst, _, _ = util.evaluate(predictions, labels81)

    print('mAP', np.mean(ap_tst))
    print('k=3', np.mean(F1_3_tst), np.mean(P_3_tst), np.mean(R_3_tst))
    print('k=5', np.mean(F1_5_tst), np.mean(P_5_tst), np.mean(R_5_tst))
    df = pd.DataFrame()
    df['classes'] = tag81
    df['F1_3'] = F1_3_tst
    df['P_3'] = P_3_tst
    df['R_3'] = R_3_tst

    df['F1_5'] = F1_5_tst
    df['P_5'] = P_5_tst
    df['R_5'] = R_5_tst

    df['ap'] = ap_tst

    df.to_csv(opt.save_path + 'F1_f_test_tresnet_zsl_vgg19.csv')

    # gzsl
    F1_3_tst, P_3_tst, R_3_tst = util.evaluate_k(3, predictions_gzsl, labels1k)
    F1_5_tst, P_5_tst, R_5_tst = util.evaluate_k(5, predictions_gzsl, labels1k)
    ap_tst, _, _ = util.evaluate(predictions_gzsl, labels1k)

    print('mAP', np.mean(ap_tst))
    print('k=3', np.mean(F1_3_tst), np.mean(P_3_tst), np.mean(R_3_tst))
    print('k=5', np.mean(F1_5_tst), np.mean(P_5_tst), np.mean(R_5_tst))
    df1 = pd.DataFrame()
    df1['F1_3'] = F1_3_tst
    df1['P_3'] = P_3_tst
    df1['R_3'] = R_3_tst

    df1['F1_5'] = F1_5_tst
    df1['P_5'] = P_5_tst
    df1['R_5'] = R_5_tst

    df1['ap'] = ap_tst

    df1.to_csv(opt.save_path + 'F1_f_test_ChannelAtt_gzsl_vgg19.csv')


if __name__ == '__main__':
    test()
