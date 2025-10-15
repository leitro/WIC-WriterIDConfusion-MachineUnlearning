import torch.utils.data as D
from tqdm import tqdm
import cv2
import numpy as np
import random

WORD_LEVEL = True

if WORD_LEVEL:
    OUTPUT_MAX_LEN = 21
    IMG_WIDTH = 800
    base_dir = '/home/lkang/datasets/IAM'
    train_set = f'{base_dir}/RWTH_partition/RWTH.iam_word_gt_final.train.wid.azAZ'
    valid_set = f'{base_dir}/RWTH_partition/RWTH.iam_word_gt_final.valid.wid.azAZ'
    test_set = f'{base_dir}/RWTH_partition/RWTH.iam_word_gt_final.test.wid.azAZ'
    forget_set = f'{train_set}.forget'
    retain_set = f'{train_set}.retain'

IMG_HEIGHT = 64


labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'] # IAM sub
letter2index = {label: n for n, label in enumerate(labels)}
index2letter = {v: k for k, v in letter2index.items()}
num_classes = len(labels)
#tokens = {'CLS_TOKEN': 0, 'EOS_TOKEN': 1, 'PAD_TOKEN': 2}
tokens = {'EOS_TOKEN': 0, 'PAD_TOKEN': 1}
num_tokens = len(tokens.keys())
num_vocab = num_classes + num_tokens

from wids_trainset import train_wids
num_wids = len(train_wids)
wid2index = {wid: n for n, wid in enumerate(train_wids)}
index2wid = {v: k for k, v in wid2index.items()}


class IAM(D.Dataset):
    def __init__(self, file_label):
        self.file_label = file_label
        self.output_max_len = OUTPUT_MAX_LEN

    def __getitem__(self, index):
        word = self.file_label[index]
        wid, img, img_width = self.readImage_keepRatio(word[0])
        label, label_mask = self.label_padding(' '.join(word[1:]), num_tokens)
        return word[0], wid, img, img_width, label

    def __len__(self):
        return len(self.file_label)

    def readImage_keepRatio(self, file_name):
        wid, file_name = file_name.split(',')
        wid = int(wid)
        wid = wid2index[wid]
        url = f'{base_dir}/word_images_tot/{file_name}.png'
        img = cv2.imread(url, 0)

        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error
        img = img/255. # 0-255 -> 0-1
        img = 1. - img
        img_width = img.shape[-1]

        if img_width > IMG_WIDTH:
            outImg = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            #outImg = img[:, :IMG_WIDTH]
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
            outImg[:, :img_width] = img
        outImg = outImg.astype('float32')
        return wid, outImg, img_width

    def label_padding(self, labels, num_tokens):
        new_label_len = []
        ll = [letter2index[i] for i in labels]
        num = self.output_max_len - len(ll) - 2
        new_label_len.append(len(ll)+2)
        ll = np.array(ll) + num_tokens
        ll = list(ll)
        ll = ll + [tokens['EOS_TOKEN']]
        if not num == 0:
            ll.extend([tokens['PAD_TOKEN']] * num) # replace PAD_TOKEN

        def make_weights(seq_lens, output_max_len):
            new_out = []
            for i in seq_lens:
                ele = [1]*i + [0]*(output_max_len -i)
                new_out.append(ele)
            return new_out
        return ll, make_weights(new_label_len, self.output_max_len)

def loadData():
    with open(forget_set, 'r', encoding='utf8') as f_fo:
        data_fo = f_fo.readlines()
        file_label_forget = [i[:-1].split(' ') for i in data_fo]

    with open(retain_set, 'r', encoding='utf8') as f_re:
        data_re = f_re.readlines()
        file_label_retain = [i[:-1].split(' ') for i in data_re]

    data_forget = IAM(file_label_forget)
    data_retain = IAM(file_label_retain)
    return data_forget, data_retain

if __name__ == '__main__':
    pass

