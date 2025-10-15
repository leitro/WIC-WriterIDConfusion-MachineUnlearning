import torch
import os
from tqdm import tqdm
from jiwer import cer, wer
from torch.utils.data import DataLoader
from einops import rearrange
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel
from model import REC
from dataset import loadData, num_wids, num_vocab, IMG_WIDTH, tokens, num_tokens, index2letter

cuda = 'cuda'


BATCH_SIZE = 64
MAX_EPOCHS = 200
BETA = 1.
EARLY_STOP = 50
LEARNING_RATE = 4e-4
lr_milestone = [40, 80, 120, 160]
lr_gamma = 0.8


def collate_fn(batch):
    batch_size = len(batch)
    urls = []
    wids = []
    imgs = []
    img_widths = []
    labels = []
    for ii, (url, wid, img, img_width, label) in enumerate(batch):
        urls.append(url)
        wids.append(wid)
        imgs.append(img)
        img_widths.append(img_width)
        labels.append(label)

    wids = torch.tensor(wids)

    imgs_np = np.stack(imgs)
    imgs_tensor = torch.from_numpy(imgs_np)

    widths = torch.tensor(img_widths) // 4 + 1
    positions = torch.arange(IMG_WIDTH//4 + 1).unsqueeze(0).expand(batch_size, -1)
    masks = (positions < widths.unsqueeze(1)).float()

    labels_np = np.stack(labels)
    labels_tensor = torch.from_numpy(labels_np)
    return urls, wids, imgs_tensor, masks, labels_tensor


def idx2text(idxs):
    indices = idxs.tolist()
    letters = [index2letter[idx-num_tokens] for idx in indices if idx != tokens['PAD_TOKEN'] and idx != tokens['EOS_TOKEN']]
    text = ''.join(letters)
    return text


def cer_wer(preds, labels, tot_urls):
    tot_size = labels.size(0)
    tot_cer = 0.
    tot_wer = 0.
    for i in range(tot_size):
        pred_indices = preds[i]
        gt_indices = labels[i]
        url = tot_urls[i]

        mask_pred = (pred_indices != tokens['PAD_TOKEN']) & (pred_indices != tokens['EOS_TOKEN'])
        mask_gt = (gt_indices != tokens['PAD_TOKEN']) & (gt_indices != tokens['EOS_TOKEN'])
        pred_indices = pred_indices[mask_pred]
        gt_indices = gt_indices[mask_gt]

        pred_text = idx2text(pred_indices)
        gt_text = idx2text(gt_indices)

        tot_cer += cer(gt_text, pred_text)
        tot_wer += wer(gt_text, pred_text)

    return tot_cer/tot_size, tot_wer/tot_size


def train():
    train_data, valid_data, test_data = loadData()
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4, shuffle=True)
    model = REC(num_wids, num_vocab).to(cuda)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestone, gamma=lr_gamma)

    best_cer = 1e3
    best_epoch = 0
    max_num = 0
    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        lr = scheduler.get_last_lr()[0]
        cor_wid = 0
        tot_wid = 0
        text_pred = []
        text_gt = []
        tot_urls = []
        #for urls, wids, imgs, masks, labels in tqdm(train_dataloader):
        for urls, wids, imgs, masks, labels in train_dataloader:
            wids = wids.to(cuda)
            imgs = imgs.to(cuda)
            masks = masks.to(cuda)
            labels = labels.to(cuda)
            wid_loss, rec_loss, wid_pred, rec_pred = model(imgs, masks, wids, labels)
            loss = wid_loss + BETA * rec_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accuracy
            cor_wid += (wid_pred == wids).sum().item()
            tot_wid += wids.size(0)

            text_pred.append(rec_pred)
            text_gt.append(labels)
            tot_urls.extend(urls)

        scheduler.step()

        tot_rec_pred = torch.cat(text_pred, dim=0)
        tot_rec_gt = torch.cat(text_gt, dim=0)
        res_cer, res_wer = cer_wer(tot_rec_pred, tot_rec_gt, tot_urls)
        print(f'TRAIN Epoch-{epoch} wid_loss: {wid_loss:.3f}, rec_loss: {rec_loss:.3f}, wid_acc: {cor_wid/tot_wid*100:.2f}%, CER: {res_cer*100:.2f}%, WER: {res_wer*100:.2f}%, lr: {lr:.6f}')

        ## evaluation
        res_cer_t = eval(model)

        if res_cer_t < best_cer:
            best_cer = res_cer_t
            best_epoch = epoch
            max_num = 0
            save_model(model, epoch, prefix='base')
        else:
            max_num += 1
        if max_num >= EARLY_STOP:
            print(f'BEST CER_t: {res_cer_t*100:.2f}% at epoch {best_epoch}')
            break


def eval(model):
    _, valid_data, test_data = loadData()
    valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    model.eval()
    cor_wid = 0
    tot_wid = 0
    text_pred = []
    text_gt = []
    tot_urls = []
    for urls, wids, imgs, masks, labels in valid_dataloader:
        wids = wids.to(cuda)
        imgs = imgs.to(cuda)
        masks = masks.to(cuda)
        labels = labels.to(cuda)
        with torch.no_grad():
            wid_loss, rec_loss, _, rec_pred = model(imgs, masks, wids, labels)

        # accuracy
        text_pred.append(rec_pred)
        text_gt.append(labels)
        tot_urls.extend(urls)

    tot_rec_pred = torch.cat(text_pred, dim=0)
    tot_rec_gt = torch.cat(text_gt, dim=0)
    res_cer, res_wer = cer_wer(tot_rec_pred, tot_rec_gt, tot_urls)
    print(f'    [VALID] rec_loss_t: {rec_loss:.4f}, CER_t: {res_cer*100:.2f}%, WER_t: {res_wer*100:.2f}%')
    return res_cer


def save_model(model, epoch, prefix=''):
    if not os.path.exists('weights'):
        os.makedirs('weights')
    name = f'weights/{prefix}-t5-htr-best.model'
    torch.save(model.state_dict(), name)
    

if __name__ == '__main__':
    train()
