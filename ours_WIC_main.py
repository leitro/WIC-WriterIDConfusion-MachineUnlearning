import torch
import time
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
from dataset_unlearn import loadData, num_wids, num_vocab, IMG_WIDTH, tokens, num_tokens, index2letter
from dataset import loadData as loadDataOri

cuda = 'cuda'

BATCH_SIZE = 32
BETA = 1.


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


def check_pruning(model, prefix):  
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name: 
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    sparsity = zero_params / total_params
    print(f"[{prefix}] Model sparsity: {sparsity:.2%}")


def train(weight_url, lr, LAMBDA_WIC=0.5):
    forget_data, retain_data = loadData()
    retain_dataloader = DataLoader(
        retain_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4, shuffle=True
    )
    forget_dataloader = DataLoader(
        forget_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4, shuffle=True
    )
    model = REC(num_wids, num_vocab).to(cuda)
    checkpoint = torch.load(f'weights/{weight_url}')
    model.load_state_dict(checkpoint)
    print(f'Loading weights {weight_url} ... Done!')
    check_pruning(model, 'before')
    #eval(model)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9
    )
    #LAMBDA_WIC = 0.5  # How hard to enforce unlearning. Tune 0.2-1.0 if desired

    cor_wid = 0
    tot_wid = 0
    text_pred = []
    text_gt = []
    tot_urls = []

    for iiter in range(1, 10000+1):
        model.train()
        retain_iter = iter(retain_dataloader)
        forget_iter = iter(forget_dataloader)

        try:
            urls_r, wids_r, imgs_r, masks_r, labels_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_dataloader)
            urls_r, wids_r, imgs_r, masks_r, labels_r = next(retain_iter)
        try:
            urls_f, wids_f, imgs_f, masks_f, labels_f = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_dataloader)
            urls_f, wids_f, imgs_f, masks_f, labels_f = next(forget_iter)

        # Concatenate all
        urls = urls_r + urls_f
        wids = torch.cat([wids_r, wids_f], dim=0)
        imgs = torch.cat([imgs_r, imgs_f], dim=0)
        masks = torch.cat([masks_r, masks_f], dim=0)
        labels = torch.cat([labels_r, labels_f], dim=0)

        # Build mask: True for forget samples, False for retain
        is_forget = torch.zeros_like(wids, dtype=torch.bool)
        is_forget[wids_r.size(0):] = True
        is_retain = ~is_forget

        wids = wids.to(cuda)
        imgs = imgs.to(cuda)
        masks = masks.to(cuda)
        labels = labels.to(cuda)

        # Forward pass (must support return_logits)
        wid_loss, rec_loss, wid_pred, rec_pred, wid_logits, ocr_logits = model(
            imgs, masks, wids, labels, return_loss=True, return_logits=True
        )

        # (1) WriterID-CE on retain set only
        if is_retain.any():
            ce_retain = F.cross_entropy(
                wid_logits[is_retain], wids[is_retain]
            )
        else:
            ce_retain = torch.tensor(0., device=wid_logits.device)

        # (2) KL-to-uniform on forget set only (WIC loss)
        if is_forget.any():
            log_probs_f = F.log_softmax(wid_logits[is_forget], dim=1)
            n_classes = wid_logits.size(1)
            uniform = torch.full_like(log_probs_f, 1.0 / n_classes)
            kl_forget = F.kl_div(log_probs_f, uniform, reduction='batchmean')
        else:
            kl_forget = torch.tensor(0., device=wid_logits.device)

        # (3) Sequence recognition loss as usual
        # Already computed: rec_loss

        # (4) Total loss
        loss = ce_retain + LAMBDA_WIC * kl_forget + BETA * rec_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy etc. for logging
        cor_wid += (wid_pred == wids).sum().item()
        tot_wid += wids.size(0)
        text_pred.append(rec_pred)
        text_gt.append(labels)
        tot_urls.extend(urls)

        if iiter % 1000 == 0:
            tot_rec_pred = torch.cat(text_pred, dim=0)
            tot_rec_gt = torch.cat(text_gt, dim=0)
            res_cer, res_wer = cer_wer(tot_rec_pred, tot_rec_gt, tot_urls)
            print(
                f'TRAIN Iter-{iiter} wid_loss: {wid_loss:.3f}, rec_loss: {rec_loss:.3f}, wid_acc: {cor_wid / tot_wid * 100:.2f}%, CER: {res_cer * 100:.2f}%, WER: {res_wer * 100:.2f}%, lr: {lr:.6f}'
            )
            cor_wid = 0
            tot_wid = 0
            text_pred = []
            text_gt = []
            tot_urls = []

            save_model(model, prefix=f'WIC_iter-{iiter}_{LAMBDA_WIC}') # writerID Confusion loss
            check_pruning(model, f'Iter-{iiter}')
            res_cer_t = eval(model)


def eval(model):
    forget_data, retain_data = loadData()
    _, _, test_data = loadDataOri()
    forget_dataloader = DataLoader(forget_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    retain_dataloader = DataLoader(retain_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    cer_forget = _eval_func(model, forget_dataloader, 'forget')
    cer_retain = _eval_func(model, retain_dataloader, 'retain')
    cer_test = _eval_func(model, test_dataloader, 'test')
    return cer_test


def _eval_func(model, dataloader, prefix):
    s_time = time.time()
    model.eval()
    cor_wid = 0
    tot_wid = 0
    text_pred = []
    text_gt = []
    tot_urls = []
    for urls, wids, imgs, masks, labels in dataloader:
        wids = wids.to(cuda)
        imgs = imgs.to(cuda)
        masks = masks.to(cuda)
        labels = labels.to(cuda)
        with torch.no_grad():
            wid_pred, rec_pred = model(imgs, masks, wids, labels, return_loss=False)

        # accuracy
        cor_wid += (wid_pred == wids).sum().item()
        tot_wid += wids.size(0)
        text_pred.append(rec_pred)
        text_gt.append(labels)
        tot_urls.extend(urls)

    tot_rec_pred = torch.cat(text_pred, dim=0)
    tot_rec_gt = torch.cat(text_gt, dim=0)
    res_cer, res_wer = cer_wer(tot_rec_pred, tot_rec_gt, tot_urls)
    if prefix == 'test':
        print(f'[{prefix}] CER: {res_cer*100:.2f}%, WER: {res_wer*100:.2f}%, time: {time.time()-s_time:.1f}s')
    else:
        print(f'[{prefix}] wid_acc: {cor_wid/tot_wid*100:.2f}%, CER: {res_cer*100:.2f}%, WER: {res_wer*100:.2f}%, time: {time.time()-s_time:.1f}s')
    return res_cer


def save_model(model, prefix=''):
    if not os.path.exists('weights'):
        os.makedirs('weights')
    name = f'weights/{prefix}-t5-htr.model'
    torch.save(model.state_dict(), name)
    

if __name__ == '__main__':
    for lamb in [0.2, 0.4, 0.6, 0.8, 1]:
        print(f'------------------LAMBDA_WIC: {lamb}------------------')
        train('unlearn-pruned_fine-grained-prune_embed-0.4_low-0.2_mid-0.4_high-0.2_ff-0.2.model', lr=2e-4, LAMBDA_WIC=lamb)
