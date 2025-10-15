import torch
import time
import os
from tqdm import tqdm
from jiwer import cer, wer
import torch.nn.utils.prune as prune
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

BATCH_SIZE = 512
prob_prune = {'embedding': 0.4, 't5_attn_low': 0.2, 't5_attn_mid': 0.4, 't5_attn_high': 0.2, 't5_ff': 0.2}
prefix_prune = f"embed-{prob_prune['embedding']:.1f}_low-{prob_prune['t5_attn_low']:.1f}_mid-{prob_prune['t5_attn_mid']:.1f}_high-{prob_prune['t5_attn_high']:.1f}_ff-{prob_prune['t5_ff']:.1f}"


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


def prune_model(weight_url, epsilon=1e-8):
    forget_data, retain_data = loadData()
    forget_dataloader = DataLoader(forget_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    retain_dataloader = DataLoader(retain_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    model = REC(num_wids, num_vocab).to(cuda)
    checkpoint = torch.load(f'weights/{weight_url}')
    model.load_state_dict(checkpoint)
    print(f'Loading weights {weight_url} ... Done!')
    
    check_pruning(model, 'before')
    #eval(model) 

    layers_to_monitor = {
        'embedding': model.embedding,
        ##'writer_classifier': model.writer_classifier,
        ##'text_classifier': model.text_classifier,
    }

    for i, block in enumerate(model.t5_encoder.encoder.block):
        layers_to_monitor[f't5_attn_{i}_q'] = block.layer[0].SelfAttention.q
        layers_to_monitor[f't5_attn_{i}_k'] = block.layer[0].SelfAttention.k
        layers_to_monitor[f't5_attn_{i}_v'] = block.layer[0].SelfAttention.v
        layers_to_monitor[f't5_attn_{i}_o'] = block.layer[0].SelfAttention.o

        layers_to_monitor[f't5_ff_{i}_wi'] = block.layer[1].DenseReluDense.wi
        layers_to_monitor[f't5_ff_{i}_wo'] = block.layer[1].DenseReluDense.wo


    stats = {'forget': {}, 'retain': {}}
    for dataset in stats:
        stats[dataset] = {
            name: {'sum': 0, 'sum_squares': 0, 'count': 0}
            for name in layers_to_monitor
        }

    # Hook function to collect activations
    def get_activation(name, dataset):
        def hook(model, input, output):
            with torch.no_grad():
                activation = output.detach().cpu()
                if activation.dim() == 2:
                    # For layers with output shape (batch_size, features)
                    stats[dataset][name]['sum'] += activation.sum(dim=0)
                    stats[dataset][name]['sum_squares'] += (activation ** 2).sum(dim=0)
                    stats[dataset][name]['count'] += activation.size(0)
                elif activation.dim() == 3:
                    # For layers with output shape (batch_size, seq_len, features)
                    activation = activation.view(-1, activation.size(-1))
                    stats[dataset][name]['sum'] += activation.sum(dim=0)
                    stats[dataset][name]['sum_squares'] += (activation ** 2).sum(dim=0)
                    stats[dataset][name]['count'] += activation.size(0)
        return hook

    # Function to register hooks and collect activations
    def collect_activations(dataloader, dataset):
        hooks = []
        for name, layer in layers_to_monitor.items():
            hooks.append(layer.register_forward_hook(get_activation(name, dataset)))
        model.eval()
        for urls, wids, imgs, masks, labels in dataloader:
            wids = wids.to(cuda)
            imgs = imgs.to(cuda)
            masks = masks.to(cuda)
            labels = labels.to(cuda)
            with torch.no_grad():
                wid_pred, rec_pred = model(imgs, masks, wids, labels, return_loss=False)
        for hook in hooks:
            hook.remove()

    s_time = time.time()
    collect_activations(forget_dataloader, 'forget')
    print(f'[forget] collect_activations: {time.time()-s_time:.1f}s')
    s_time = time.time()
    collect_activations(retain_dataloader, 'retain')
    print(f'[retain] collect_activations: {time.time()-s_time:.1f}s')

    # Compute importance scores and prune neurons
    for name, layer in layers_to_monitor.items():
        if name.startswith('embedding'):
            pruning_percent = prob_prune['embedding']
        elif name.startswith('t5_attn'):
            if int(name.split('_')[2]) <= 3:
                pruning_percent = prob_prune['t5_attn_low']
            if 4 <= int(name.split('_')[2]) <= 7:
                pruning_percent = prob_prune['t5_attn_mid']
            if int(name.split('_')[2]) >= 8:
                pruning_percent = prob_prune['t5_attn_high']
        elif name.startswith('t5_ff'):
            pruning_percent = prob_prune['t5_ff']
        
        print(f'{name}, {pruning_percent}')

        f_stats = stats['forget'][name]
        r_stats = stats['retain'][name]

        f_mean_square = f_stats['sum_squares'] / f_stats['count']  # E[x^2]
        f_l2_norm = torch.sqrt(f_mean_square)  # sqrt(E[x^2])

        r_mean_square = r_stats['sum_squares'] / r_stats['count']
        r_l2_norm = torch.sqrt(r_mean_square)

        importance = (f_l2_norm + epsilon) / (r_l2_norm + epsilon)

        # Prune neurons with the highest importance scores
        num_neurons = importance.numel()
        num_prune = int(pruning_percent * num_neurons)
        prune_indices = torch.topk(importance, num_prune, largest=True).indices

        # Create a mask and apply pruning
        mask = torch.ones_like(layer.weight.data)
        if isinstance(layer, nn.Linear):
            mask[prune_indices, :] = 0  # Zero out rows corresponding to pruned neurons
        else:
            continue  # Skip layers that are not linear

        prune.custom_from_mask(layer, name='weight', mask=mask)

    # Remove pruning reparameterization to finalize pruning
    for layer in layers_to_monitor.values():
        prune.remove(layer, 'weight')

    save_pruned_model(model, prefix=f'fine-grained-prune_{prefix_prune}')
    check_pruning(model, 'after')
    eval(model)


def check_pruning(model, prefix):
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    sparsity = zero_params / total_params
    print(f"[{prefix}] Model sparsity: {sparsity:.2%}")

def eval(model):
    forget_data, retain_data = loadData()
    _, _, test_data = loadDataOri()
    forget_dataloader = DataLoader(forget_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    retain_dataloader = DataLoader(retain_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    _eval_func(model, forget_dataloader, 'forget')
    _eval_func(model, retain_dataloader, 'retain')
    _eval_func(model, test_dataloader, 'test')


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


def save_pruned_model(model, prefix=''):
    # Ensure that all pruning reparameterizations are removed
    for module in model.modules():
        # Remove pruning reparameterization for nn.Linear layers
        if isinstance(module, nn.Linear):
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')
        # Remove pruning from T5 feed-forward layers
        elif hasattr(module, 'wo') and isinstance(module.wo, nn.Linear):
            if hasattr(module.wo, 'weight_orig'):
                prune.remove(module.wo, 'weight')

    if not os.path.exists('weights'):
        os.makedirs('weights')
    name = f'weights/unlearn-pruned_{prefix}.model'
    torch.save(model.state_dict(), name)


if __name__ == '__main__':
    prune_model('base-t5-htr-best.model')

