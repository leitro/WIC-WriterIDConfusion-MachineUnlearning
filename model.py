import torch
from einops import rearrange
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel
from dataset import tokens

FEAT = 768

class REC(nn.Module):
    def __init__(self, num_writers, vocab_size):
        super(REC, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Height: 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Height: 32 -> 16
        )
        self.embedding = nn.Linear(16 * 128, FEAT)
        self.cls_token_embedding = nn.Parameter(torch.randn(1, 1, FEAT))
        self.t5_encoder = T5EncoderModel.from_pretrained('google-t5/t5-base')
        self.writer_classifier = nn.Linear(FEAT, num_writers)
        self.text_classifier = nn.Linear(FEAT, vocab_size)
        self.writer_loss_fn = nn.CrossEntropyLoss()
        self.text_loss_fn = nn.CrossEntropyLoss(ignore_index=tokens['PAD_TOKEN']) 

    def forward(self, images, masks, wids, labels, return_loss=True, return_logits=False):
        # images: (batch_size, 1, 64, variable_width)
        batch_size = images.size(0)
        images = images.unsqueeze(1)

        features = self.conv_layers(images)  # (batch_size, 128, 16, len)
        features = features.permute(0, 3, 2, 1)
        len_seq = features.size(1)
        features = features.reshape(batch_size, len_seq, -1)
        embedded_features = self.embedding(features)  # (batch_size, len, 512)

        cls_embedding = self.cls_token_embedding.expand(batch_size, -1, -1)  # (batch_size, 1, 512)
        encoder_input = torch.cat([cls_embedding, embedded_features], dim=1)  # (batch_size, len+1, 512)
        encoder_output = self.t5_encoder(inputs_embeds=encoder_input, attention_mask=masks)

        # Writer classification using CLS token output
        cls_output = encoder_output.last_hidden_state[:, 0, :]  # (batch_size, 512)
        writer_logits = self.writer_classifier(cls_output)  # (batch_size, num_writers)

        text_outputs = encoder_output.last_hidden_state[:, 1:21, :] 
        text_logits = self.text_classifier(text_outputs)  # (batch_size, len, vocab_size)
        wid_pred = writer_logits.argmax(dim=1)
        rec_pred = text_logits.argmax(dim=-1)

        return_list = []
        if return_loss:
            writer_loss = self.writer_loss_fn(writer_logits, wids)
            text_loss = self.text_loss_fn(text_logits.view(-1, text_logits.size(-1)), labels.view(-1))
            return_list.append(writer_loss)
            return_list.append(text_loss)

        return_list.append(wid_pred)
        return_list.append(rec_pred)

        if return_logits:
            return_list.append(writer_logits)
            return_list.append(text_logits)

        return return_list

