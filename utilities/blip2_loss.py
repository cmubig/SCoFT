import os
import re
import json
import numpy as np

from PIL import Image
from tqdm import tqdm
import pandas as pd
import ast
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import copy
import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from lavis.models import load_model

import io
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class BLIP2FeatureLoss(torch.nn.Module):
    def __init__(self, device=None, variant="pretrain"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.variant = variant
        model = load_model("blip2", variant, is_eval=True, device=device).float()
        self.model = model.to(self.device, dtype=torch.float32)
        self.model.eval()

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.trans = transforms.Compose([
                transforms.Resize((224, 224),interpolation=InterpolationMode.BICUBIC),
                # transforms.ToTensor(),
                normalize,
                ])  

    def forward(self, pred, target):

        pred_norm = (pred + 1.0)/2
        traget_norm = (target + 1.0)/2

        pred = self.trans(pred_norm).to(torch.float32)
        target = self.trans(traget_norm).to(torch.float32)

        # assume pred and target already on device
        self.model.image_size = pred.shape[-3:]
        # pred_batch = pred.unsqueeze(0)
        # target_batch = target.unsqueeze(0)
        pred_batch = pred
        target_batch = target
        pred_feat = self.model.ln_vision(self.model.visual_encoder(pred_batch))
        target_feat = self.model.ln_vision(self.model.visual_encoder(target_batch))
        pred_att = torch.ones(pred_feat.size()[:-1], dtype=torch.long).to(self.device)
        target_att = torch.ones(target_feat.size()[:-1], dtype=torch.long).to(self.device)

        pred_query_tokens = self.model.query_tokens.expand(pred_feat.shape[0], -1, -1)
        target_query_tokens = self.model.query_tokens.expand(target_feat.shape[0], -1, -1)

        pred_output = self.model.Qformer.bert(
            query_embeds=pred_query_tokens,
            encoder_hidden_states=pred_feat,
            encoder_attention_mask=pred_att,
            use_cache=True,
            return_dict=True,
        )
        target_output = self.model.Qformer.bert(
            query_embeds=target_query_tokens,
            encoder_hidden_states=target_feat,
            encoder_attention_mask=target_att,
            use_cache=True,
            return_dict=True,
        )

        pred_embed = self.model.vision_proj(pred_output.last_hidden_state) # B x QUERY x DIM
        target_embed = self.model.vision_proj(target_output.last_hidden_state) # B x QUERY x DIM

        pred_embed = F.normalize(pred_embed,dim=-1)
        target_embed = F.normalize(target_embed,dim=-1)

        # reduce batch dim
        pred_embed = pred_embed.squeeze(0)
        target_embed = target_embed.squeeze(0)
        difference = torch.nn.MSELoss()(pred_embed, target_embed)
        return difference




# reference functions

class BLIP2QFormerModelWrapper:
    def __init__(self, root_dir, device, variant="pretrain"):
        self.variant = variant
        self.root_dir = root_dir
        model = load_model("blip2", variant, is_eval=True, device=device).float()
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=256):
        num_text = len(texts)
        text_bs = text_batch_size
        text_ids = []
        text_embeds = []  
        text_atts = []
        for i in range(0, num_text, text_bs):
            text = texts[i: min(num_text, i+text_bs)]
            text_input = self.model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device) 
            text_output = self.model.Qformer.bert(text_input.input_ids, attention_mask = text_input.attention_mask, return_dict=True)  
            text_embed = F.normalize(self.model.text_proj(text_output.last_hidden_state[:,0,:]), dim=-1)
            text_embeds.append(text_embed)   
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)

        text_embeds = torch.cat(text_embeds,dim=0)
        text_ids = torch.cat(text_ids,dim=0)
        text_atts = torch.cat(text_atts,dim=0)
        # text_ids[:,0] = self.model.tokenizer.enc_token_id
        return text_embeds, text_ids, text_atts
    
    @torch.no_grad()
    def get_image_embeddings(self, image_loader):
        image_feats = []
        image_embeds = []
        for batch in tqdm(image_loader):
            image = batch["image"]
            image = image.to(self.device)
            self.model.image_size = image.shape[-3:]
            image_feat = self.model.ln_vision(self.model.visual_encoder(image))
            image_att = torch.ones(image_feat.size()[:-1], dtype=torch.long).to(self.device)
            query_tokens = self.model.query_tokens.expand(image_feat.shape[0], -1, -1)
            query_output = self.model.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_feat,
                encoder_attention_mask=image_att,
                use_cache=True,
                return_dict=True,
            )
            image_embed = self.model.vision_proj(query_output.last_hidden_state) # B x QUERY x DIM
            image_embed = F.normalize(image_embed,dim=-1) 

            image_feats.append(image_feat.cpu())
            image_embeds.append(image_embed)
            
        image_feats = torch.cat(image_feats,dim=0)
        image_embeds = torch.cat(image_embeds,dim=0)
        return image_feats, image_embeds