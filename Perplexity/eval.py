#%%
import torch
import numpy as np
from torch.autograd import Variable
from sklearn import metrics

import datetime
from typing import Dict, Tuple, List
import logging
import os
import pickle as pkl
import json 
import torch.backends.cudnn as cudnn

from tqdm import tqdm

import sys
import random
sys.path.append("..")
import Parameters

def seed_all(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

from torch.nn.modules.loss import CrossEntropyLoss
from transformers import AutoTokenizer
from transformers import BioGptForCausalLM, GPT2LMHeadModel
criterion = CrossEntropyLoss(reduction="none")

model_name = 'microsoft/biogpt'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
if model_name == 'gpt2':
    model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
elif model_name == 'microsoft/biogpt':
    model = BioGptForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
GPT_batch_size = 4

#%%

def cal_khd(path):

    with open(path, 'r') as fl:
        data = json.load(fl)

    out_list = []
    for k, v in data.items():
        out = v['out']
        if v['in'] != '':
            out_list.append(out)
    tokens = tokenizer( out_list,
                        truncation = True,
                        padding = True,
                        max_length = 500,
                        return_tensors="pt")
    
    target_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    L = len(out_list)
    assert L > 0
    ret_log_L = []

    for l in tqdm(range(0, L, GPT_batch_size)):
        R = min(L, l + GPT_batch_size)
        target = target_ids[l:R, :].to(device)
        attention = attention_mask[l:R, :].to(device)
        outputs = model(input_ids = target,
                        attention_mask = attention,
                        labels = target)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target[..., 1:].contiguous()
        Loss = criterion(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
        Loss = Loss.view(-1, shift_logits.shape[1])
        attention = attention[..., 1:].contiguous()
        log_Loss = (torch.mean(Loss * attention.float(), dim = 1) / torch.mean(attention.float(), dim = 1))
        ret_log_L.append(log_Loss.detach())

    ret_log_L = list(torch.cat(ret_log_L, -1).cpu().numpy())
    Per = np.exp(ret_log_L)

    return Per

def check_perplexity(s):

    model.eval()

    tokens = tokenizer( [s],
                        truncation = True,
                        padding = True,
                        max_length = 500,
                        return_tensors="pt")
    target_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    target = target_ids.to(device)
    attention = attention_mask.to(device)
    outputs = model(input_ids = target,
                    attention_mask = attention,
                    labels = target)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target[..., 1:].contiguous()
    Loss = criterion(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
    Loss = Loss.view(-1, shift_logits.shape[1])
    attention = attention[..., 1:].contiguous()
    log_Loss = (torch.mean(Loss * attention.float(), dim = 1) / torch.mean(attention.float(), dim = 1))
    ret_log_L = list(log_Loss.detach().cpu().numpy())
    Per = np.exp(ret_log_L)
    return Per[0]

# cal_khd('../DiseaseSpecific/generate_abstract/min_0.7_biogpt.json')
import argparse
parser = argparse.ArgumentParser(description='Eval')
parser.add_argument('--mode', type=str, default='local')
args = parser.parse_args()

if args.mode == 'local':
    tmp_list = ['random']
    toles = ['0.7', '0.5', '0.3']
    chat = []
    bioBART = []
    for a in tmp_list:
        record_chat = []
        record_bioBART = []
        for tole in toles:
            Per = cal_khd(f'../DiseaseSpecific/generate_abstract/{a}_{tole}_chat.json')
            record_chat+= list(Per)
            Per = cal_khd(f'../DiseaseSpecific/generate_abstract/{a}_{tole}{0.8}_bioBART_finetune.json')
            record_bioBART += list(Per)
        chat += record_chat
        bioBART += record_bioBART
    chat = np.array(chat)
    bioBART = np.array(bioBART)
    print(chat.shape, bioBART.shape)
    print(np.mean(chat), np.std(chat))
    print(np.mean(bioBART), np.std(bioBART))
    with open(f'eval_local_perplexity{tmp_list[0]}.pkl', 'wb') as fl:
        pkl.dump([bioBART, chat], fl)

if args.mode == 'global':

    toles = ['0.7', '0.5', '0.3']
    pre = 'random'
    chat = []
    bioBART = []
    for tole in toles:
        
        Per = cal_khd(f'../DiseaseAgnostic/generate_abstract/{pre}{tole}_chat.json')
        chat += list(Per)
        Per = cal_khd(f'../DiseaseAgnostic/generate_abstract/{pre}{tole}{0.8}_bioBART_finetune.json')
        bioBART += list(Per)
    chat = np.array(chat)
    bioBART = np.array(bioBART)
    print(chat.shape, bioBART.shape)
    print(np.mean(chat), np.std(chat))
    print(np.mean(bioBART), np.std(bioBART))
    with open(f'eval_global_perplexity{pre}.pkl', 'wb') as fl:
        pkl.dump([bioBART, chat], fl)