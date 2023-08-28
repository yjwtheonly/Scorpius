'''
A file modified on https://github.com/PeruBhardwaj/AttributionAttack/blob/main/KGEAttack/ConvE/utils.py
'''
#%%
import logging
import time
from tqdm import tqdm
import io
import pandas as pd
import numpy as np
import os
import json

import argparse
import torch
import random

from yaml import parse

from model import Conve, Distmult, Complex

logger = logging.getLogger(__name__)
#%%
def generate_dicts(data_path):
    with open (os.path.join(data_path, 'entities_dict.json'), 'r') as f:
        ent_to_id = json.load(f)
    with open (os.path.join(data_path, 'relations_dict.json'), 'r') as f:
        rel_to_id = json.load(f)
    n_ent = len(list(ent_to_id.keys()))
    n_rel = len(list(rel_to_id.keys()))
    
    return n_ent, n_rel, ent_to_id, rel_to_id

def save_data(file_name, data):
    with open(file_name, 'w') as fl:
        for item in data:
            fl.write("%s\n" % "\t".join(map(str, item)))

def load_data(file_name, drop = True):
    df = pd.read_csv(file_name, sep='\t', header=None, names=None, dtype=str)
    if drop:
        df = df.drop_duplicates()
    else:
        pass
    return df.values

def seed_all(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def add_model(args, n_ent, n_rel):
    if args.model is None:
        model = Distmult(args, n_ent, n_rel)
    elif args.model == 'distmult':
        model = Distmult(args, n_ent, n_rel)
    elif args.model == 'complex':
        model = Complex(args, n_ent, n_rel)
    elif args.model == 'conve':
        model = Conve(args, n_ent, n_rel)
    else:
        raise Exception("Unknown model!")

    return model

def load_model(model_path, args, n_ent, n_rel, device):
    # add a model and load the pre-trained params
    model = add_model(args, n_ent, n_rel)
    model.to(device)
    logger.info('Loading saved model from {0}'.format(model_path))
    state = torch.load(model_path)
    model_params = state['state_dict']
    params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
    for key, size, count in params:
        logger.info('Key:{0}, Size:{1}, Count:{2}'.format(key, size, count))
        
    model.load_state_dict(model_params)
    model.eval()
    logger.info(model)
    
    return model

def add_eval_parameters(parser):

    # parser.add_argument('--eval-mode', type = str, default = 'all', help = 'Method to evaluate the attack performance. Default: all. (all or single)')
    parser.add_argument('--cuda-name', type = str, required = True, help = 'Start a main thread on each cuda.')
    parser.add_argument('--direct', action='store_true', help = 'Directly add edge or not.')
    parser.add_argument('--seperate', action='store_true', help = 'Evaluate seperatly or not')
    parser.add_argument('--mode', type = str, default = '', help = ' '' or '' ')
    parser.add_argument('--mask-ratio', type=str, default='', help='Mask ratio for Fig4b')
    return parser

def add_attack_parameters(parser):

    # parser.add_argument('--target-split', type=str, default='0_100_1', help='Ranks to use for target set. Values are 0 for ranks==1; 1 for ranks <=10; 2 for ranks>10 and ranks<=100. Default: 1')
    parser.add_argument('--target-split', type=str, default='min', help='Methods for target triple selection. Default: min. (min or top_?, top means top_0.1)')
    parser.add_argument('--target-size', type=int, default=50, help='Number of target triples. Default: 50')
    parser.add_argument('--target-existed', action='store_true', help='Whether the targeted s_?_o already exists.')

    # parser.add_argument('--budget', type=int, default=1, help='Budget for each target triple for each corruption side')

    parser.add_argument('--attack-goal', type = str, default='single', help='Attack goal. Default: single. (single or global)')
    parser.add_argument('--neighbor-num', type = int, default=20, help='Max neighbor num for each side. Default: 20')
    parser.add_argument('--candidate-mode', type = str, default='quadratic', help = 'The method to generate candidate edge. Default: quadratic. (quadratic or linear)')
    parser.add_argument('--reasonable-rate', type = float, default=0.7, help = 'The added edge\'s existance rank prob greater than this rate')
    parser.add_argument('--added-edge-num', type = str, default='', help = 'How many edges to add for each target edge. Default: '' means 1.')
    # parser.add_argument('--neighbor-num', type = int, default=200, help='Max neighbor num for each side. Default: 200')
    # parser.add_argument('--candidate-mode', type = str, default='linear', help = 'The method to generate candidate edge. Default: quadratic. (quadratic or linear)')
    parser.add_argument('--attack-batch-size', type=int, default=256, help='Batch size for processing neighbours of target')
    parser.add_argument('--template-mode', type=str, default = 'manual', help = 'Template mode for transforming edge to single sentense. Default: manual. (manual or auto)')

    parser.add_argument('--update-lissa', action='store_true', help = 'Update lissa cache or not.')

    parser.add_argument('--GPT-batch-size', type=int, default = 64, help = 'Batch size for GPT2 when calculating LM score. Default: 64')
    parser.add_argument('--LM-softmax', action='store_true', help = 'Use a softmax head on LM prob or not.')
    parser.add_argument('--LMprob-mode', type=str, default='relative', help = 'Use the absolute LM score or calculate the destruction score when target word is replaced. Default: absolute. (absolute or relative)')

    parser.add_argument('--load-existed', action='store_true', help = 'Use cached intermidiate results or not, when only --reasonable-rate changed, set this param to True')

    return parser

def get_argument_parser():
    '''Generate an argument parser'''
    parser = argparse.ArgumentParser(description='Graph embedding')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
    
    parser.add_argument('--data', type=str, default='GNBR', help='Dataset to use: { GNBR }')
    parser.add_argument('--model', type=str, default='distmult', help='Choose from: {distmult, conve, complex}')
    
    parser.add_argument('--transe-margin', type=float, default=0.0, help='Margin value for TransE scoring function. Default:0.0')
    parser.add_argument('--transe-norm', type=int, default=2, help='P-norm value for TransE scoring function. Default:2')
    
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--lr-decay', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--max-norm', action='store_true', help='Option to add unit max norm constraint to entity embeddings')
    
    parser.add_argument('--train-batch-size', type=int, default=64, help='Batch size for train split (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, help='Batch size for test split (default: 128)')
    parser.add_argument('--valid-batch-size', type=int, default=128, help='Batch size for valid split (default: 128)')
    parser.add_argument('--KG-valid-rate', type = float, default=0.1, help='Validation rate during KG embedding training. (default: 0.1)')
    
    parser.add_argument('--save-influence-map', action='store_true', help='Save the influence map during training for gradient rollback.')
    parser.add_argument('--add-reciprocals', action='store_true')

    parser.add_argument('--embedding-dim', type=int, default=128, help='The embedding dimension (1D). Default: 128')
    parser.add_argument('--stack-width', type=int, default=16, help='The first dimension of the reshaped/stacked 2D embedding. Second dimension is inferred. Default: 20')
    #parser.add_argument('--stack_height', type=int, default=10, help='The second dimension of the reshaped/stacked 2D embedding. Default: 10')
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.3, help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('-num-filters', default=32,   type=int, help='Number of filters for convolution')
    parser.add_argument('-kernel-size', default=3, type=int, help='Kernel Size for convolution')
    
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    
    parser.add_argument('--reg-weight', type=float, default=5e-2, help='Weight for regularization. Default: 5e-2')
    parser.add_argument('--reg-norm', type=int, default=3, help='Norm for regularization. Default: 2')
    # parser.add_argument('--resume', action='store_true', help='Restore a saved model.')
    # parser.add_argument('--resume-split', type=str, default='test', help='Split to evaluate a restored model')
    # parser.add_argument('--reproduce-results', action='store_true', help='Use the hyperparameters to reproduce the results.')
    # parser.add_argument('--original-data', type=str, default='FB15k-237', help='Dataset to use; this option is needed to set the hyperparams to reproduce the results for training after attack, default: FB15k-237')
    return parser

def set_hyperparams(args):
    if args.model == 'distmult':
        args.lr = 0.005
        args.train_batch_size = 1024
        args.reg_norm = 3
    elif args.model == 'complex':
        args.lr = 0.005
        args.reg_norm = 3
        args.input_drop = 0.4
        args.train_batch_size = 1024
    elif args.model == 'conve':
        args.lr = 0.005
        args.train_batch_size = 1024
        args.reg_weight = 0.0
    
    # args.damping = 0.01 
    # args.lissa_repeat = 1 
    # args.lissa_depth = 1
    # args.scale = 500
    # args.lissa_batch_size = 100

    args.damping = 0.01 
    args.lissa_repeat = 1 
    args.lissa_depth = 1
    args.scale = 400
    args.lissa_batch_size = 300
    return args
