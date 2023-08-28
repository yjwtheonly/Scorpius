#%%
import pickle as pkl
from typing import Dict, Tuple, List
import os
import numpy as np
import json
import dill
import logging
import argparse 
import math
from pprint import pprint
import pandas as pd
from collections import defaultdict
import copy
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.autograd as autograd
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss

from model import Distmult, Complex, Conve
import utils

import sys

import dill

sys.path.append("..")
import Parameters

from transformers import GPT2Tokenizer, GPT2LMHeadModel

logger =  None
def generate_nghbrs_single_entity(x, edge_nghbrs, bound):

    ret_S = set(x)
    ret_L = [x]
    b = 0 
    while(b < len(ret_L)):
        s = ret_L[b]
        if s in edge_nghbrs.keys():
            for v in edge_nghbrs[s]:
                if v not in ret_S:
                    ret_S.add(v)
                    ret_L.append(v)
                    if len(ret_L) == bound:
                        return ret_L
        b += 1
    return ret_L

def generate_nghbrs(target_data, edge_nghbrs, args):
    n_dict = {}
    for i, (s, r, o) in enumerate(target_data):
        L_s = generate_nghbrs_single_entity(s, edge_nghbrs, args.neighbor_num)
        L_o = generate_nghbrs_single_entity(o, edge_nghbrs, args.neighbor_num)
        n_dict[i] = list(set(L_s + L_o))
        n_dict[i].sort()
    return n_dict 
#%%
def check_edge(s, r, o, used_trip = None, args = None):
    """Double check"""
    if args is None:
        return True
    if not args.target_existed:
        assert (s+'_'+o in used_trip) == args.target_existed
    else:
        s = entityid_to_nodetype[s]
        o = entityid_to_nodetype[o]
        r_tp = Parameters.edge_id_to_type[int(r)]
        r_tp = r_tp.split(':')[0]
        r_tp = r_tp.split('-')
        assert s == r_tp[0] and o == r_tp[1]

def get_model_loss(batch, model, device, args = None):
    s,r,o = batch[:,0], batch[:,1], batch[:,2]

    emb_s = model.emb_e(s).squeeze(dim=1)
    emb_r = model.emb_rel(r).squeeze(dim=1)
    emb_o = model.emb_e(o).squeeze(dim=1)

    if args.add_reciprocals:
        r_rev = r + n_rel
        emb_rrev = model.emb_rel(r_rev).squeeze(dim=1)
    else:
        r_rev = r
        emb_rrev = emb_r

    pred_sr = model.forward(emb_s, emb_r, mode='rhs')
    loss_sr = model.loss(pred_sr, o) # Cross entropy loss

    pred_or = model.forward(emb_o, emb_rrev, mode='lhs')
    loss_or = model.loss(pred_or, s)

    train_loss = loss_sr + loss_or
    return train_loss

def get_model_loss_without_softmax(batch, model, device=None):

    with torch.no_grad():
        s,r,o = batch[:,0], batch[:,1], batch[:,2]

        emb_s = model.emb_e(s).squeeze(dim=1)
        emb_r = model.emb_rel(r).squeeze(dim=1)

        pred = model.forward(emb_s, emb_r)
        return -pred[range(o.shape[0]), o]

def lp_regularizer(model, weight, p):
    trainable_params = [model.emb_e.weight, model.emb_rel.weight]
    norm = 0
    for i in range(len(trainable_params)):
        norm += weight * torch.sum( torch.abs(trainable_params[i]) ** p)
    return norm

def n3_regularizer(factors, weight, p):
    norm = 0
    for f in factors:
        norm += weight * torch.sum(torch.abs(f) ** p)
    return norm / factors[0].shape[0] 

def get_train_loss(batch, model, device, args):
    #batch = batch[0].to(device)
    s,r,o = batch[:,0], batch[:,1], batch[:,2]

    emb_s = model.emb_e(s).squeeze(dim=1)
    emb_r = model.emb_rel(r).squeeze(dim=1)
    emb_o = model.emb_e(o).squeeze(dim=1)

    if args.add_reciprocals:
        r_rev = r + n_rel
        emb_rrev = model.emb_rel(r_rev).squeeze(dim=1)
    else:
        r_rev = r
        emb_rrev = emb_r

    pred_sr = model.forward(emb_s, emb_r, mode='rhs')
    loss_sr = model.loss(pred_sr, o) # loss is cross entropy loss

    pred_or = model.forward(emb_o, emb_rrev, mode='lhs')
    loss_or = model.loss(pred_or, s)

    train_loss = loss_sr + loss_or
    
    if (args.reg_weight != 0.0 and args.reg_norm == 3):
        #self.logger.info('Computing regularizer weight')
        if model == 'complex':
            emb_dim = args.embedding_dim #int(self.args.embedding_dim/2)
            lhs = (emb_s[:, :emb_dim], emb_s[:, emb_dim:])
            rel = (emb_r[:, :emb_dim], emb_r[:, emb_dim:])
            rel_rev = (emb_rrev[:, :emb_dim], emb_rrev[:, emb_dim:])
            rhs = (emb_o[:, :emb_dim], emb_o[:, emb_dim:])

            #print(lhs[0].shape, lhs[1].shape)
            factors_sr = (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                        torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                        torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
                      )
            factors_or = (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                        torch.sqrt(rel_rev[0] ** 2 + rel_rev[1] ** 2),
                        torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
                      )
        else:
            factors_sr = (emb_s, emb_r, emb_o)
            factors_or = (emb_s, emb_rrev, emb_o)

        train_loss  += n3_regularizer(factors_sr, args.reg_weight, p=3)
        train_loss  += n3_regularizer(factors_or, args.reg_weight, p=3)

    if (args.reg_weight != 0.0 and args.reg_norm == 2):
        train_loss += lp_regularizer(model, args.reg_weight, p=2)
    
    return train_loss
def hv(loss, model_params, v):
    grad = autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
    Hv = autograd.grad(grad, model_params, grad_outputs=v)
    return Hv
def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

def get_inverse_hvp_lissa(v, model, device, param_influence, train_data, args):

    damping = args.damping
    num_samples = args.lissa_repeat
    scale = args.scale 
    train_batch_size = args.lissa_batch_size
    lissa_num_batches = math.ceil(train_data.shape[0]/train_batch_size)
    recursion_depth = int(lissa_num_batches*args.lissa_depth)

    ihvp = None
    # print('inversing hvp...')
    for i in range(num_samples):
        cur_estimate = v
        #lissa_data_iterator = iter(train_loader)
        input_data = torch.from_numpy(train_data.astype('int64'))
        actual_examples = input_data[torch.randperm(input_data.shape[0]), :]
        del input_data
        
        b_begin = 0
        for j in range(recursion_depth):
            model.zero_grad() # same as optimizer.zero_grad()
            if b_begin >= actual_examples.shape[0]:
                b_begin = 0
                input_data = torch.from_numpy(train_data.astype('int64'))
                actual_examples = input_data[torch.randperm(input_data.shape[0]), :]
                del input_data
            
            input_batch = actual_examples[b_begin: b_begin + train_batch_size]
            input_batch = input_batch.to(device)
            
            train_loss = get_train_loss(input_batch, model, device, args)
            
            hvp = hv(train_loss, param_influence, cur_estimate)
            cur_estimate = [_a + (1-damping)*_b - _c / scale for _a, _b, _c in zip(v, cur_estimate, hvp)]
            # if (j%200 == 0) or (j == recursion_depth -1 ):
            #     logger.info("Recursion at depth %s: norm is %f" % (j, np.linalg.norm(gather_flat_grad(cur_estimate).cpu().numpy())))
            
            b_begin += train_batch_size
        
        if ihvp == None:
            ihvp = [_a / scale for _a in cur_estimate]
        else:
            ihvp = [_a + _b / scale for _a, _b in zip(ihvp, cur_estimate)]

    # logger.info("Final ihvp norm is %f" % (np.linalg.norm(gather_flat_grad(ihvp).cpu().numpy())))
    return_ihvp = gather_flat_grad(ihvp)
    return_ihvp /= num_samples
    
    return return_ihvp

#%%
def before_global_attack(device, n_rel, data, target_data, neighbors, model, 
                    filters:Dict[str, Dict[Tuple[str, int], torch.Tensor]],
                    entityid_to_nodetype, batch_size, args, lissa_path, target_disease):

    if os.path.exists(lissa_path) and not args.update_lissa:
        with open(lissa_path, 'rb') as fl:
            ret = dill.load(fl)
        return ret
    ret = {}

    test_data = []
    for i in target_disease:
        tp = entityid_to_nodetype[str(i)]
        # r = torch.LongTensor([[10]]).to(device)
        assert tp == 'disease'
        if tp == 'disease':
            for target in target_data:
                test_data.append([str(target), str(10), str(i)])
    test_data = np.array(test_data)

    for target_trip in tqdm(test_data):

        target_trip_ori = target_trip
        trip_name = '_'.join(list(target_trip_ori))
        target_trip = target_trip[None, :] # add a batch dimension
        target_trip = torch.from_numpy(target_trip.astype('int64')).to(device)
        # target_s, target_r, target_o = target_trip[:,0], target_trip[:,1], target_trip[:,2]
        # target_vec = model.score_triples_vec(target_s, target_r, target_o)

        model.eval()
        model.zero_grad()
        target_loss = get_model_loss(target_trip, model, device)
        target_grads = autograd.grad(target_loss, param_influence)

        model.train()
        inverse_hvp = get_inverse_hvp_lissa(target_grads, model, device, 
                                            param_influence, data, args)
        model.eval()
        inverse_hvp = inverse_hvp.detach().cpu().unsqueeze(0)
        ret[trip_name] = inverse_hvp
    with open(lissa_path, 'wb') as fl:
        dill.dump(ret, fl)
    return ret
    
def global_addtion_attack(device, n_rel, data, target_data, neighbors, model, 
                    filters:Dict[str, Dict[Tuple[str, int], torch.Tensor]],
                    entityid_to_nodetype, batch_size, args, lissa, target_disease):

    logger.info('------  Generating edits per target triple ------')
    start_time = time.time()
    logger.info('Start time: {0}'.format(str(start_time)))

    used_trip = set()
    print("Processing used triples ...")
    for s, r, o in tqdm(data):
        used_trip.add(s+'_'+o)
        # used_trip.add(o+'_'+s)
    print('Size of used triples:', len(used_trip))
    logger.info('Size of used triples: {0}'.format(len(used_trip)))

    ret_trip = []
    score_record = []
    real_add_rank_ratio = 0

    with open(score_path, 'rb') as fl:
        score_record = pkl.load(fl)
    for i, target in enumerate(target_data):

        print('\n\n------  Attacking target tripid:', i, 'tot:', len(target_data), '   ------')
        # lissa_hvp = []
        target_trip = []
        for disease in target_disease:
            target_trip.append([target, str(10), disease])
        #     nm = '{}_{}_{}'.format(target, 10, disease)
        #     lissa_hvp.append(lissa[nm])
        # lissa_hvp = torch.cat(lissa_hvp, dim = 0).to(device)

        target_trip = np.array(target_trip)
        target_trip = torch.from_numpy(target_trip.astype('int64')).to(device)

        model.eval()
        model.zero_grad()
        target_loss = get_model_loss(target_trip, model, device)
        target_grads = autograd.grad(target_loss, param_influence)

        model.train()
        inverse_hvp = get_inverse_hvp_lissa(target_grads, model, device, 
                                            param_influence, data, args)

        model.eval()

        nghbr_trip = []
        s = str(target)
        tp = entityid_to_nodetype[s]
        for nghbr in tqdm(neighbors):
            o = str(nghbr)
            if s!=o and s+'_'+o not in used_trip:
                for r in range(n_rel):
                    if (tp, r) in filters["rhs"].keys() and filters["rhs"][(tp, r)][int(o)] == True:
                            nghbr_trip.append([s, str(r), o])

        nghbr_trip = np.asarray(nghbr_trip)
        influences = []      
        edge_losses = []
        
        # nghbr_cos_log_prob, nghbr_LM_log_prob = score_record[i]
        # assert nghbr_cos_log_prob.shape[0] == nghbr_trip.shape[0]
        
        for train_trip in tqdm(nghbr_trip):
            #model.train() #batch norm cannot be used here
            train_trip = train_trip[None, :] # add batch dim
            train_trip = torch.from_numpy(train_trip.astype('int64')).to(device)
            #### L-train gradient ####
            edge_loss = get_model_loss_without_softmax(train_trip, model, device).squeeze()
            edge_losses.append(edge_loss.unsqueeze(0).detach())
            model.zero_grad()
            train_loss = get_model_loss(train_trip, model, device, args)
            train_grads = autograd.grad(train_loss, param_influence)
            train_grads = gather_flat_grad(train_grads)
            influence = torch.dot(inverse_hvp, train_grads) #default dim=1
            influences.append(influence.unsqueeze(0).detach())

        edge_losses = torch.cat(edge_losses, dim = -1)
        influences = torch.cat(influences, dim = -1)
        edge_losses_log_prob = torch.log(F.softmax(-edge_losses, dim = -1))
        influences_log_prob = torch.log(F.softmax(influences, dim = -1))

        inf_score_sorted, influences_sort = torch.sort(influences_log_prob, -1, descending=True)
        edge_score_sorted, edge_sort = torch.sort(edge_losses_log_prob, -1, descending=True)
        influences_sort = influences_sort.cpu().numpy()
        edge_sort = edge_sort.cpu().numpy()
        inf_score_sorted = inf_score_sorted.cpu().numpy()
        edge_score_sorted = edge_score_sorted.cpu().numpy()

        logger.info('')
        logger.info('Top 8 inf_score: {}'.format(" ".join(map(str, list(inf_score_sorted[:8])))))
        logger.info('Top 8 edge_score: {}'.format(" ".join(map(str, list(edge_score_sorted[:8])))))

        nghbr_cos_log_prob = influences_log_prob.detach().cpu().numpy()
        nghbr_LM_log_prob = edge_losses_log_prob.detach().cpu().numpy()
        max_sim = np.max(nghbr_cos_log_prob)
        min_sim = np.min(nghbr_cos_log_prob)
        max_LM = np.max(nghbr_LM_log_prob)
        min_LM = np.min(nghbr_LM_log_prob)

        # final_score = nghbr_cos_log_prob + nghbr_LM_log_prob
        final_score = nghbr_cos_log_prob

        index = np.argmax(final_score[:-1])
        # p = np.where(index == edge_sort)[0][0]
        # logger.info('Added edge\'s edge rank ratio: {}'.format(p / edge_sort.shape[0]))
        real_add_rank_ratio += p
        add_trip = nghbr_trip[index]
        logger.info('max_inf: {0:.8}, min_inf: {1:.8}, max_edge: {2:.8}, min_edge: {3:.8}'.format(max_sim, min_sim, max_LM, min_LM))
        logger.info('Attack trip: {0}_{1}_{2}.\n Influnce score: {3:.8}. Edge score: {4:.8}.'.format(add_trip[0], add_trip[1], add_trip[2],
                                                                                                                            nghbr_cos_log_prob[index], nghbr_LM_log_prob[index]))
        ret_trip.append(add_trip)   
        score_record.append((nghbr_cos_log_prob, nghbr_LM_log_prob)) 
    real_add_rank_ratio = real_add_rank_ratio  / target_data.shape[0]
    logger.info('Mean real ratio: {}.'.format(real_add_rank_ratio))
    return ret_trip, score_record

def addition_attack(param_influence, device, n_rel, data, target_data, neighbors, model, 
                    filters:Dict[str, Dict[Tuple[str, int], torch.Tensor]],
                    entityid_to_nodetype, batch_size, args, load_Record = False, divide_bound = None, data_mean = None, data_std = None, cache_intermidiate = True):

    if logger:
        logger.info('------  Generating edits per target triple ------')
    start_time = time.time()
    if logger:
        logger.info('Start time: {0}'.format(str(start_time)))

    used_trip = set()
    print("Processing used triples ...")
    for s, r, o in tqdm(data):
        used_trip.add(s+'_'+o)
        # used_trip.add(o+'_'+s)
    print('Size of used triples:', len(used_trip))
    if logger:
        logger.info('Size of used triples: {0}'.format(len(used_trip)))

    nghbr_trip_len = []
    ret_trip = []
    score_record = []
    direct_add_rank_ratio = 0
    real_add_rank_ratio = 0
    bad_ratio = 0

    RRcord = []
    print('****'*10)
    if load_Record:
        print('Load intermidiate file')
        with open(intermidiate_path, 'rb') as fl:
            RRcord = dill.load(fl)
    else:
        print('Donnot load intermidiate file')

    for i, target_trip in enumerate(target_data):

        print('\n\n------  Attacking target tripid:', i, '   ------')
        target_nghbrs = neighbors[i]
        for a in target_nghbrs:
            if str(a) == '-1':
                raise Exception('pppp')

        target_trip_ori = target_trip
        check_edge(target_trip[0], target_trip[1], target_trip[2], used_trip)
        target_trip = target_trip[None, :] # add a batch dimension
        target_trip = torch.from_numpy(target_trip.astype('int64')).to(device)
        # target_s, target_r, target_o = target_trip[:,0], target_trip[:,1], target_trip[:,2]
        # target_vec = model.score_triples_vec(target_s, target_r, target_o)

        model.eval()

        if load_Record:
            o_target_trip, nghbr_trip, edge_losses, influences, edge_losses_log_prob, influences_log_prob = RRcord[i]
            assert (o_target_trip.cpu() == target_trip.cpu()).sum().item() == 3
        else:
            model.zero_grad()
            target_loss = get_model_loss(target_trip, model, device, args)
            target_grads = autograd.grad(target_loss, param_influence)

            model.train()
            inverse_hvp = get_inverse_hvp_lissa(target_grads, model, device, 
                                                param_influence, data, args)

            model.eval()
            nghbr_trip = []
            valid_trip = 0
            if args.candidate_mode == 'quadratic':
                s_o_list = [(i, j) for i in target_nghbrs for j in target_nghbrs]
            elif args.candidate_mode == 'linear':
                s_o_list = [(j, i) for i in target_nghbrs for j in [target_trip_ori[0], target_trip_ori[2]]] \
                        +  [(i, j) for i in target_nghbrs for j in [target_trip_ori[0], target_trip_ori[2]]]
            else:
                raise Exception('Wrong candidate_mode: '+args.candidate_mode)
            for s, o in tqdm(s_o_list): 
                tp = entityid_to_nodetype[s]
                if s!=o and s+'_'+o not in used_trip:
                    for r in range(n_rel):
                        if (tp, r) in filters["rhs"].keys() and filters["rhs"][(tp, r)][int(o)] == True:
                            # check_edge(s, r, o)
                            valid_trip += 1
                            nghbr_trip.append([s, str(r), o])
                            # logger.info('{0}_{1}_{2}'.format(s, str(r), o))
            nghbr_trip_len.append(len(nghbr_trip))
            print('Valid trip:', valid_trip)

            if target_trip_ori[0]+'_'+target_trip_ori[2] not in used_trip:
                nghbr_trip.append(target_trip_ori)
            nghbr_trip = np.asarray(nghbr_trip)
            print("Edge scoring ...")

            influences = []
            edge_losses = []

            for train_trip in tqdm(nghbr_trip):
                #model.train() #batch norm cannot be used here
                train_trip = train_trip[None, :] # add batch dim
                train_trip = torch.from_numpy(train_trip.astype('int64')).to(device)
                #### L-train gradient ####
                edge_loss = get_model_loss_without_softmax(train_trip, model, device).squeeze()
                edge_losses.append(edge_loss.unsqueeze(0).detach())
                model.zero_grad()
                train_loss = get_model_loss(train_trip, model, device, args)
                train_grads = autograd.grad(train_loss, param_influence)
                train_grads = gather_flat_grad(train_grads)
                influence = torch.dot(inverse_hvp, train_grads) #default dim=1
                influences.append(influence.unsqueeze(0).detach())  

            edge_losses = torch.cat(edge_losses, dim = -1)
            influences = torch.cat(influences, dim = -1)
            edge_losses_log_prob = torch.log(F.softmax(-edge_losses, dim = -1))
            influences_log_prob = torch.log(F.softmax(influences, dim = -1))
            std_scale = torch.std(edge_losses_log_prob) / torch.std(influences_log_prob)
            influences_log_prob = (influences_log_prob - influences_log_prob.mean()) * std_scale + edge_losses_log_prob.mean()
            
            RRcord.append([target_trip.detach(), nghbr_trip, edge_losses, influences, edge_losses_log_prob, influences_log_prob])

        inf_score_sorted, influences_sort = torch.sort(influences_log_prob, -1, descending=True)
        edge_score_sorted, edge_sort = torch.sort(edge_losses_log_prob, -1, descending=True)

        influences_sort = influences_sort.cpu().numpy()
        edge_sort = edge_sort.cpu().numpy()
        inf_score_sorted = inf_score_sorted.cpu().numpy()
        edge_score_sorted = edge_score_sorted.cpu().numpy()
        edge_losses = edge_losses.cpu().numpy()

        p = np.where(influences_sort[0] == edge_sort)[0][0]
        direct_add_rank_ratio += p / edge_sort.shape[0]
        if logger:
            logger.info('Top 8 inf_score: {}'.format(" ".join(map(str, list(inf_score_sorted[:8])))))
            logger.info('Top 8 edge_score: {}'.format(" ".join(map(str, list(edge_score_sorted[:8])))))

        nghbr_cos_log_prob = influences_log_prob.detach().cpu().numpy()
        nghbr_LM_log_prob = edge_losses_log_prob.detach().cpu().numpy()
        max_sim = nghbr_cos_log_prob[influences_sort[0]]
        min_sim = nghbr_cos_log_prob[influences_sort[-1]]
        max_LM = nghbr_LM_log_prob[edge_sort[0]]
        min_LM = nghbr_LM_log_prob[edge_sort[-1]]
        direct_score_0 = 0
        direct_score_1 = 0
        if target_trip_ori[0]+'_'+target_trip_ori[2] not in used_trip:
            direct_score_0 = nghbr_cos_log_prob[-1]
            direct_score_1 = nghbr_LM_log_prob[-1]
    
        # bound = math.log(1 / nghbr_LM_log_prob.shape[0])
        bound = 1 - args.reasonable_rate
        edge_losses = (edge_losses - data_mean) / data_std
        edge_losses_prob =  1 / ( 1 + np.exp(edge_losses - divide_bound) )
        nghbr_LM_log_prob[edge_losses_prob < bound] = -(1e20)

        final_score = nghbr_cos_log_prob + nghbr_LM_log_prob

        index = np.argmax(final_score[:-1])
        sort_index = [(i, final_score[i])for i in range(len(final_score) - 1)]
        sort_index = sorted(sort_index, key=lambda x: x[1], reverse=True)
        assert sort_index[0][0] == index

        p = np.where(index == edge_sort)[0][0]
        if logger:
            logger.info('Bad edge ratio: {}'.format((edge_losses_prob < bound).mean()))
            logger.info('Bounded edge\'s edge rank ratio: {}'.format(p / edge_sort.shape[0]))
        real_add_rank_ratio += p / edge_sort.shape[0]
        bad_ratio += (edge_losses_prob < bound).mean()

        add_trip = nghbr_trip[index]

        if (int(add_trip[0]) == int(-1)):
            add_trip[0], add_trip[1], add_trip[2] = -1, -1, -1
            print(final_score.shape, index, edge_losses_prob[index], bound)
            raise Exception('??')

        if logger:
            logger.info('max_inf: {0:.8}, min_inf: {1:.8}, max_edge: {2:.8}, min_edge: {3:.8}'.format(max_sim, min_sim, max_LM, min_LM))
            logger.info('Target trip: {0}_{1}_{2}. Attack trip: {3}_{4}_{5}.\n Influnce score: {6:.8}. Edge score: {7:.8}. Direct score: {8:.8} + {9:.8}'.format(target_trip_ori[0],target_trip_ori[1], target_trip_ori[2], 
                                                                                                                            add_trip[0], add_trip[1], add_trip[2],
                                                                                                                            nghbr_cos_log_prob[index], nghbr_LM_log_prob[index],
                                                                                                                            direct_score_0, direct_score_1))
        if (args.added_edge_num == '' or int(args.added_edge_num) == 1):
            ret_trip.append(add_trip)
        else:
            edge_num = int(args.added_edge_num)
            for i in range(edge_num):
                ret_trip.append(nghbr_trip[sort_index[i][0]])
        score_record.append((nghbr_cos_log_prob, nghbr_LM_log_prob)) 
    
    if not load_Record and cache_intermidiate:
        with open(intermidiate_path, 'wb') as fl:
            dill.dump(RRcord, fl)
    direct_add_rank_ratio = direct_add_rank_ratio / target_data.shape[0]
    real_add_rank_ratio = real_add_rank_ratio  / target_data.shape[0]
    bad_ratio = bad_ratio / target_data.shape[0]
    if logger:
        logger.info('Mean direct ratio: {}. Mean real ratio: {}. Mean bad ratio: {}'.format(direct_add_rank_ratio, real_add_rank_ratio, bad_ratio))
    return ret_trip, score_record

def calculate_edge_bound(data, model, device, n_ent):

    tmp = np.random.choice(a = data.shape[0], size = data.shape[0] // 10, replace=False)
    existed_data= data[tmp, :]

    print('calculating edge bound ...')
    print(existed_data.shape)

    existed_edge = set()
    for src_trip in existed_data:
        existed_edge.add('_'.join(list(src_trip)))
    
    not_existed = []
    for s, r, o in  existed_data:

        if np.random.randint(0, n_ent) % 2 == 0:
            while True:
                oo = np.random.randint(0, n_ent)
                if '_'.join([s, r, str(oo)]) not in existed_edge:
                    not_existed.append([s, r, str(oo)])
                    break
        else:
            while True:
                ss = np.random.randint(0, n_ent)
                if '_'.join([str(ss), r, o]) not in existed_edge:
                    not_existed.append([str(ss), r, o])
                    break   
    existed_data = np.array(existed_data)
    not_existed = np.array(not_existed)
    existed_data = torch.from_numpy(existed_data.astype('int64')).to(device)
    not_existed = torch.from_numpy(not_existed.astype('int64')).to(device)
    loss_existed = get_model_loss_without_softmax(existed_data, model).cpu().numpy()
    loss_not_existed = get_model_loss_without_softmax(not_existed, model).cpu().numpy()
    tot_loss = np.hstack((loss_existed, loss_not_existed))
    tot_mean, tot_std = np.mean(tot_loss), np.std(tot_loss)

    loss_existed = (loss_existed - tot_mean) / tot_std
    loss_not_existed = (loss_not_existed - tot_mean) / tot_std

    print('Tot mean: {}, Tot std: {}'.format(tot_mean, tot_std))

    # print(np.mean(loss_existed), np.std(loss_existed), np.max(loss_existed))
    # print(np.mean(loss_not_existed), np.std(loss_not_existed), np.min(loss_not_existed))
    l_mean, l_std = np.mean(loss_existed), np.std(loss_existed)
    r_mean, r_std = np.mean(loss_not_existed), np.std(loss_not_existed)

    A = -1/(l_std**2) + 1/(r_std**2)
    B = 2 * (-r_mean/(r_std**2) + l_mean/(l_std**2))
    C = (r_mean**2)/(r_std**2)-(l_mean**2)/(l_std**2) + np.log((r_std**2)/(l_std**2))

    delta = B**2 - 4*A*C

    x_1 = ( -B + math.sqrt(delta) ) / (2*A)
    x_2 = ( -B - math.sqrt(delta) ) / (2*A)

    x = None
    if (x_1 > l_mean and x_1 < r_mean):
        x = x_1
    if (x_2 > l_mean and x_2 < r_mean):
        x = x_2
    if not x:
        raise Exception('Bad model!!!!')
    TP = (loss_existed < x).mean()
    TN = (loss_not_existed > x).mean()
    FP = (loss_not_existed < x).mean()
    FN = (loss_existed > x).mean()
    print('X:{}, TP:{}, TN:{}, FP:{}, FN{}'.format(x, TP, TN, FP, FN))

    sig_existed = 1 / ( 1 + np.exp(loss_existed- x) ) # negtive important
    sig_not_existed = 1 / ( 1 + np.exp(loss_not_existed - x) )

    print('Positive mean score:', sig_existed.mean(),'Negetive mean score:', sig_not_existed.mean())

    return x, tot_mean, tot_std


#%%
if __name__ == '__main__':
    parser = utils.get_argument_parser()
    parser = utils.add_attack_parameters(parser)
    args = parser.parse_args()
    args = utils.set_hyperparams(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.device1 = device
    if torch.cuda.device_count() >= 2:
        args.device = "cuda:0"
        args.device1 = "cuda:1"

    utils.seed_all(args.seed)
    np.set_printoptions(precision=5)
    cudnn.benchmark = False

    model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
    model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)
    data_path = os.path.join('processed_data', args.data)
    target_path = os.path.join(data_path, 'DD_target_{0}_{1}_{2}_{3}_{4}_{5}.txt'.format(args.model, args.data, args.target_split, args.target_size, 'exists:'+str(args.target_existed), args.attack_goal))
    lissa_path = 'lissa/{0}_{1}_{2}'.format(args.model, 
                                                args.data, 
                                                args.target_size)
    intermidiate_path = 'intermidiate/{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(args.model, 
                                                                args.target_split, 
                                                                args.target_size, 
                                                                'exists:'+str(args.target_existed),
                                                                args.neighbor_num,
                                                                args.candidate_mode,
                                                                args.attack_goal)
    log_path = 'logs/attack_logs/cos_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}'.format(args.model, 
                                                                args.target_split, 
                                                                args.target_size, 
                                                                'exists:'+str(args.target_existed),
                                                                args.neighbor_num,
                                                                args.candidate_mode,
                                                                args.attack_goal,
                                                                str(args.reasonable_rate))
    print(log_path)
    attack_path = os.path.join('attack_results', args.data, 'cos_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}{8}.txt'.format(args.model, 
                                                                                                        args.target_split, 
                                                                                                        args.target_size, 
                                                                                                        'exists:'+str(args.target_existed),
                                                                                                        args.neighbor_num,
                                                                                                        args.candidate_mode,
                                                                                                        args.attack_goal,
                                                                                                        str(args.reasonable_rate),
                                                                                                        str(args.added_edge_num)))

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                datefmt = '%m/%d/%Y %H:%M:%S',
                                level = logging.INFO,
                                filename = log_path
                            )
    logger = logging.getLogger(__name__)
    logger.info(vars(args))
    #%%
    n_ent, n_rel, ent_to_id, rel_to_id = utils.generate_dicts(data_path)
    data  = utils.load_data(os.path.join(data_path, 'all.txt'))
    with open(os.path.join(data_path, 'filter.pickle'), 'rb') as fl:
        filters = pkl.load(fl)
    with open(os.path.join(data_path, 'entityid_to_nodetype.json'), 'r') as fl:
        entityid_to_nodetype = json.load(fl)
    with open(os.path.join(data_path, 'edge_nghbrs.pickle'), 'rb') as fl:
        edge_nghbrs = pkl.load(fl)
    with open(os.path.join(data_path, 'disease_meshid.pickle'), 'rb') as fl:
        disease_meshid = pkl.load(fl)
    with open(os.path.join(data_path, 'entities_dict.json'), 'r') as fl:
        entity_to_id = json.load(fl)
    with open(Parameters.GNBRfile+'entity_raw_name', 'rb') as fl:
        entity_raw_name = pkl.load(fl)
    #%%
    init_mask = np.asarray([0] * n_ent).astype('int64')
    init_mask = (init_mask == 1)
    for k, v in filters.items():
        for kk, vv in v.items():
            tmp = init_mask.copy()
            tmp[np.asarray(vv)] = True
            t = torch.ByteTensor(tmp).to(args.device)
            filters[k][kk] = t
    #%%
    model = utils.load_model(model_path, args, n_ent, n_rel, args.device)
    divide_bound, data_mean, data_std = calculate_edge_bound(data, model, args.device, n_ent)
    # index = torch.LongTensor([0, 1]).to(device)
    # print(model.emb_rel(index)[:, :32])
    # print(model.emb_e(index)[:, :32])
    # raise Exception
    #%%
    target_data = utils.load_data(target_path)
    if args.attack_goal == 'single':
        neighbors = generate_nghbrs(target_data, edge_nghbrs, args)
    elif args.attack_goal == 'global':
        s_set = set()
        for s, r, o in target_data:
            s_set.add(s)
        target_data = list(s_set)
        target_data.sort()
        target_data = np.array(target_data, dtype=str)
        neighbors = []
        for i in list(range(n_ent)):
            tp = entityid_to_nodetype[str(i)]
            # r = torch.LongTensor([[10]]).to(device)
            if tp == 'gene':
                neighbors.append(str(i))
        target_disease = []
        tid = 1
        bound = 50
        while True:
            meshid = disease_meshid[tid][0]
            fre = disease_meshid[tid][1]
            if len(entity_raw_name[meshid]) > 4:
                target_disease.append(entity_to_id[meshid])
                bound -= 1
                if bound == 0:
                    break
            tid += 1
    else:
        raise Exception('Wrong attack_goal: '+args.attack_goal)

    param_optimizer = list(model.named_parameters())
    param_influence = []
    for n,p in param_optimizer:
        param_influence.append(p)
    if args.attack_goal == 'single':
        len_list = []
        for v in neighbors.values():
            len_list.append(len(v))
        mean_len = np.mean(len_list)
    else:
        mean_len = len(neighbors)
    print('Mean length of neighbors:', mean_len)
    logger.info("Mean length of neighbors: {0}".format(mean_len))

    # GPT_LM = LMscore_calculator(data_path, args)
    lissa_num_batches = math.ceil(data.shape[0]/args.lissa_batch_size)
    logger.info('-------- Lissa Params for IHVP --------')
    logger.info('Damping: {0}'.format(args.damping))
    logger.info('Lissa_repeat: {0}'.format(args.lissa_repeat))
    logger.info('Lissa_depth: {0}'.format(args.lissa_depth))
    logger.info('Scale: {0}'.format(args.scale))
    logger.info('Lissa batch size: {0}'.format(args.lissa_batch_size))
    logger.info('Lissa num bacthes: {0}'.format(lissa_num_batches))

    score_path = os.path.join('attack_results', args.data, 'score_cos_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}{8}.txt'.format(args.model, 
                                                                                                        args.target_split, 
                                                                                                        args.target_size, 
                                                                                                        'exists:'+str(args.target_existed),
                                                                                                        args.neighbor_num,
                                                                                                        args.candidate_mode,
                                                                                                        args.attack_goal,
                                                                                                        str(args.reasonable_rate),
                                                                                                        str(args.added_edge_num)))

    if args.attack_goal == 'single':
        attack_trip, score_record = addition_attack(param_influence, args.device, n_rel, data, target_data, neighbors, model, filters, entityid_to_nodetype, args.attack_batch_size, args, load_Record = args.load_existed, divide_bound = divide_bound, data_mean = data_mean, data_std = data_std)
    else:
        # lissa = before_global_attack(args.device, n_rel, data, target_data, neighbors, model, filters, entityid_to_nodetype, args.attack_batch_size, args, lissa_path, target_disease)

        attack_trip, score_record = global_addtion_attack(args.device, n_rel, data, target_data, neighbors, model, filters, entityid_to_nodetype, args.attack_batch_size, args, None, target_disease)

    utils.save_data(attack_path, attack_trip)

    logger.info("Attack triples are saved in " + attack_path)
    with open(score_path, 'wb') as fl:
        pkl.dump(score_record, fl)