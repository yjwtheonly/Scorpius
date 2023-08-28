#%%
import logging
from symbol import parameters
from textwrap import indent
import os
import tempfile
import sys
from matplotlib import collections
import pandas as pd
import json
from glob import glob
from tqdm import tqdm
import numpy as np
from pprint import pprint
import torch
import pickle as pkl
from collections import Counter 
# print(dir(collections))
import networkx as nx
from collections import Counter
import utils
from torch.nn import functional as F
sys.path.append("..")
import Parameters
from DiseaseSpecific.attack import calculate_edge_bound, get_model_loss_without_softmax

#%%
def load_data(file_name):
    df = pd.read_csv(file_name, sep='\t', header=None, names=None, dtype=str)
    df = df.drop_duplicates()
    return df.values

parser = utils.get_argument_parser()
parser.add_argument('--reasonable-rate', type = float, default=0.7, help = 'The added edge\'s existance rank prob greater than this rate')
parser.add_argument('--init-mode', type = str, default='single', help = 'How to select target nodes') # 'single' for case study 
parser.add_argument('--added-edge-num', type = str, default = '', help = 'Added edge num')

args = parser.parse_args()
args = utils.set_hyperparams(args)
utils.seed_all(args.seed)
graph_edge_path = '../DiseaseSpecific/processed_data/GNBR/all.txt'
idtomeshid_path = '../DiseaseSpecific/processed_data/GNBR/entities_reverse_dict.json'
model_path = f'../DiseaseSpecific/saved_models/GNBR_{args.model}_128_0.2_0.3_0.3.model'
data_path = '../DiseaseSpecific/processed_data/GNBR'
with open(Parameters.GNBRfile+'original_entity_raw_name', 'rb') as fl:
    full_entity_raw_name = pkl.load(fl)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
n_ent, n_rel, ent_to_id, rel_to_id = utils.generate_dicts(data_path)
model = utils.load_model(model_path, args, n_ent, n_rel, args.device)
print(device)

graph_edge = utils.load_data(graph_edge_path)
with open(idtomeshid_path, 'r') as fl:
    idtomeshid = json.load(fl)
print(graph_edge.shape, len(idtomeshid))

divide_bound, data_mean, data_std = calculate_edge_bound(graph_edge, model, args.device, n_ent)
print('Defender ...')
print(divide_bound, data_mean, data_std)

meshids = list(idtomeshid.values())
cal = {
    'chemical' : 0,
    'disease' : 0,
    'gene' : 0
}
for meshid in meshids:
    cal[meshid.split('_')[0]] += 1
# pprint(cal)

def check_reasonable(s, r, o):

    train_trip = np.asarray([[s, r, o]])
    train_trip = torch.from_numpy(train_trip.astype('int64')).to(device)
    edge_loss = get_model_loss_without_softmax(train_trip, model, device).squeeze()
    # edge_losse_log_prob = torch.log(F.softmax(-edge_loss, dim = -1))

    edge_loss = edge_loss.item() 
    edge_loss = (edge_loss - data_mean) / data_std
    edge_losses_prob =  1 / ( 1 + np.exp(edge_loss - divide_bound) )
    bound = 1 - args.reasonable_rate

    return (edge_losses_prob > bound),  edge_losses_prob

edgeid_to_edgetype = {}
edgeid_to_reversemask = {}
for k, id_list in Parameters.edge_type_to_id.items():
    for iid, mask in zip(id_list, Parameters.reverse_mask[k]):
        edgeid_to_edgetype[str(iid)] = k
        edgeid_to_reversemask[str(iid)] = mask
reverse_tot = 0
G = nx.DiGraph()
for s, r, o in graph_edge:
    assert idtomeshid[s].split('_')[0] == edgeid_to_edgetype[r].split('-')[0]
    if edgeid_to_reversemask[r] == 1:
        reverse_tot += 1
        G.add_edge(int(o), int(s))
    else:
        G.add_edge(int(s), int(o))
# print(reverse_tot)
print('Edge num:', G.number_of_edges(), 'Node num:', G.number_of_nodes())
pagerank_value_1 = nx.pagerank(G, max_iter = 200, tol=1.0e-7) 

#%%
with open(Parameters.UMLSfile+'drug_term', 'rb') as fl:
    drug_term = pkl.load(fl)
with open(Parameters.GNBRfile+'entity_raw_name', 'rb') as fl:
    entity_raw_name = pkl.load(fl)
drug_meshid = []
for meshid, nm in entity_raw_name.items():
    if nm.lower() in drug_term and meshid.split('_')[0] == 'chemical':
        drug_meshid.append(meshid)
drug_meshid = set(drug_meshid)
pr = list(pagerank_value_1.items())
pr.sort(key = lambda x: x[1])
sorted_rank = { 'chemical' : [],
                'gene' : [],
                'disease': [],
                'merged' : []}
for iid, score in pr:
    tp = idtomeshid[str(iid)].split('_')[0]
    if tp == 'chemical':
        if idtomeshid[str(iid)] in drug_meshid:
            sorted_rank[tp].append((iid, score))
    else:
        sorted_rank[tp].append((iid, score))
        sorted_rank['merged'].append((iid, score))
llen = len(sorted_rank['merged']) 
sorted_rank['merged'] = sorted_rank['merged'][llen * 3 // 4 : ]
print(len(sorted_rank['chemical']))
print(len(sorted_rank['gene']), len(sorted_rank['disease']), len(sorted_rank['merged']))

#%%
Target_node_list = []
Attack_edge_list = []
if args.init_mode == '':

    if args.added_edge_num != '' and args.added_edge_num != '1':
        raise Exception('added_edge_num must be 1 when init_mode=='' ')
    for init_p in [0.1, 0.3, 0.5, 0.7, 0.9]:

        p  = len(sorted_rank['chemical']) * init_p
        print('Init p:', init_p)
        target_node_list = []
        attack_edge_list = []
        num_max_eq = 0
        mean_rank_of_total_max = 0
        for pp in tqdm(range(int(p)-10, int(p)+10)):
            target = sorted_rank['chemical'][pp][0]
            target_node_list.append(target)

            candidate_list = []
            score_list = []
            loss_list = []
            for iid, score in sorted_rank['merged']:
                a = G.number_of_edges(iid, target) + 1
                if a != 1:
                    continue
                b = G.out_degree(iid) + 1
                tp = idtomeshid[str(iid)].split('_')[0]
                edge_losses = []
                r_list = []
                for r in range(len(edgeid_to_edgetype)):
                    r_tp = edgeid_to_edgetype[str(r)]
                    if (edgeid_to_reversemask[str(r)] == 0 and r_tp.split('-')[0] == tp and r_tp.split('-')[1] == 'chemical'):
                        train_trip = np.array([[iid, r, target]])
                        train_trip = torch.from_numpy(train_trip.astype('int64')).to(device)
                        edge_loss = get_model_loss_without_softmax(train_trip, model, device).squeeze()
                        edge_losses.append(edge_loss.unsqueeze(0).detach())
                        r_list.append(r)
                    elif(edgeid_to_reversemask[str(r)] == 1 and r_tp.split('-')[0] == 'chemical' and r_tp.split('-')[1] == tp):
                        train_trip = np.array([[iid, r, target]]) # add batch dim
                        train_trip = torch.from_numpy(train_trip.astype('int64')).to(device)
                        edge_loss = get_model_loss_without_softmax(train_trip, model, device).squeeze()
                        edge_losses.append(edge_loss.unsqueeze(0).detach())
                        r_list.append(r)
                if len(edge_losses)==0:
                    continue
                min_index = torch.argmin(torch.cat(edge_losses, dim = 0))
                r = r_list[min_index]
                r_tp = edgeid_to_edgetype[str(r)]
                
                if (edgeid_to_reversemask[str(r)] == 0):
                    bo, prob = check_reasonable(iid, r, target)
                    if bo:
                        candidate_list.append((iid, r, target))
                        score_list.append(score * a / b)
                        loss_list.append(edge_losses[min_index].item())
                if (edgeid_to_reversemask[str(r)] == 1):
                    bo, prob = check_reasonable(target, r, iid)
                    if bo:
                        candidate_list.append((target, r, iid))
                        score_list.append(score * a / b)
                        loss_list.append(edge_losses[min_index].item())
            
            if len(candidate_list) == 0:
                attack_edge_list.append((-1, -1, -1))
                continue
            norm_score = np.array(score_list) / np.sum(score_list)
            norm_loss = np.exp(-np.array(loss_list)) / np.sum(np.exp(-np.array(loss_list)))

            total_score = norm_score * norm_loss
            max_index = np.argmax(total_score)
            attack_edge_list.append(candidate_list[max_index])

            score_max_index = np.argmax(norm_score)
            if score_max_index == max_index:
                num_max_eq += 1

            score_index_list = list(zip(list(range(len(norm_score))), norm_score))
            score_index_list.sort(key = lambda x: x[1], reverse = True)
            max_index_in_score = score_index_list.index((max_index, norm_score[max_index]))
            mean_rank_of_total_max += max_index_in_score / len(norm_score)
        print('num_max_eq:', num_max_eq)
        print('mean_rank_of_total_max:', mean_rank_of_total_max / 20)
        Target_node_list.append(target_node_list)
        Attack_edge_list.append(attack_edge_list)
else:
    assert args.init_mode == 'random' or args.init_mode == 'single'
    print(f'Init mode : {args.init_mode}')
    utils.seed_all(args.seed)

    if args.init_mode == 'random':
        index = np.random.choice(len(sorted_rank['chemical']), 400, replace = False)
    else:
        # index = [5807, 6314, 5799, 5831, 3954, 5654, 5649, 5624, 2412, 2407]
        
        index = np.random.choice(len(sorted_rank['chemical']), 400, replace = False)
        with open(f'../pagerank/results/After_distmult_0.7random10.pkl', 'rb') as fl:
            edge = pkl.load(fl)
        with open('../pagerank/results/Init_0.7random.pkl', 'rb') as fl:
            init = pkl.load(fl)
        increase = (np.array(init) - np.array(edge)) / np.array(init)
        increase = increase.reshape(-1)
        selected_index = np.argsort(increase)[::-1][:10]
        index = [index[i] for i in selected_index]
    target_node_list = []
    attack_edge_list = []
    num_max_eq = 0
    mean_rank_of_total_max = 0

    for pp in tqdm(index):
        target = sorted_rank['chemical'][pp][0]
        target_node_list.append(target)

        print('Target:', entity_raw_name[idtomeshid[str(target)]])

        candidate_list = []
        score_list = []
        loss_list = []
        main_dict = {}
        for iid, score in sorted_rank['merged']:
            a = G.number_of_edges(iid, target) + 1
            if a != 1:
                continue
            b = G.out_degree(iid) + 1
            tp = idtomeshid[str(iid)].split('_')[0]
            edge_losses = []
            r_list = []
            for r in range(len(edgeid_to_edgetype)):
                r_tp = edgeid_to_edgetype[str(r)]
                if (edgeid_to_reversemask[str(r)] == 0 and r_tp.split('-')[0] == tp and r_tp.split('-')[1] == 'chemical'):
                    train_trip = np.array([[iid, r, target]])
                    train_trip = torch.from_numpy(train_trip.astype('int64')).to(device)
                    edge_loss = get_model_loss_without_softmax(train_trip, model, device).squeeze()
                    edge_losses.append(edge_loss.unsqueeze(0).detach())
                    r_list.append(r)
                elif(edgeid_to_reversemask[str(r)] == 1 and r_tp.split('-')[0] == 'chemical' and r_tp.split('-')[1] == tp):
                    train_trip = np.array([[iid, r, target]]) # add batch dim
                    train_trip = torch.from_numpy(train_trip.astype('int64')).to(device)
                    edge_loss = get_model_loss_without_softmax(train_trip, model, device).squeeze()
                    edge_losses.append(edge_loss.unsqueeze(0).detach())
                    r_list.append(r)
            if len(edge_losses)==0:
                continue
            min_index = torch.argmin(torch.cat(edge_losses, dim = 0))
            r = r_list[min_index]
            r_tp = edgeid_to_edgetype[str(r)]
            

            old_len = len(candidate_list)
            if (edgeid_to_reversemask[str(r)] == 0):
                bo, prob = check_reasonable(iid, r, target)
                if bo:
                    candidate_list.append((iid, r, target))
                    score_list.append(score * a / b)
                    loss_list.append(edge_losses[min_index].item())
            if (edgeid_to_reversemask[str(r)] == 1):
                bo, prob = check_reasonable(target, r, iid)
                if bo:
                    candidate_list.append((target, r, iid))
                    score_list.append(score * a / b)
                    loss_list.append(edge_losses[min_index].item())

            if len(candidate_list) != old_len:
                if int(iid) in main_iid:
                    main_dict[iid] = len(candidate_list) - 1
        
        if len(candidate_list) == 0:
            if args.added_edge_num == '' or int(args.added_edge_num) == 1:
                attack_edge_list.append((-1,-1,-1))
            else:
                attack_edge_list.append([])
            continue
        norm_score = np.array(score_list) / np.sum(score_list)
        norm_loss = np.exp(-np.array(loss_list)) / np.sum(np.exp(-np.array(loss_list)))

        total_score = norm_score * norm_loss
        total_score_index = list(zip(range(len(total_score)), total_score))
        total_score_index.sort(key = lambda x: x[1], reverse = True)

        norm_score_index = np.argsort(norm_score)[::-1]
        norm_loss_index = np.argsort(norm_loss)[::-1]
        total_index = np.argsort(total_score)[::-1]
        assert total_index[0] == total_score_index[0][0]
        
        max_index = np.argmax(total_score)
        assert max_index == total_score_index[0][0]

        tmp_add = []
        add_num = 1
        if args.added_edge_num == '' or int(args.added_edge_num) == 1:
            attack_edge_list.append(candidate_list[max_index])
        else:
            add_num = int(args.added_edge_num)
            for i in range(add_num):
                tmp_add.append(candidate_list[total_score_index[i][0]])
            attack_edge_list.append(tmp_add)

        score_max_index = np.argmax(norm_score)
        if score_max_index == max_index:
            num_max_eq += 1
        score_index_list = list(zip(list(range(len(norm_score))), norm_score))
        score_index_list.sort(key = lambda x: x[1], reverse = True)
        max_index_in_score = score_index_list.index((max_index, norm_score[max_index]))
        mean_rank_of_total_max += max_index_in_score / len(norm_score)
    print('num_max_eq:', num_max_eq)
    print('mean_rank_of_total_max:', mean_rank_of_total_max / 400)
    Target_node_list = target_node_list
    Attack_edge_list = attack_edge_list
print(np.array(Target_node_list).shape)
print(np.array(Attack_edge_list).shape)
with open(f'processed_data/target_{args.reasonable_rate}{args.init_mode}.pkl', 'wb') as fl:
    pkl.dump(Target_node_list, fl)
with open(f'processed_data/attack_edge_{args.model}_{args.reasonable_rate}{args.init_mode}{args.added_edge_num}.pkl', 'wb') as fl:
    pkl.dump(Attack_edge_list, fl)