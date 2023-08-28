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
parser.add_argument('--mode', type = str, default='', help = ' "" or chat or bioBART')
parser.add_argument('--init-mode', type = str, default='random', help = 'How to select target nodes') # 'single' for case study 
parser.add_argument('--added-edge-num', type = str, default = '', help = 'Added edge num')

args = parser.parse_args()
args = utils.set_hyperparams(args)
utils.seed_all(args.seed)
graph_edge_path = '../DiseaseSpecific/processed_data/GNBR/all.txt'
idtomeshid_path = '../DiseaseSpecific/processed_data/GNBR/entities_reverse_dict.json'
model_path = f'../DiseaseSpecific/saved_models/GNBR_{args.model}_128_0.2_0.3_0.3.model'
data_path = '../DiseaseSpecific/processed_data/GNBR'
target_path = f'processed_data/target_{args.reasonable_rate}{args.init_mode}.pkl'
attack_path = f'processed_data/attack_edge_{args.model}_{args.reasonable_rate}{args.init_mode}{args.added_edge_num}{args.mode}.pkl'

with open(Parameters.GNBRfile+'original_entity_raw_name', 'rb') as fl:
    full_entity_raw_name = pkl.load(fl)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")

args.device = device
n_ent, n_rel, ent_to_id, rel_to_id = utils.generate_dicts(data_path)
model = utils.load_model(model_path, args, n_ent, n_rel, args.device)

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

with open(target_path, 'rb') as fl:
    Target_node_list = pkl.load(fl)
with open(attack_path, 'rb') as fl:
    Attack_edge_list = pkl.load(fl)
with open(Parameters.UMLSfile+'drug_term', 'rb') as fl:
    drug_term = pkl.load(fl)
with open(Parameters.GNBRfile+'entity_raw_name', 'rb') as fl:
    entity_raw_name = pkl.load(fl)
drug_meshid = []
for meshid, nm in entity_raw_name.items():
    if nm.lower() in drug_term and meshid.split('_')[0] == 'chemical':
        drug_meshid.append(meshid)
drug_meshid = set(drug_meshid)

if args.init_mode == 'single':
    name_list = []
    for target in Target_node_list:
        name = entity_raw_name[idtomeshid[str(target)]]
        name_list.append(name)
    with open(f'results/name_list_{args.reasonable_rate}{args.init_mode}.txt', 'w') as fl:
        fl.write('\n'.join(name_list))

if args.init_mode == 'single':
    Target_node_list = [[Target_node_list[i]] for i in range(len(Target_node_list))]
    Attack_edge_list = [[Attack_edge_list[i]] for i in range(len(Attack_edge_list))]
else:
    print(len(Attack_edge_list), len(Target_node_list))
    tmp_target_node_list = []
    tmp_attack_edge_list = []
    for l in range(0,len(Target_node_list), 50):
        r = min(l+50, len(Target_node_list))
        tmp_target_node_list.append(Target_node_list[l:r])
        tmp_attack_edge_list.append(Attack_edge_list[l:r])
    Target_node_list = tmp_target_node_list
    Attack_edge_list = tmp_attack_edge_list

# for i, init_p in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):

#     target_node_list = Target_node_list[i]
#     attack_edge_list = Attack_edge_list[i]
Init = []
After = []
# final_init = []
# final_after = []
for i, (target_node_list, attack_edge_list) in enumerate(zip(Target_node_list, Attack_edge_list)):

    G = nx.DiGraph()
    for s, r, o in graph_edge:
        assert idtomeshid[s].split('_')[0] == edgeid_to_edgetype[r].split('-')[0]
        if edgeid_to_reversemask[r] == 1:
            G.add_edge(int(o), int(s))
        else:
            G.add_edge(int(s), int(o))

    pagerank_value_1 = nx.pagerank(G, max_iter = 200, tol=1.0e-7) 
    
    for target, attack_list in tqdm(list(zip(target_node_list, attack_edge_list))):
        pr = list(pagerank_value_1.items())
        pr.sort(key = lambda x: x[1])
        list_iid = []
        for iid, score in pr:
            tp = idtomeshid[str(iid)].split('_')[0]
            if tp == 'chemical':
                # if idtomeshid[str(iid)] in drug_meshid:
                list_iid.append(iid)
        init_rank = len(list_iid) - list_iid.index(target)
        # init_rank = 1 - list_iid.index(target) / len(list_iid)
        Init.append(init_rank)

    for target, attack_list in tqdm(list(zip(target_node_list, attack_edge_list))):
        
        if args.mode == '' and (args.added_edge_num == '' or int(args.added_edge_num) == 1):
            if int(attack_list[0]) == -1:
                attack_list = []
            else:
                attack_list = [attack_list]
        if len(attack_list) > 0:
            for s, r, o in attack_list:
                bo, prob = check_reasonable(s, r, o)
                if bo:
                    if edgeid_to_reversemask[str(r)] == 1:
                        G.add_edge(int(o), int(s))
                    else:
                        G.add_edge(int(s), int(o))
    pagerank_value_1 = nx.pagerank(G, max_iter = 200, tol=1.0e-7) 
    for target, attack_list in tqdm(list(zip(target_node_list, attack_edge_list))):
        pr = list(pagerank_value_1.items())
        pr.sort(key = lambda x: x[1])
        list_iid = []
        for iid, score in pr:
            tp = idtomeshid[str(iid)].split('_')[0]
            if tp == 'chemical':
                # if idtomeshid[str(iid)] in drug_meshid:
                list_iid.append(iid)
        after_rank = len(list_iid) - list_iid.index(target)
        # after_rank = 1 - list_iid.index(target) / len(list_iid)
        After.append(after_rank)
    with open(f'results/Init_{args.reasonable_rate}{args.init_mode}.pkl', 'wb') as fl:
        pkl.dump(Init, fl)
    with open(f'results/After_{args.model}_{args.reasonable_rate}{args.init_mode}{args.added_edge_num}{args.mode}.pkl', 'wb') as fl:
        pkl.dump(After, fl)
    print(np.mean(Init), np.std(Init))
    print(np.mean(After), np.std(After))
