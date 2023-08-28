#%%
import torch
import numpy as np
from torch.autograd import Variable
from sklearn import metrics

import datetime
from typing import Dict, Tuple, List
import logging
import os
import utils
import pickle as pkl
import json 
import torch.backends.cudnn as cudnn

from tqdm import tqdm

import sys
sys.path.append("..")
import Parameters

parser = utils.get_argument_parser()
parser = utils.add_attack_parameters(parser)
parser.add_argument('--mode', type=str, default='sentence', help='sentence, finetune, biogpt, bioBART')
parser.add_argument('--action', type=str, default='parse', help='parse or extract')
parser.add_argument('--ratio', type = str, default='', help='ratio of the number of changed words')
args = parser.parse_args()
args = utils.set_hyperparams(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
utils.seed_all(args.seed)
np.set_printoptions(precision=5)
cudnn.benchmark = False

data_path = os.path.join('processed_data', args.data)
target_path = os.path.join(data_path, 'DD_target_{0}_{1}_{2}_{3}_{4}_{5}.txt'.format(args.model, args.data, args.target_split, args.target_size, 'exists:'+str(args.target_existed), args.attack_goal))
attack_path = os.path.join('attack_results', args.data, 'cos_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.txt'.format(args.model, 
                                                        args.target_split, 
                                                        args.target_size, 
                                                        'exists:'+str(args.target_existed),
                                                        args.neighbor_num,
                                                        args.candidate_mode,
                                                        args.attack_goal,
                                                        str(args.reasonable_rate)))
modified_attack_path = os.path.join('attack_results', args.data, 'cos_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}{8}.txt'.format(args.model, 
                                                        args.target_split, 
                                                        args.target_size, 
                                                        'exists:'+str(args.target_existed),
                                                        args.neighbor_num,
                                                        args.candidate_mode,
                                                        args.attack_goal,
                                                        str(args.reasonable_rate),
                                                        args.mode))
attack_data = utils.load_data(attack_path, drop=False)
#%%
with open(os.path.join(data_path, 'entities_reverse_dict.json')) as fl:
    id_to_meshid = json.load(fl)
with open(os.path.join(data_path, 'entities_dict.json'), 'r') as fl:
    meshid_to_id = json.load(fl)
with open(Parameters.GNBRfile+'entity_raw_name', 'rb') as fl:
    entity_raw_name = pkl.load(fl)
with open(Parameters.GNBRfile+'retieve_sentence_through_edgetype', 'rb') as fl:
    retieve_sentence_through_edgetype = pkl.load(fl)
with open(Parameters.GNBRfile+'raw_text_of_each_sentence', 'rb') as fl:
    raw_text_sen = pkl.load(fl)
with open(Parameters.GNBRfile+'original_entity_raw_name', 'rb') as fl:
    full_entity_raw_name = pkl.load(fl)
for k, v in entity_raw_name.items():
    assert v in full_entity_raw_name[k]

#find unique
once_set = set()
twice_set = set()

with open('generate_abstract/valid_entity.json', 'r') as fl:
    valid_entity = json.load(fl)
valid_entity = set(valid_entity)

good_name = set()
for k, v, in full_entity_raw_name.items():
    names = list(v)
    for name in names:
        # if name == 'in a':
        #     print(names)
        good_name.add(name)
        # if name not in once_set:
        #     once_set.add(name)
        # else:
        #     twice_set.add(name)
# assert 'WNK4' in once_set
# good_name = set.difference(once_set, twice_set)
# assert 'in a' not in good_name
# assert 'STE20' not in good_name
# assert 'STE20' not in valid_entity
# assert 'STE20-related proline-alanine-rich kinase' not in good_name
# assert 'STE20-related proline-alanine-rich kinase' not in valid_entity
# raise Exception

name_to_type = {}
name_to_meshid = {}

for k, v, in full_entity_raw_name.items():
    names = list(v)
    for name in names:
        if name in good_name:
            name_to_type[name] = k.split('_')[0]
            name_to_meshid[name] = k

import spacy
import networkx as nx
import pprint

def check(p, s):

    if p < 1 or p >= len(s):
        return True
    return not((s[p]>='a' and s[p]<='z') or (s[p]>='A' and s[p]<='Z') or (s[p]>='0' and s[p]<='9'))

def raw_to_format(sen):

    text = sen
    l = 0
    ret = []
    while(l < len(text)):
        bo =False
        if text[l] != ' ':
            for i in range(len(text), l, -1): # reversing is important !!!
                cc = text[l:i]
                if (cc in good_name or cc in valid_entity) and check(l-1, text) and check(i, text):
                    ret.append(cc.replace(' ', '_'))
                    l = i
                    bo = True
                    break
        if not bo:
            ret.append(text[l])
            l += 1
    return ''.join(ret)

if args.mode == 'sentence':
    with open(f'generate_abstract/{args.target_split}_{args.reasonable_rate}_chat.json', 'r') as fl:
        draft = json.load(fl)
elif args.mode == 'finetune':
    with open(f'generate_abstract/{args.target_split}_{args.reasonable_rate}_sentence_finetune.json', 'r') as fl:
        draft = json.load(fl)
elif args.mode == 'bioBART':
    with open(f'generate_abstract/{args.target_split}_{args.reasonable_rate}{args.ratio}_bioBART_finetune.json', 'r') as fl:
        draft = json.load(fl)
elif args.mode == 'biogpt':
    with open(f'generate_abstract/{args.target_split}_{args.reasonable_rate}_biogpt.json', 'r') as fl:
        draft = json.load(fl)
else:
    raise Exception('No!!!')

nlp = spacy.load("en_core_web_sm")

type_set = set()
for aa in range(36):
    dependency_sen_dict = retieve_sentence_through_edgetype[aa]['manual']
    tmp_dict = retieve_sentence_through_edgetype[aa]['auto']
    dependencys = list(dependency_sen_dict.keys()) + list(tmp_dict.keys())
    for dependency in dependencys:
        dep_list = dependency.split(' ')
        for sub_dep in dep_list:
            sub_dep_list = sub_dep.split('|')
            assert(len(sub_dep_list) == 3)
            type_set.add(sub_dep_list[1])
# print('Type:', type_set)

if args.action == 'parse':
# dp_path, sen_list = list(dependency_sen_dict.items())[0]
# check
# paper_id, sen_id = sen_list[0]
# sen = raw_text_sen[paper_id][sen_id]
# doc = nlp(sen['text'])
# print(dp_path, '\n')
# pprint.pprint(sen)
# print()
# for token in doc:
#     print((token.head.text, token.text, token.dep_))

    out = ''
    for k, v_dict in draft.items():
        input = v_dict['in']
        output = v_dict['out']
        if input == '':
            continue
        output = output.replace('\n', ' ')
        doc = nlp(output)
        for sen in doc.sents:
            out += raw_to_format(sen.text) + '\n'
    with open(f'generate_abstract/{args.target_split}_{args.reasonable_rate}_{args.mode}_parsein.txt', 'w') as fl:
        fl.write(out)
elif args.action == 'extract':

    # dependency_to_type_id = {}
    # for k, v in Parameters.edge_type_to_id.items():
    #     dependency_to_type_id[k] = {}
    #     for type in v:
    #         LL = list(retieve_sentence_through_edgetype[type]['manual'].keys()) + list(retieve_sentence_through_edgetype[type]['auto'].keys())
    #         for dp in LL:
    #             dependency_to_type_id[k][dp] = type
    if os.path.exists('generate_abstract/dependency_to_type_id.pickle'):
        with open('generate_abstract/dependency_to_type_id.pickle', 'rb') as fl:
            dependency_to_type_id = pkl.load(fl)
    else:
        dependency_to_type_id = {}
        print('Loading path data ...')
        for k in Parameters.edge_type_to_id.keys():
            start, end = k.split('-')
            dependency_to_type_id[k] = {}
            inner_edge_type_to_id = Parameters.edge_type_to_id[k]
            inner_edge_type_dict = Parameters.edge_type_dict[k]
            cal_manual_num = [0] * len(inner_edge_type_to_id)
            with open('../GNBRdata/part-i-'+start+'-'+end+'-path-theme-distributions.txt', 'r') as fl:
                for i, line in tqdm(list(enumerate(fl.readlines()))):
                    tmp = line.split('\t')
                    if i == 0:
                        head = [tmp[i] for i in range(1, len(tmp), 2)]
                        assert ' '.join(head) == ' '.join(inner_edge_type_dict[0])
                        continue
                    probability = [float(tmp[i]) for i in range(1, len(tmp), 2)]
                    flag_list = [int(tmp[i]) for i in range(2, len(tmp), 2)]
                    indices = np.where(np.asarray(flag_list) == 1)[0]
                    if len(indices) >= 1:
                        tmp_p = [cal_manual_num[i] for i in indices]
                        p = indices[np.argmin(tmp_p)]
                        cal_manual_num[p] += 1
                    else:
                        p = np.argmax(probability)
                    assert tmp[0].lower() not in dependency_to_type_id.keys()
                    dependency_to_type_id[k][tmp[0].lower()] = inner_edge_type_to_id[p]
        with open('generate_abstract/dependency_to_type_id.pickle', 'wb') as fl:
            pkl.dump(dependency_to_type_id, fl)
    
    # record = []
    # with open(f'generate_abstract/par_parseout.txt', 'r') as fl:
    #     Tmp = []
    #     tmp = []
    #     for i,line in enumerate(fl.readlines()):
    #         # print(len(line), line)
    #         line = line.replace('\n', '')
    #         if len(line) > 1:
    #             tmp.append(line)
    #         else:
    #             Tmp.append(tmp)
    #             tmp = []
    #         if len(Tmp) == 3:
    #             record.append(Tmp)
    #             Tmp = []

    # print(len(record))
    # record_index = 0 
    # add = 0
    # Attack = []
    # for ii in range(100):

    #     # input = v_dict['in']
    #     # output = v_dict['out']
    #     # output = output.replace('\n', ' ')
    #     s, r, o = attack_data[ii]
    #     dependency_sen_dict = retieve_sentence_through_edgetype[int(r)]['manual']
        
    #     target_dp = set()
    #     for dp_path, sen_list in dependency_sen_dict.items():
    #         target_dp.add(dp_path)
    #     DP_list = []
    #     for _ in range(1):
    #         dp_dict = {}
    #         data = record[record_index]
    #         record_index += 1
    #         dp_paths = data[2]
    #         nodes_list = []
    #         edges_list = []
    #         for line in dp_paths:
    #             ttp, tmp = line.split('(')
    #             assert tmp[-1] == ')'
    #             tmp = tmp[:-1]
    #             e1, e2 = tmp.split(', ')
    #             if not ttp in type_set and ':' in ttp:
    #                 ttp = ttp.split(':')[0]
    #             dp_dict[f'{e1}_x_{e2}'] = [e1, ttp, e2]
    #             dp_dict[f'{e2}_x_{e1}'] = [e1, ttp, e2]
    #             nodes_list.append(e1)
    #             nodes_list.append(e2)
    #             edges_list.append((e1, e2))
    #         nodes_list = list(set(nodes_list))
    #         pure_name = [('-'.join(name.split('-')[:-1])).replace('_', ' ') for name in nodes_list]
    #         graph = nx.Graph(edges_list)

    #         type_list = [name_to_type[name] if name in good_name else '' for name in pure_name]
    #         # print(type_list)
    #         # for i in range(len(type_list)):
    #         #     print(pure_name[i], type_list[i])
    #         for i in range(len(nodes_list)):
    #             if type_list[i] != '':
    #                 for j in range(len(nodes_list)):
    #                     if i != j and type_list[j] != '':
    #                         if f'{type_list[i]}-{type_list[j]}' in Parameters.edge_type_to_id.keys():
    #                             # print(f'{type_list[i]}_{type_list[j]}')
    #                             ret_path = []
    #                             sp = nx.shortest_path(graph, source=nodes_list[i], target=nodes_list[j])
    #                             start = sp[0]
    #                             end = sp[-1]
    #                             for k in range(len(sp)-1):
    #                                 e1, ttp, e2 = dp_dict[f'{sp[k]}_x_{sp[k+1]}']
    #                                 if e1 == start:
    #                                     e1 = 'start_entity-x'
    #                                 if e2 == start:
    #                                     e2 = 'start_entity-x'
    #                                 if e1 == end:
    #                                     e1 = 'end_entity-x'
    #                                 if e2 == end:
    #                                     e2 = 'end_entity-x'
    #                                 ret_path.append(f'{"-".join(e1.split("-")[:-1])}|{ttp}|{"-".join(e2.split("-")[:-1])}'.lower())
    #                             dependency_P = ' '.join(ret_path)
    #                             DP_list.append((f'{type_list[i]}-{type_list[j]}', 
    #                                             name_to_meshid[pure_name[i]], 
    #                                             name_to_meshid[pure_name[j]], 
    #                                             dependency_P))
        
    #     boo = False
    #     modified_attack = []
    #     for k, ss, tt, dp in DP_list:
    #         if dp in dependency_to_type_id[k].keys():
    #             tp = str(dependency_to_type_id[k][dp])
    #             id_ss = str(meshid_to_id[ss])
    #             id_tt = str(meshid_to_id[tt])
    #             modified_attack.append(f'{id_ss}*{tp}*{id_tt}')
    #             if int(dependency_to_type_id[k][dp]) == int(r):
    #                 # if id_to_meshid[s] == ss and id_to_meshid[o] == tt:
    #                 boo = True
    #     modified_attack = list(set(modified_attack))
    #     modified_attack = [k.split('*') for k in modified_attack]
    #     if boo:
    #         add += 1
    #     # else:
    #         # print(ii)
            
    #         # for i in range(len(type_list)):
    #         #     if type_list[i]:
    #         #         print(pure_name[i], type_list[i])
    #         # for k, ss, tt, dp in DP_list:
    #         #     print(k, dp)
    #         # print(record[record_index - 1])
    #         # raise Exception('No!!')
    #     Attack.append(modified_attack)

    record = []
    with open(f'generate_abstract/{args.target_split}_{args.reasonable_rate}_{args.mode}_parseout.txt', 'r') as fl:
        Tmp = []
        tmp = []
        for i,line in enumerate(fl.readlines()):
            # print(len(line), line)
            line = line.replace('\n', '')
            if len(line) > 1:
                tmp.append(line)
            else:
                if len(Tmp) == 2:
                    if len(tmp) == 1 and '/' in tmp[0].split(' ')[0]:
                        Tmp.append([])
                        record.append(Tmp)
                        Tmp = []
                Tmp.append(tmp)
                if len(Tmp) == 2 and tmp[0][:5] != '(ROOT':
                    print(record[-1][2])
                    raise Exception('??')
                tmp = []
            if len(Tmp) == 3:
                record.append(Tmp)
                Tmp = []
    with open(f'generate_abstract/{args.target_split}_{args.reasonable_rate}_{args.mode}_parsein.txt', 'r') as fl:
        parsin = fl.readlines()

    print('Record len', len(record), 'Parsin len:', len(parsin))
    record_index = 0 
    add = 0

    Attack = []
    for ii, (k, v_dict) in enumerate(tqdm(draft.items())):

        input = v_dict['in']
        output = v_dict['out']
        output = output.replace('\n', ' ')
        s, r, o = attack_data[ii]
        assert ii == int(k.split('_')[-1])
        
        DP_list = []
        if input != '':

            dependency_sen_dict = retieve_sentence_through_edgetype[int(r)]['manual']
            target_dp = set()
            for dp_path, sen_list in dependency_sen_dict.items():
                target_dp.add(dp_path)
            doc = nlp(output)
            
            for sen in doc.sents:
                dp_dict = {}
                if record_index >= len(record):
                    break
                data = record[record_index]
                record_index += 1
                dp_paths = data[2]
                nodes_list = []
                edges_list = []
                for line in dp_paths:
                    aa = line.split('(')
                    if len(aa) == 1:
                        print(ii)
                        print(sen)
                        print(data)
                        raise Exception
                    ttp, tmp = aa[0], aa[1]
                    assert tmp[-1] == ')'
                    tmp = tmp[:-1]
                    e1, e2 = tmp.split(', ')
                    if not ttp in type_set and ':' in ttp:
                        ttp = ttp.split(':')[0]
                    dp_dict[f'{e1}_x_{e2}'] = [e1, ttp, e2]
                    dp_dict[f'{e2}_x_{e1}'] = [e1, ttp, e2]
                    nodes_list.append(e1)
                    nodes_list.append(e2)
                    edges_list.append((e1, e2))
                nodes_list = list(set(nodes_list))
                pure_name = [('-'.join(name.split('-')[:-1])).replace('_', ' ') for name in nodes_list]
                graph = nx.Graph(edges_list)

                type_list = [name_to_type[name] if name in good_name else '' for name in pure_name]
                # print(type_list)
                for i in range(len(nodes_list)):
                    if type_list[i] != '':
                        for j in range(len(nodes_list)):
                            if i != j and type_list[j] != '':
                                if f'{type_list[i]}-{type_list[j]}' in Parameters.edge_type_to_id.keys():
                                    # print(f'{type_list[i]}_{type_list[j]}')
                                    ret_path = []
                                    sp = nx.shortest_path(graph, source=nodes_list[i], target=nodes_list[j])
                                    start = sp[0]
                                    end = sp[-1]
                                    for k in range(len(sp)-1):
                                        e1, ttp, e2 = dp_dict[f'{sp[k]}_x_{sp[k+1]}']
                                        if e1 == start:
                                            e1 = 'start_entity-x'
                                        if e2 == start:
                                            e2 = 'start_entity-x'
                                        if e1 == end:
                                            e1 = 'end_entity-x'
                                        if e2 == end:
                                            e2 = 'end_entity-x'
                                        ret_path.append(f'{"-".join(e1.split("-")[:-1])}|{ttp}|{"-".join(e2.split("-")[:-1])}'.lower())
                                    dependency_P = ' '.join(ret_path)
                                    DP_list.append((f'{type_list[i]}-{type_list[j]}', 
                                                    name_to_meshid[pure_name[i]], 
                                                    name_to_meshid[pure_name[j]], 
                                                    dependency_P))
        
        boo = False
        modified_attack = []
        for k, ss, tt, dp in DP_list:
            if dp in dependency_to_type_id[k].keys():
                tp = str(dependency_to_type_id[k][dp])
                id_ss = str(meshid_to_id[ss])
                id_tt = str(meshid_to_id[tt])
                modified_attack.append(f'{id_ss}*{tp}*{id_tt}')
                if int(dependency_to_type_id[k][dp]) == int(r):
                    if id_to_meshid[s] == ss and id_to_meshid[o] == tt:
                        boo = True
        modified_attack = list(set(modified_attack))
        modified_attack = [k.split('*') for k in modified_attack]
        if boo:
            # print(DP_list)
            add += 1
        Attack.append(modified_attack)
    print(add)
    print('End record_index:', record_index)
    with open(modified_attack_path, 'wb') as fl:
        pkl.dump(Attack, fl)
else:
    raise Exception('Wrong action !!')