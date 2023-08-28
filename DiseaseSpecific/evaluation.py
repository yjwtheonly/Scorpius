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
from tqdm import tqdm
import torch.backends.cudnn as cudnn

import sys
sys.path.append("..")
import Parameters

logger = logging.getLogger(__name__)

def get_model_loss_without_softmax(batch, model, device=None):

    with torch.no_grad():
        s,r,o = batch[:,0], batch[:,1], batch[:,2]

        emb_s = model.emb_e(s).squeeze(dim=1)
        emb_r = model.emb_rel(r).squeeze(dim=1)

        pred = model.forward(emb_s, emb_r)
        return -pred[range(o.shape[0]), o]

def check(trip, model, reasonable_rate, device, data_mean = -4.008113861083984, data_std = 5.153779983520508, divide_bound = 0.05440050354114886):

    if args.model == 'distmult':
        pass
    elif args.model == 'conve':
        data_mean = 13.890259742
        data_std = 12.396190643
        divide_bound = -0.1986345871
    else:
        raise Exception('Wrong model!!')
    trip = np.array(trip)
    train_trip = trip[None, :] 
    train_trip = torch.from_numpy(train_trip.astype('int64')).to(device)
    edge_loss = get_model_loss_without_softmax(train_trip, model, device).squeeze().item()

    bound = 1 - reasonable_rate
    edge_loss = (edge_loss - data_mean) / data_std
    edge_loss_prob =  1 / ( 1 + np.exp(edge_loss - divide_bound))
    return edge_loss_prob > bound


def get_ranking(model, queries,
                valid_filters:Dict[str, Dict[Tuple[str, int], torch.Tensor]],
                device, batch_size, entityid_to_nodetype, exists_edge):
    """
    Ranking for target generation.
    """
    ranks = []
    total_nums = []
    b_begin = 0

    for b_begin in range(0, len(queries), 1):
        b_queries = queries[b_begin : b_begin+1]
        s,r,o = b_queries[:,0], b_queries[:,1], b_queries[:,2]
        r_rev = r
        lhs_score = model.score_or(o, r_rev, sigmoid=False) #this gives scores not probabilities
        # print(b_queries.shape)
        for i, query in enumerate(b_queries):

            if not args.target_existed:
                tp1 = entityid_to_nodetype[str(query[0].item())]
                tp2 = entityid_to_nodetype[str(query[2].item())]
                filter = valid_filters['lhs'][(tp2, query[1].item())].clone()
                filter[exists_edge['lhs'][str(query[2].item())]] = False
                filter = (filter == False)
            else:
                tp1 = entityid_to_nodetype[str(query[0].item())]
                tp2 = entityid_to_nodetype[str(query[2].item())]
                filter = valid_filters['lhs'][(tp2, query[1].item())]
                filter = (filter == False)
            
            # if (str(query[2].item())) == '16566':
            #     print('16566', filter.sum(), valid_filters['lhs'][(tp2, query[1].item())].sum(), tp2, query[1].item())
            #     raise Exception('??')

            score = lhs_score
            #     target_value = rhs_score[i, query[0].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            score[i][filter] = 1e6
            total_nums.append(n_ent - filter.sum().item())
            # write base the saved values
            # if b_begin < len(queries) // 2:
            #     score[i][query[2].item()] = target_value
            # else:
            #     score[i][query[0].item()] = target_value

        # sort and rank
        min_values, sort_v  = torch.sort(score, dim=1, descending=False) #low scores get low number ranks

        sort_v = sort_v.cpu().numpy()
        
        for i, query in enumerate(b_queries):
            # find the rank of the target entities
            rank = np.where(sort_v[i]==query[0].item())[0][0]

            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks.append(rank)

    #logger.info('Ranking done for all queries')
    return ranks, total_nums
    
    
def evaluation(model, queries,
                valid_filters:Dict[str, Dict[Tuple[str, int], torch.Tensor]],
                device, batch_size, entityid_to_nodetype, exists_edge, eval_type = '', attack_data = None, ori_ranks = None, ori_totals = None):
    
    #get ranking
    ranks, total_nums = get_ranking(model, queries, valid_filters, device, batch_size, entityid_to_nodetype, exists_edge)
    ranks, total_nums = np.array(ranks), np.array(total_nums)
    # print(ranks)
    # print(total_nums)
    # print(ranks)
    # print(total_nums)

    ranks = total_nums - ranks

    if (attack_data is not None):
        for i, tri in enumerate(attack_data):
            if args.mode == '':
                if args.added_edge_num == '' or int(args.added_edge_num) == 1:
                    if int(tri[0]) == -1:
                        ranks[i] = ori_ranks[i]
                        total_nums[i] = ori_totals[i]
                else:
                    if int(tri[0][0]) == -1:
                        ranks[i] = ori_ranks[i]
                        total_nums[i] = ori_totals[i]
            else:
                if len(tri) == 0:
                    ranks[i] = ori_ranks[i]
                    total_nums[i] = ori_totals[i]

    mean = (ranks / total_nums).mean()
    std = (ranks / total_nums).std()
    #final logging
    hits_at = np.arange(1,11)
    hits_at_both = list(map(lambda x: np.mean((ranks <= x), dtype=np.float64).item(), 
                                      hits_at))
    mr = np.mean(ranks, dtype=np.float64).item()
    
    mrr = np.mean(1. / ranks, dtype=np.float64).item()
    
    logger.info('')
    logger.info('-'*50)
    # logger.info(split+'_'+save_name)
    logger.info('')
    if eval_type:
        logger.info(eval_type)
    else:
        logger.info('after attck')

    for i in hits_at:
        logger.info('Hits @{0}: {1}'.format(i, hits_at_both[i-1]))
    logger.info('Mean rank: {0}'.format( mr))
    logger.info('Mean reciprocal rank lhs: {0}'.format(mrr))
    logger.info('Mean proportion: {0}'.format(mean))
    logger.info('Std proportion: {0}'.format(std))
    logger.info('Mean candidate num: {0}'.format(np.mean(total_nums)))
    
#     with open(os.path.join('results', split + '_' + save_name + '.txt'), 'a') as text_file:
#         text_file.write('Epoch: {0}\n'.format(epoch))
#         text_file.write('Lhs denotes ranking by subject corruptions \n')
#         text_file.write('Rhs denotes ranking by object corruptions \n')
#         for i in hits_at:
#             text_file.write('Hits left @{0}: {1}\n'.format(i, hits_at_lhs[i-1]))
#             text_file.write('Hits right @{0}: {1}\n'.format(i, hits_at_rhs[i-1]))
#             text_file.write('Hits @{0}: {1}\n'.format(i, np.mean([hits_at_lhs[i-1],hits_at_rhs[i-1]]).item()))
#         text_file.write('Mean rank lhs: {0}\n'.format( mr_lhs))
#         text_file.write('Mean rank rhs: {0}\n'.format(mr_rhs))
#         text_file.write('Mean rank: {0}\n'.format( np.mean([mr_lhs, mr_rhs])))
#         text_file.write('MRR lhs: {0}\n'.format( mrr_lhs))
#         text_file.write('MRR rhs: {0}\n'.format(mrr_rhs))
#         text_file.write('MRR: {0}\n'.format(np.mean([mrr_rhs, mrr_lhs])))
#         text_file.write('-------------------------------------------------\n')
        
        
    results = {}
    for i in hits_at:
        results['hits @{}'.format(i)] = hits_at_both[i-1]
    results['mrr'] = mrr
    results['mr'] = mr
    results['proportion'] = mean
    results['std'] = std
    
    return results, list(ranks), list(total_nums)


parser = utils.get_argument_parser()
parser = utils.add_attack_parameters(parser)
parser = utils.add_eval_parameters(parser)
args = parser.parse_args()
args = utils.set_hyperparams(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
utils.seed_all(args.seed)
np.set_printoptions(precision=5)
cudnn.benchmark = False
   
data_path = os.path.join('processed_data', args.data)
target_path = os.path.join(data_path, 'DD_target_{0}_{1}_{2}_{3}_{4}_{5}.txt'.format(args.model, args.data, args.target_split, args.target_size, 'exists:'+str(args.target_existed), args.attack_goal))

log_path = 'logs/evaluation_logs/cos_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}{8}'.format(args.model, 
                                                            args.target_split, 
                                                            args.target_size, 
                                                            'exists:'+str(args.target_existed),
                                                            args.neighbor_num,
                                                            args.candidate_mode,
                                                            args.attack_goal,
                                                            str(args.reasonable_rate),
                                                            args.mode)
record_path = 'eval_record/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}{8}{9}{10}'.format(args.model, 
                                                            args.target_split, 
                                                            args.target_size, 
                                                            'exists:'+str(args.target_existed),
                                                            args.neighbor_num,
                                                            args.candidate_mode,
                                                            args.attack_goal,
                                                            str(args.reasonable_rate),
                                                            args.mode,
                                                            str(args.added_edge_num),
                                                            args.mask_ratio)
init_record_path = 'eval_record/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}{8}'.format(args.model, 
                                                            args.target_split, 
                                                            args.target_size, 
                                                            'exists:'+str(args.target_existed),
                                                            args.neighbor_num,
                                                            args.candidate_mode,
                                                            args.attack_goal,
                                                            str(args.reasonable_rate),
                                                            'init')

if args.seperate:
    record_path += '_seperate'
    log_path += '_seperate'
else:
    record_path += '_batch'

if args.direct:
    log_path += '_direct'
    record_path += '_direct'
else:
    log_path += '_nodirect'
    record_path += '_nodirect'

dis_turbrbed_path_pre = os.path.join(data_path, 'evaluation')
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO,
                            filename = log_path
                           )
logger = logging.getLogger(__name__)
logger.info(vars(args))

n_ent, n_rel, ent_to_id, rel_to_id = utils.generate_dicts(data_path)
model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)
model = utils.load_model(model_path, args, n_ent, n_rel, device)

ori_data  = utils.load_data(os.path.join(data_path, 'all.txt'))
target_data = utils.load_data(target_path)

index = range(len(target_data))
index = np.random.permutation(index)
target_data = target_data[index]

if args.direct:
    assert args.attack_goal == 'single'
    raise Exception('This option is abandoned in this version .')
    # disturbed_data = list(ori_data) + list(target_data)
else:
    
    attack_path = os.path.join('attack_results', args.data, 'cos_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}{8}{9}{10}.txt'.format(args.model, 
                                                            args.target_split, 
                                                            args.target_size, 
                                                            'exists:'+str(args.target_existed),
                                                            args.neighbor_num,
                                                            args.candidate_mode,
                                                            args.attack_goal,
                                                            str(args.reasonable_rate),
                                                            args.mode,
                                                            str(args.added_edge_num),
                                                            args.mask_ratio))
    if args.mode == '':
        attack_data = utils.load_data(attack_path, drop=False)
        if not(args.added_edge_num == '' or int(args.added_edge_num) == 1):
            assert int(args.added_edge_num) * len(target_data) == len(attack_data)
            attack_data = attack_data.reshape((len(target_data), int(args.added_edge_num), 3))
            attack_data = attack_data[index]
        else:
            assert len(target_data) == len(attack_data)
            attack_data = attack_data[index]
        # if not args.seperate:
        #     disturbed_data = list(ori_data) + list(attack_data)
    else:
        with open(attack_path, 'rb') as fl:
            attack_data = pkl.load(fl)

        tmp_attack_data = []
        for vv in attack_data:
            a_attack = []
            for v in vv:
                if check(v, model, args.reasonable_rate, device):
                     a_attack.append(v)
            tmp_attack_data.append(a_attack)
        attack_data = tmp_attack_data
        attack_data = [attack_data[i] for i in index]

        # if not args.seperate:
        #     disturbed_data = list(ori_data)
        #     if args.mode == '':
        #         for aa in list(attack_data):
        #             if int(aa[0]) != -1:
        #                 disturbed_data.append(aa)
        #     else:
        #         for vv in attack_data:
        #             for v in vv:
        #                 disturbed_data.append(v)

with open(os.path.join(data_path, 'filter.pickle'), 'rb') as fl:
    valid_filters = pkl.load(fl)
with open(os.path.join(data_path, 'entityid_to_nodetype.json'), 'r') as fl:
    entityid_to_nodetype = json.load(fl)
with open(Parameters.GNBRfile+'entity_raw_name', 'rb') as fl:
    entity_raw_name = pkl.load(fl)
with open(os.path.join(data_path, 'disease_meshid.pickle'), 'rb') as fl:
    disease_meshid = pkl.load(fl)
with open(os.path.join(data_path, 'entities_dict.json'), 'r') as fl:
    entity_to_id = json.load(fl)

if args.attack_goal == 'global':
    raise Exception('Please refer to pagerank method in global setting.')
    # target_disease = []
    # tid = 1
    # bound = 50
    # while True:
    #     meshid = disease_meshid[tid][0]
    #     fre = disease_meshid[tid][1]
    #     if len(entity_raw_name[meshid]) > 4:
    #         target_disease.append(entity_to_id[meshid])
    #         bound -= 1
    #         if bound == 0:
    #             break
    #     tid += 1
    # s_set = set()
    # for s, r, o in target_data:
    #     s_set.add(s)
    # target_data = list(s_set)
    # target_data.sort()

    # target_list = []
    # for s in target_data:
    #     for o in target_disease:
    #         target_list.append([str(s), str(10), str(o)])
    # target_data = np.array(target_list, dtype = str)

init_mask = np.asarray([0] * n_ent).astype('int64')
init_mask = (init_mask == 1)
for k, v in valid_filters.items():
    for kk, vv in v.items():
        tmp = init_mask.copy()
        tmp[np.asarray(vv)] = True
        t = torch.ByteTensor(tmp).to(device)
        valid_filters[k][kk] = t
# print('what??', valid_filters['lhs'][('disease', 10)].sum())

exists_edge = {'lhs':{}, 'rhs':{}}
for s, r, o in ori_data:
    if s not in exists_edge['rhs'].keys():
        exists_edge['rhs'][s] = []
    if o not in exists_edge['lhs'].keys():
        exists_edge['lhs'][o] = []
    exists_edge['rhs'][s].append(int(o))
    exists_edge['lhs'][o].append(int(s))
target_data = torch.from_numpy(target_data.astype('int64')).to(device)
# print(target_data[:5, :])
ori_results, ori_ranks, ori_totals = evaluation(model, target_data, valid_filters, device, args.test_batch_size, entityid_to_nodetype, exists_edge, 'original')
print('Original:', ori_results)
with open(init_record_path, 'wb') as fl:
    pkl.dump([ori_results, ori_ranks, ori_totals], fl)

# raise Exception('Check Original Rank!!!')

thread_name = args.model+'_'+args.target_split+'_'+args.attack_goal+'_'+str(args.reasonable_rate)+str(args.added_edge_num)+str(args.mask_ratio)
if args.direct:
    thread_name += '_direct'
else:
    thread_name += '_nodirect'
if args.seperate:
    thread_name += '_seperate'
else:
    thread_name += '_batch'
thread_name += args.mode

disturbed_data_path = os.path.join(dis_turbrbed_path_pre, 'all_{}.txt'.format(thread_name))

if args.seperate:
    # assert len(attack_data) * len(target_disease) == len(target_data)
    assert len(attack_data) == len(target_data)
    # final_result = None
    Ranks = []
    Totals = []
    print('Training model {}...'.format(thread_name))
    for i in tqdm(range(len(attack_data))):
        attack_trip = attack_data[i]
        if args.mode == '':
            attack_trip = [attack_trip]
        # target = target_data[i*len(target_disease) : (i+1)*len(target_disease)]
        target = target_data[i: i+1, :]
        if len(attack_trip) > 0 and int(attack_trip[0][0]) != -1:
            disturbed_data = list(ori_data) + attack_trip
            disturbed_data = np.array(disturbed_data)
            utils.save_data(disturbed_data_path, disturbed_data)

            cmd = 'CUDA_VISIBLE_DEVICES={} python main_multiprocess.py --data {} --model {} --thread-name {}'.format(args.cuda_name,args.data, args.model, thread_name)
            os.system(cmd)
            model_name = '{0}_{1}_{2}_{3}_{4}_{5}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop, thread_name)
            model_path = 'saved_models/evaluation/{0}_{1}.model'.format(args.data, model_name)
            model = utils.load_model(model_path, args, n_ent, n_rel, device)
            a_results, a_ranks, a_total_nums = evaluation(model, target, valid_filters, device, args.test_batch_size, entityid_to_nodetype, exists_edge)
            assert len(a_ranks) == 1
            if not final_result:
                final_result = a_results
            else:
                for k in final_result.keys():
                    final_result[k] += a_results[k]
            Ranks += a_ranks
            Totals += a_total_nums
        else:
            Ranks += [ori_ranks[i]]
            Totals += [ori_totals[i]]
            final_result['proportion'] += ori_ranks[i] / ori_totals[i]
    for k in final_result.keys():
        final_result[k] /= attack_data.shape[0]
    print('Final !!!')
    print(final_result)
    logger.info('Final !!!!')
    for k, v in final_result.items():
        logger.info('{} : {}'.format(k, v))
    tmp = np.array(Ranks) / np.array(Totals)
    print('Std:', np.std(tmp))
    with open(record_path, 'wb') as fl:
        pkl.dump([final_result, Ranks, Totals], fl)

else:
    assert len(target_data) == len(attack_data)
    print('Attack shape:'   , len(attack_data))
    Results = []
    Ranks = []
    Totals = []
    for l in range(0, len(target_data), 50):
        r = min(l+50, len(target_data))
        t_target_data = target_data[l:r]
        t_attack_data = attack_data[l:r]
        t_ori_ranks = ori_ranks[l:r]
        t_ori_totals = ori_totals[l:r]
        if args.mode == '':
            if not(args.added_edge_num == '' or int(args.added_edge_num) == 1):
                tt_attack_data = []
                for vv in t_attack_data:
                    tt_attack_data += list(vv)
                t_attack_data = tt_attack_data
        else:
            assert args.mode == 'sentence' or args.mode == 'bioBART'
            tt_attack_data = []
            for vv in t_attack_data:
                tt_attack_data += vv
            t_attack_data = tt_attack_data
        disturbed_data = list(ori_data) + list(t_attack_data)


        utils.save_data(disturbed_data_path, disturbed_data)
        cmd = 'CUDA_VISIBLE_DEVICES={} python main_multiprocess.py --data {} --model {} --thread-name {}'.format(args.cuda_name,args.data, args.model, thread_name)
        print('Training model {}...'.format(thread_name))
        os.system(cmd)
        model_name = '{0}_{1}_{2}_{3}_{4}_{5}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop, thread_name)
        model_path = 'saved_models/evaluation/{0}_{1}.model'.format(args.data, model_name)
        model = utils.load_model(model_path, args, n_ent, n_rel, device)
        a_results, a_ranks, a_totals = evaluation(model, t_target_data, valid_filters, device, args.test_batch_size, entityid_to_nodetype, exists_edge, attack_data = attack_data[l:r], ori_ranks = t_ori_ranks, ori_totals = t_ori_totals)
        print(f'************Current l: {l}\n', a_results)
        assert len(a_ranks) == t_target_data.shape[0]
        Results += [a_results]
        Ranks += list(a_ranks)
        Totals += list(a_totals)
    with open(record_path, 'wb') as fl:
        pkl.dump([Results, Ranks, Totals, index], fl)