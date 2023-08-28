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
parser.add_argument('--reasonable-rate', type = float, default=0.7, help = 'The added edge\'s existance rank prob greater than this rate')
parser.add_argument('--mode', type=str, default='sentence', help='sentence, biogpt or finetune')
parser.add_argument('--init-mode', type = str, default='random', help = 'How to select target nodes')
parser.add_argument('--ratio', type = str, default='', help='ratio of the number of changed words')
args = parser.parse_args()
args = utils.set_hyperparams(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
utils.seed_all(args.seed)
np.set_printoptions(precision=5)
cudnn.benchmark = False

data_path = '../DiseaseSpecific/processed_data/GNBR'
target_path = f'processed_data/target_{args.reasonable_rate}{args.init_mode}.pkl'
attack_path = f'processed_data/attack_edge_{args.model}_{args.reasonable_rate}{args.init_mode}.pkl'

# target_data = utils.load_data(target_path)
with open(target_path, 'rb') as fl:
    Target_node_list = pkl.load(fl)
with open(attack_path, 'rb') as fl:
    Attack_edge_list = pkl.load(fl)
attack_data = np.array(Attack_edge_list).reshape(-1, 3)
# assert target_data.shape == attack_data.shape
#%%

with open('../DiseaseSpecific/processed_data/GNBR/entities_reverse_dict.json') as fl:
    id_to_meshid = json.load(fl)
with open(Parameters.GNBRfile+'entity_raw_name', 'rb') as fl:
    entity_raw_name = pkl.load(fl)
with open(Parameters.GNBRfile+'retieve_sentence_through_edgetype', 'rb') as fl:
    retieve_sentence_through_edgetype = pkl.load(fl)
with open(Parameters.GNBRfile+'raw_text_of_each_sentence', 'rb') as fl:
    raw_text_sen = pkl.load(fl)

if args.mode == 'sentence':
    import torch 
    from torch.nn.modules.loss import CrossEntropyLoss
    from transformers import AutoTokenizer
    from transformers import BioGptForCausalLM
    criterion = CrossEntropyLoss(reduction="none")

    print('Generating GPT input ...')

    tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')
    tokenizer.pad_token = tokenizer.eos_token
    model = BioGptForCausalLM.from_pretrained('microsoft/biogpt', pad_token_id=tokenizer.eos_token_id)
    model.to(device)
    model.eval()
    GPT_batch_size = 24
    single_sentence = {}
    test_text = []
    test_dp = []
    test_parse = []
    for i, (s, r, o) in enumerate(tqdm(attack_data)):
        
        s = str(s)
        r = str(r)
        o = str(o)
        if int(s) != -1:

            dependency_sen_dict = retieve_sentence_through_edgetype[int(r)]['manual']
            candidate_sen = []
            Dp_path = []
            L = len(dependency_sen_dict.keys())
            bound = 500 // L
            if bound == 0:
                bound = 1
            for dp_path, sen_list in dependency_sen_dict.items():
                if len(sen_list) > bound:
                    index = np.random.choice(np.array(range(len(sen_list))), bound, replace=False)
                    sen_list = [sen_list[aa] for aa in index]
                candidate_sen += sen_list
                Dp_path += [dp_path] * len(sen_list)

            text_s = entity_raw_name[id_to_meshid[s]]
            text_o = entity_raw_name[id_to_meshid[o]]
            candidate_text_sen = []
            candidate_ori_sen = []
            candidate_parse_sen = []

            for paper_id, sen_id in candidate_sen:
                sen = raw_text_sen[paper_id][sen_id]
                text = sen['text']
                candidate_ori_sen.append(text)
                ss = sen['start_formatted']
                oo = sen['end_formatted']
                text = text.replace('-LRB-', '(')
                text = text.replace('-RRB-', ')')
                text = text.replace('-LSB-', '[')
                text = text.replace('-RSB-', ']')
                text = text.replace('-LCB-', '{')
                text = text.replace('-RCB-', '}')
                parse_text = text
                parse_text = parse_text.replace(ss, text_s.replace(' ', '_'))
                parse_text = parse_text.replace(oo, text_o.replace(' ', '_'))
                text = text.replace(ss, text_s)
                text = text.replace(oo, text_o)
                text = text.replace('_', ' ')
                candidate_text_sen.append(text)
                candidate_parse_sen.append(parse_text)
            tokens = tokenizer( candidate_text_sen,
                                truncation = True,
                                padding = True,
                                max_length = 300,
                                return_tensors="pt")
            target_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

            L = len(candidate_text_sen)
            assert L > 0
            ret_log_L = []
            for l in range(0, L, GPT_batch_size):
                R = min(L, l + GPT_batch_size)
                target = target_ids[l:R, :]
                attention = attention_mask[l:R, :]
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
            sen_score = list(zip(candidate_text_sen, ret_log_L, candidate_ori_sen, Dp_path, candidate_parse_sen))
            sen_score.sort(key = lambda x: x[1])
            test_text.append(sen_score[0][2])
            test_dp.append(sen_score[0][3])
            test_parse.append(sen_score[0][4])
            single_sentence.update({f'{s}_{r}_{o}_{i}': sen_score[0][0]})

        else:
            single_sentence.update({f'{s}_{r}_{o}_{i}': ''})

    with open(f'generate_abstract/{args.init_mode}{args.reasonable_rate}_sentence.json', 'w') as fl:
        json.dump(single_sentence, fl, indent=4)
    # with open('generate_abstract/test.txt', 'w') as fl:
    #     fl.write('\n'.join(test_text))
    # with open('generate_abstract/dp.txt', 'w') as fl:
    #     fl.write('\n'.join(test_dp))
    with open (f'generate_abstract/path/{args.init_mode}{args.reasonable_rate}_path.json', 'w') as fl:
        fl.write('\n'.join(test_dp))
    with open (f'generate_abstract/path/{args.init_mode}{args.reasonable_rate}_temp.json', 'w') as fl:
        fl.write('\n'.join(test_text))

elif args.mode == 'finetune':

    import spacy
    import pprint
    from transformers import AutoModel, AutoTokenizer,BartForConditionalGeneration

    print('Finetuning ...')

    with open(f'generate_abstract/{args.init_mode}{args.reasonable_rate}_chat.json', 'r') as fl:
        draft = json.load(fl)
    with open (f'generate_abstract/path/{args.init_mode}{args.reasonable_rate}_path.json', 'r') as fl:
        dpath = fl.readlines()
    
    nlp = spacy.load("en_core_web_sm")
    if os.path.exists(f'generate_abstract/bioBART/{args.init_mode}{args.reasonable_rate}{args.ratio}_candidates.json'):
        with open(f'generate_abstract/bioBART/{args.init_mode}{args.reasonable_rate}{args.ratio}_candidates.json', 'r') as fl:
            ret_candidates = json.load(fl)
    else:

        def find_mini_span(vec, words, check_set):
            

            def cal(text, sset):
                add = 0
                for tt in sset:
                    if tt in text:
                        add += 1
                return add
            text = ' '.join(words)
            max_add = cal(text, check_set)

            minn = 10000000
            span = ''
            rc = None
            for i  in range(len(vec)):
                if vec[i] == True:
                    p = -1
                    for j in range(i+1, len(vec)+1):
                        if vec[j-1] == True:
                            text = ' '.join(words[i:j])
                            if cal(text, check_set) == max_add:
                                p = j
                                break
                    if p > 0:
                        if (p-i) < minn:
                            minn = p-i
                            span = ' '.join(words[i:p])
                            rc = (i, p)
            if rc:
                for i in range(rc[0], rc[1]):
                    vec[i] = True
            return vec, span

        def mask_func(tokenized_sen):

            if len(tokenized_sen) == 0:
                return []
            token_list = []
            # for sen in tokenized_sen:
            #     for token in sen:
            #         token_list.append(token)
            for sen in tokenized_sen:
                token_list += sen.text.split(' ')
            if args.ratio == '':
                P = 0.3
            else:
                P = float(args.ratio)

            ret_list = []
            i = 0
            mask_num = 0
            while i < len(token_list):
                t = token_list[i]
                if '.' in t or '(' in t or ')' in t or '[' in t or ']' in t:
                    ret_list.append(t)
                    i += 1
                    mask_num = 0
                else:
                    length = np.random.poisson(3)
                    if np.random.rand() < P and length > 0:
                        if mask_num < 8:
                            ret_list.append('<mask>')
                            mask_num += 1
                        i += length
                    else:
                        ret_list.append(t)
                        i += 1
                        mask_num = 0
            return [' '.join(ret_list)]
                            
        model = BartForConditionalGeneration.from_pretrained('GanjinZero/biobart-large')
        model.eval()
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained('GanjinZero/biobart-large')

        ret_candidates = {}
        dpath_i = 0

        for i,(k, v) in enumerate(tqdm(draft.items())):

            input = v['in'].replace('\n', '')
            output = v['out'].replace('\n', '')
            s, r, o = attack_data[i]
            s = str(s)
            o = str(o)
            r = str(r)

            if int(s) == -1:
                ret_candidates[str(i)] = {'span': '', 'prompt' : '', 'out' : [], 'in': [], 'assist': []}
                continue

            path_text = dpath[dpath_i].replace('\n', '')
            dpath_i += 1
            text_s = entity_raw_name[id_to_meshid[s]]
            text_o = entity_raw_name[id_to_meshid[o]]

            doc = nlp(output)
            words= input.split(' ')
            tokenized_sens = [sen for sen in doc.sents]
            sens = np.array([sen.text for sen in doc.sents])

            checkset = set([text_s, text_o])
            e_entity = set(['start_entity', 'end_entity'])
            for path in path_text.split(' '):
                a, b, c = path.split('|')
                if a not in e_entity:
                    checkset.add(a)
                if c not in e_entity:
                    checkset.add(c)
            vec = []
            l = 0
            while(l < len(words)):
                bo =False
                for j in range(len(words), l, -1): # reversing is important !!!
                    cc = ' '.join(words[l:j])
                    if (cc in checkset):
                        vec += [True] * (j-l)
                        l = j
                        bo = True
                        break
                if not bo:
                    vec.append(False)
                    l += 1
            vec, span = find_mini_span(vec, words, checkset)
            # vec = np.vectorize(lambda x: x in checkset)(words)
            vec[-1] = True
            prompt = []
            mask_num = 0
            for j, bo in enumerate(vec):
                if not bo:
                    mask_num += 1
                else:
                    if mask_num > 0:
                        # mask_num = mask_num // 3 # span length ~ poisson distribution (lambda = 3)
                        mask_num = max(mask_num, 1)
                        mask_num= min(8, mask_num)
                        prompt += ['<mask>'] * mask_num
                    prompt.append(words[j])
                    mask_num = 0
            prompt = ' '.join(prompt)
            Text = []
            Assist = []
            
            for j in range(len(sens)):
                Bart_input = list(sens[:j]) + [prompt] +list(sens[j+1:])
                assist = list(sens[:j]) + [input] +list(sens[j+1:])
                Text.append(' '.join(Bart_input))
                Assist.append(' '.join(assist))
            
            for j in range(len(sens)):
                Bart_input = mask_func(tokenized_sens[:j]) + [input] + mask_func(tokenized_sens[j+1:])
                assist = list(sens[:j]) + [input] +list(sens[j+1:])
                Text.append(' '.join(Bart_input))
                Assist.append(' '.join(assist))

            batch_size = len(Text) // 2
            Outs = []
            for l in range(2):
                A = tokenizer(Text[batch_size * l:batch_size * (l+1)],
                truncation = True,
                padding = True,
                max_length = 1024,
                return_tensors="pt")
                input_ids = A['input_ids'].to(device)
                attention_mask = A['attention_mask'].to(device)
                aaid = model.generate(input_ids, num_beams = 5, max_length = 1024)
                outs = tokenizer.batch_decode(aaid, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                Outs += outs
            ret_candidates[str(i)] = {'span': span, 'prompt' : prompt, 'out' : Outs, 'in': Text, 'assist': Assist}
        with open(f'generate_abstract/bioBART/{args.init_mode}{args.reasonable_rate}{args.ratio}_candidates.json', 'w') as fl:
            json.dump(ret_candidates, fl, indent = 4)
    
    from torch.nn.modules.loss import CrossEntropyLoss
    from transformers import BioGptForCausalLM
    criterion = CrossEntropyLoss(reduction="none")

    tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')
    tokenizer.pad_token = tokenizer.eos_token
    model = BioGptForCausalLM.from_pretrained('microsoft/biogpt', pad_token_id=tokenizer.eos_token_id)
    model.to(device)
    model.eval()

    ret = {}
    case_study = {}
    dpath_i = 0
    for i,(k, v) in enumerate(tqdm(draft.items())):

        span = ret_candidates[str(i)]['span']
        prompt = ret_candidates[str(i)]['prompt']
        sen_list = ret_candidates[str(i)]['out']
        BART_in = ret_candidates[str(i)]['in']
        Assist = ret_candidates[str(i)]['assist']

        s, r, o = attack_data[i]
        s = str(s)
        r = str(r)
        o = str(o)

        if int(s) == -1:
            ret[k] = {'prompt': '', 'in':'', 'out': ''}
            p_ret[k] = {'prompt': '', 'in':'', 'out': ''}
            continue

        text_s = entity_raw_name[id_to_meshid[s]]
        text_o = entity_raw_name[id_to_meshid[o]]

        def process(text):

            for i in range(ord('A'), ord('Z')+1):
               text = text.replace(f'.{chr(i)}', f'. {chr(i)}')            
            return text

        sen_list = [process(text) for text in sen_list]
        path_text = dpath[dpath_i].replace('\n', '')
        dpath_i += 1

        checkset = set([text_s, text_o])
        e_entity = set(['start_entity', 'end_entity'])
        for path in path_text.split(' '):
            a, b, c = path.split('|')
            if a not in e_entity:
                checkset.add(a)
            if c not in e_entity:
                checkset.add(c)

        input = v['in'].replace('\n', '')
        output = v['out'].replace('\n', '')

        doc = nlp(output)
        gpt_sens = [sen.text for sen in doc.sents]
        assert len(gpt_sens) == len(sen_list) // 2

        word_sets = []
        for sen in gpt_sens:
            word_sets.append(set(sen.split(' ')))

        def sen_align(word_sets, modified_word_sets):
            
            l = 0
            while(l < len(modified_word_sets)):
                if len(word_sets[l].intersection(modified_word_sets[l])) > len(word_sets[l]) * 0.8:
                    l += 1
                else:
                    break
            if l == len(modified_word_sets):
                return -1, -1, -1, -1
            r = l + 1
            r1 = None
            r2 = None
            for pos1 in range(r, len(word_sets)):
                for pos2 in range(r, len(modified_word_sets)):
                    if len(word_sets[pos1].intersection(modified_word_sets[pos2])) > len(word_sets[pos1]) * 0.8:
                        r1 = pos1
                        r2 = pos2
                        break
                if r1 is not None:
                    break
            if r1 is None:
                r1 = len(word_sets)
                r2 = len(modified_word_sets)
            return l, r1, l, r2

        replace_sen_list = []
        boundary = []
        assert len(sen_list) % 2 == 0
        for j in range(len(sen_list) // 2):
            doc = nlp(sen_list[j])
            sens = [sen.text for sen in doc.sents]
            modified_word_sets = [set(sen.split(' ')) for sen in sens]
            l1, r1, l2, r2 = sen_align(word_sets, modified_word_sets)
            boundary.append((l1, r1, l2, r2))
            if l1 == -1:
                replace_sen_list.append(sen_list[j])
                continue
            check_text = ' '.join(sens[l2: r2])
            replace_sen_list.append(' '.join(gpt_sens[:l1] + [check_text] + gpt_sens[r1:]))
        sen_list = replace_sen_list + sen_list[len(sen_list) // 2:]

        old_L = len(sen_list)
        sen_list.append(output)
        sen_list += Assist
        tokens = tokenizer( sen_list,
                            truncation = True,
                            padding = True,
                            max_length = 1024,
                            return_tensors="pt")
        target_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        L = len(sen_list)
        ret_log_L = []
        for l in range(0, L, 5):
            R = min(L, l + 5)
            target = target_ids[l:R, :]
            attention = attention_mask[l:R, :]
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
        log_Loss = torch.cat(ret_log_L, -1).cpu().numpy()

        real_log_Loss = log_Loss.copy()
        log_Loss = log_Loss[:old_L]

        p = np.argmin(log_Loss)

        if real_log_Loss[p] > real_log_Loss[old_L]:
            if real_log_Loss[p] > real_log_Loss[p+1+old_L]:
                p = p+1+old_L
        ret[k] = {'prompt': prompt, 'in':input, 'out': sen_list[p]}
    with open(f'generate_abstract/{args.init_mode}{args.reasonable_rate}{args.ratio}_bioBART_finetune.json', 'w') as fl:
        json.dump(ret, fl, indent=4)
else:
    raise Exception('Wrong mode !!')