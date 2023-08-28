#%%
import pickle as pkl
from typing import Dict, Tuple, List
import os
import numpy as np
import json
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

from model import Distmult, Complex, Conve
import utils

# from evaluation import evaluation

#%%
class Main(object):
    def __init__(self, args):
        self.args = args 
        
        self.model_name = '{0}_{1}_{2}_{3}_{4}'.format(args.model, args.embedding_dim, args.input_drop, args.hidden_drop, args.feat_drop)
        #leaving batches from the model_name since they do not depend on model_architecture 
        # also leaving kernel size and filters, siinice don't intend to change those
        self.model_path = 'saved_models/{0}_{1}.model'.format(args.data, self.model_name)
        
        self.log_path = 'logs/{0}_{1}_{2}_{3}.log'.format(args.data, self.model_name, args.epochs, args.train_batch_size)
        self.loss_path = 'losses/{0}_{1}_{2}_{3}.pickle'.format(args.data, self.model_name, args.epochs, args.train_batch_size)
        
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO,
                            filename = self.log_path)
        self.logger = logging.getLogger(__name__)
        self.logger.info(vars(self.args))
        self.logger.info('\n')
            
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_data()
        self.model = self.add_model()
        self.optimizer = self.add_optimizer(self.model.parameters())
        
        if self.args.save_influence_map:
            self.logger.info('-------- Argument save_influence_map is set. Will use GR to compute and save influence maps ----------\n')
            # when we want to save influence during training
            self.args.add_reciprocals = False # to keep things simple
            # init an empty influence map
            self.influence_map = defaultdict(float)
            #self.influence_path = 'influence_maps/{0}_{1}.json'.format(args.data, self.model_name)
            self.influence_path = 'influence_maps/{0}_{1}.pickle'.format(args.data, self.model_name)
            # Initialize a copy of the model prams to track previous weights in an epoch
            self.previous_weights = [copy.deepcopy(param) for param in self.model.parameters()]
            self.logger.info('Shape for previous weights: {}, {}'.format(self.previous_weights[0].shape, self.previous_weights[1].shape))
    
    def load_data(self):
        ''' 
        Load the train, valid datasets
        '''
        data_path = os.path.join('processed_data', self.args.data)
        n_ent, n_rel, ent_to_id, rel_to_id = utils.generate_dicts(data_path)
        self.n_ent = n_ent
        self.n_rel = n_rel
        
        self.train_data = utils.load_data(os.path.join(data_path, 'all.txt'))
        # print(type(self.train_data), self.train_data.shape) #(1996432, 3)
        tmp = np.random.choice(a = self.train_data.shape[0], size = int(self.train_data.shape[0] * self.args.KG_valid_rate), replace=False)
        self.valid_data= self.train_data[tmp, :]

    
    def add_model(self):
        
        if self.args.model is None:
            model = Distmult(self.args, self.n_ent, self.n_rel)
        elif self.args.model == 'distmult':
            model = Distmult(self.args, self.n_ent, self.n_rel)
        elif self.args.model == 'complex':
            model = Complex(self.args, self.n_ent, self.n_rel)
        elif self.args.model == 'conve':
            model = Conve(self.args, self.n_ent, self.n_rel)
        else:
            self.logger.info('Unknown model: {0}', self.args.model)
            raise Exception("Unknown model!")
        model.to(self.device)
        return model
    
    def add_optimizer(self, parameters):
        return torch.optim.Adam(parameters, lr=self.args.lr, weight_decay=self.args.lr_decay)
    
    def save_model(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.args)
        }
        torch.save(state, self.model_path)
        self.logger.info('Saving model to {0}'.format(self.model_path))
    
    def load_model(self):
        self.logger.info('Loading saved model from {0}'.format(self.model_path))
        state = torch.load(self.model_path)
        model_params = state['state_dict']
        params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        for key, size, count in params:
            self.logger.info(key, size, count)
        self.model.load_state_dict(model_params)
        self.optimizer.load_state_dict(state['optimizer'])
    
    def lp_regularizer(self):
        # Apply p-norm regularization; assign weights to each param
        weight = self.args.reg_weight
        p = self.args.reg_norm
        
        trainable_params = [self.model.emb_e.weight, self.model.emb_rel.weight]
        norm = 0
        for i in range(len(trainable_params)):
            #norm += weight * trainable_params[i].norm(p = p)**p
            norm += weight * torch.sum( torch.abs(trainable_params[i]) ** p)
            
        return norm
        
    def n3_regularizer(self, factors):
        # factors are the embeddings for lhs, rel, rhs for triples in a batch
        weight = self.args.reg_weight
        p = self.args.reg_norm
        
        norm = 0
        for f in factors:
            norm += weight * torch.sum(torch.abs(f) ** p)
            
        return norm / factors[0].shape[0] # scale by number of triples in batch
    
    def get_influence_map(self):
        """
        Turns the influence map into a list, ready to be written to disc. (before: numpy)
        :return: the influence map with lists as values
        """
        assert self.args.save_influence_map == True
        
        for key in self.influence_map:
            self.influence_map[key] = self.influence_map[key].tolist()
        #self.logger.info('get_influence_map passed')
        return self.influence_map

    def evaluate(self, split, batch_size, epoch):
        """
        The same as self.run_epoch()
        """

        self.model.eval()
        losses = []

        with torch.no_grad():
            input_data = torch.from_numpy(self.valid_data.astype('int64'))
            actual_examples = input_data[torch.randperm(input_data.shape[0]), :]
            del input_data

            batch_size = self.args.valid_batch_size
            for b_begin in tqdm(range(0, actual_examples.shape[0], batch_size)):

                input_batch = actual_examples[b_begin: b_begin + batch_size]
                input_batch = input_batch.to(self.device)
                
                s,r,o = input_batch[:,0], input_batch[:,1], input_batch[:,2]
                
                emb_s = self.model.emb_e(s).squeeze(dim=1)
                emb_r = self.model.emb_rel(r).squeeze(dim=1)
                emb_o = self.model.emb_e(o).squeeze(dim=1)
                
                if self.args.add_reciprocals:
                    r_rev = r + self.n_rel
                    emb_rrev = self.model.emb_rel(r_rev).squeeze(dim=1)
                else:
                    r_rev = r
                    emb_rrev = emb_r
                    
                pred_sr = self.model.forward(emb_s, emb_r, mode='rhs')
                loss_sr = self.model.loss(pred_sr, o) # cross entropy loss
                
                pred_or = self.model.forward(emb_o, emb_rrev, mode='lhs')
                loss_or = self.model.loss(pred_or, s)
                
                total_loss = loss_sr + loss_or
                
                if (self.args.reg_weight != 0.0 and self.args.reg_norm == 3):
                    #self.logger.info('Computing regularizer weight')
                    if self.args.model == 'complex':
                        emb_dim = self.args.embedding_dim #int(self.args.embedding_dim/2)
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
                        
                    total_loss  += self.n3_regularizer(factors_sr)
                    total_loss  += self.n3_regularizer(factors_or)
                    
                if (self.args.reg_weight != 0.0 and self.args.reg_norm == 2):
                    total_loss += self.lp_regularizer()

                losses.append(total_loss.item())

        loss = np.mean(losses)
        self.logger.info('[Epoch:{}]:  Validating Loss:{:.6}\n'.format(epoch, loss))
        return loss


    def run_epoch(self, epoch):
        self.model.train()
        losses = []
        
        #shuffle the train dataset
        input_data = torch.from_numpy(self.train_data.astype('int64'))
        actual_examples = input_data[torch.randperm(input_data.shape[0]), :]
        del input_data
        
        batch_size = self.args.train_batch_size
        
        for b_begin in tqdm(range(0, actual_examples.shape[0], batch_size)):
            self.optimizer.zero_grad()
            input_batch = actual_examples[b_begin: b_begin + batch_size]
            input_batch = input_batch.to(self.device)
            
            s,r,o = input_batch[:,0], input_batch[:,1], input_batch[:,2]
            
            emb_s = self.model.emb_e(s).squeeze(dim=1)
            emb_r = self.model.emb_rel(r).squeeze(dim=1)
            emb_o = self.model.emb_e(o).squeeze(dim=1)
            
            if self.args.add_reciprocals:
                r_rev = r + self.n_rel
                emb_rrev = self.model.emb_rel(r_rev).squeeze(dim=1)
            else:
                r_rev = r
                emb_rrev = emb_r
                
            pred_sr = self.model.forward(emb_s, emb_r, mode='rhs')
            loss_sr = self.model.loss(pred_sr, o) # loss is cross entropy loss
            
            pred_or = self.model.forward(emb_o, emb_rrev, mode='lhs')
            loss_or = self.model.loss(pred_or, s)
            
            total_loss = loss_sr + loss_or
            
            if (self.args.reg_weight != 0.0 and self.args.reg_norm == 3):
                #self.logger.info('Computing regularizer weight')
                if self.args.model == 'complex':
                    emb_dim = self.args.embedding_dim #int(self.args.embedding_dim/2)
                    lhs = (emb_s[:, :emb_dim], emb_s[:, emb_dim:])
                    rel = (emb_r[:, :emb_dim], emb_r[:, emb_dim:])
                    rel_rev = (emb_rrev[:, :emb_dim], emb_rrev[:, emb_dim:])
                    rhs = (emb_o[:, :emb_dim], emb_o[:, emb_dim:])
                    
                    #print(lhs[0].shape, lhs[1].shape)
                    factors_sr = (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                                torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
                    factors_or = (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                                torch.sqrt(rel_rev[0] ** 2 + rel_rev[1] ** 2),
                                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
                else:
                    factors_sr = (emb_s, emb_r, emb_o)
                    factors_or = (emb_s, emb_rrev, emb_o)
                    
                total_loss  += self.n3_regularizer(factors_sr)
                total_loss  += self.n3_regularizer(factors_or)
                
            if (self.args.reg_weight != 0.0 and self.args.reg_norm == 2):
                total_loss += self.lp_regularizer()
                    
                    
            total_loss.backward()
            self.optimizer.step()
            losses.append(total_loss.item())
            
            if self.args.save_influence_map: #for gradient rollback
                with torch.no_grad():
                    prev_emb_e = self.previous_weights[0]
                    prev_emb_rel = self.previous_weights[1]
                    # need to compute the influence value per-triple
                    for idx in range(input_batch.shape[0]):
                        head, rel, tail = s[idx], r[idx], o[idx]
                        inf_head = (emb_s[idx] - prev_emb_e[head]).cpu().detach().numpy()
                        inf_tail = (emb_o[idx] - prev_emb_e[tail]).cpu().detach().numpy()
                        inf_rel = (emb_r[idx] - prev_emb_rel[rel]).cpu().detach().numpy()
                        #print(inf_head.shape, inf_tail.shape, inf_rel.shape)

                        #write the influences to dictionary
                        key_trip = '{0}_{1}_{2}'.format(head.item(), rel.item(), tail.item())
                        key = '{0}_s'.format(key_trip)
                        self.influence_map[key] += inf_head
                        #self.logger.info('Written to influence map. Key: {0}, Value shape: {1}'.format(key, inf_head.shape))
                        key = '{0}_r'.format(key_trip)
                        self.influence_map[key] += inf_rel
                        key = '{0}_o'.format(key_trip)
                        self.influence_map[key] += inf_tail

                    # update the previous weights to be tracked
                    self.previous_weights = [copy.deepcopy(param) for param in self.model.parameters()]
            
            if (b_begin%5000 == 0) or (b_begin== (actual_examples.shape[0]-1)):
                self.logger.info('[E:{} | {}]: Train Loss:{:.6}'.format(epoch, b_begin, np.mean(losses)))
                
        loss = np.mean(losses)
        self.logger.info('[Epoch:{}]:  Training Loss:{:.6}\n'.format(epoch, loss))
        return loss
    
    def fit(self):
        self.model.init()
        self.logger.info(self.model)
        
        self.logger.info('------ Start the model training ------')
        start_time = time.time()
        self.logger.info('Start time: {0}'.format(str(start_time)))
    
    
        train_losses = []
        valid_losses = []
        best_val = 10000000000.
        for epoch in range(self.args.epochs):

            print("="*15,'epoch:',epoch,'='*15)
            train_loss = self.run_epoch(epoch)
            train_losses.append(train_loss)

            if train_loss < best_val:
                best_val = train_loss
                self.save_model()
            print("Train loss: {0}, Best loss: {1}\n\n".format(train_loss, best_val))

                
        with open(self.loss_path, "wb") as fl: 
            pkl.dump({"train loss":train_losses, "valid loss":valid_losses}, fl)
        self.logger.info('Time taken to train the model: {0}'.format(str(time.time() - start_time)))
        start_time = time.time()
        
        if self.args.save_influence_map: #save the influence map
            with open(self.influence_path, "wb") as fl:   #Pickling
                pkl.dump(self.get_influence_map(), fl)
            self.logger.info('Finished saving influence map')
            self.logger.info('Time taken to save the influence map: {0}'.format(str(time.time() - start_time)))
    
#%%
parser = utils.get_argument_parser()

args = parser.parse_args()
args = utils.set_hyperparams(args)

utils.seed_all(args.seed)
np.set_printoptions(precision=5)
cudnn.benchmark = False

model = Main(args)
model.fit()