import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import copy
import random
import logging
from collections import defaultdict
from layers import MLP, FuncToNode, FuncToNodeSum
from embedding import RotatE
from torch.utils.data import Dataset, DataLoader
from torch_scatter import scatter, scatter_add, scatter_min, scatter_max, scatter_mean

import torch
from torch import distributed as dist
from torch import nn
from torch.utils import data as torch_data
import os
import pandas as pd
from datetime import datetime
import json
import networkx as nx
import matplotlib.pyplot as plt

def get_main_rules(self, rule_score, n, k, score,all_h,outfolder,query_r,edges_to_remove,iteration, flag=[]):
    outfolder = outfolder+'_'+str(iteration)
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)
    query_relation = self.graph.id2relation[(query_r)]
    query_folder = outfolder + '/' + query_relation
    if not os.path.isdir(query_folder):
        os.mkdir(query_folder)
    # Get top scored nodes for each query
    top_node_scores, top_nodes = torch.topk(score * flag, k, dim=1)
    top_rules = []
    # Para cada query
    queries = {}
    for query in range(rule_score.size(0)):
        # Select the rules that score each node
        top_nodes_rules = torch.index_select(rule_score[query].cuda(device=score.device), 0, top_nodes[query, :])
        # Select the most relevant rules for each node
        top_rule_score, top_rules_id = torch.topk(top_nodes_rules, n, dim=1)
        top_rules.append([top_nodes.cpu().numpy()[query, :], top_rules_id.cpu().numpy()])


        head = self.graph.id2entity[(all_h[query]).item()]
        query_file_name = query_folder + '/' + head.replace('::','_') + '.txt'
        if not os.path.isfile(query_file_name):
            candidate_nodes = {}
            for j in range(len(top_nodes[query])):  # Para cada nodo
                # for node in top_nodes[query]:
                node = top_nodes[query][j]
                # Append head, relation and node (e) to the file
                tail_name = self.graph.id2entity[(node).item()]
                e_triple = pd.DataFrame(['Prediction:', head, query_relation, self.graph.id2entity[(node).item()]])
                rule_k = 0

                top_rules_for_candidate = []
                G_rule = nx.DiGraph()

                for rule in top_rules_id[j]:
                    score = top_rule_score[j][rule_k]
                    rule_k = rule_k + 1
                    r_body = self.id2rule[rule.item()][1]
                    rule_body = ['Rule']
                    current_node = all_h[query].unsqueeze(0)
                    #parents_list = []
                    current_list = []
                    current_step = 0
                    subgraph_data = []
                    G_rule.add_node(current_node)
                    for step in r_body:
                        #step = r_body[k]
                        # Aquí es donde tengo que llamar a grounding y propagate
                        # Current node tiene que ser el id de los nodos de llegada
                        count = self.graph.grounding(current_node, query_r, [step], edges_to_remove).float()
                        new_nodes = torch.nonzero(count)[:, 1]
                        G_rule.add_node(new_nodes)

                        G_rule.add_edges_from([(1, 2), (1, 3)])
                        #prev_step = current_node
                        #path = torch.nonzero(count)
                        #path[:, 0] = prev_step[path[:, 0]]
                        #parent_nodes = path[:, 0]

                        #current_node = torch.nonzero(count)[:, 1]

                        # Sub-graph proposal
                        #heads = parent_nodes.detach().cpu().numpy()
                        #heads = [self.graph.id2entity[x] for x in heads]
                        #tails = current_node.detach().cpu().numpy()
                        #tails = [self.graph.id2entity[x] for x in tails]
                        # relation = np.ones(shape=current_node.size()) * step
                        #relation = np.repeat(self.graph.id2relation[(step)], current_node.size())
                        #sub_graph = np.concatenate(((np.asarray(heads))[:, None], relation[:, None],
                                                    #(np.asarray(tails))[:, None],
                                                    #(np.ones(shape=current_node.size()) * current_step)[:, None]),
                                                   #axis=1)
                        #ubgraph_data.append(sub_graph)
                        ###############
                        #current_step += 1
                        #current_list.append(current_node)
                        #parents_list.append(parent_nodes)


                        rule_body.append(self.graph.id2relation[step])

                    rule_body.append(float(score.detach().cpu().numpy()))
                    top_rules_for_candidate.append(rule_body)
                    rule_text = pd.DataFrame(rule_body)
                    #with open(query_file_name, 'a') as f:
                        #e_triple.T.to_csv(f, header=None, index=None, sep='\t', mode='w', lineterminator='\n')
                        #rule_text.T.to_csv(f, header=None, index=None, sep='\t', mode='w', lineterminator='\n')

                query_triplet = head +'->'+query_relation +'->'+tail_name
                candidate_nodes[query_triplet] = top_rules_for_candidate
                json_string = json.dumps(candidate_nodes)
                with open(query_file_name, 'w') as outfile:
                    json.dump(candidate_nodes,outfile,indent=2)


    # Rules of the real target node
    return top_rules

class Predictor(torch.nn.Module):
    def __init__(self, graph, entity_feature='bias'):
        super(Predictor, self).__init__()
        self.graph = graph
        self.num_entities = graph.entity_size
        self.num_relations = graph.relation_size
        self.entity_feature = entity_feature
        #self.embedding_path = embedding_path
        if entity_feature == 'bias':
            self.bias = torch.nn.parameter.Parameter(torch.zeros(self.num_entities))
        # elif entity_feature == 'RotatE':
        #     self.RotatE = RotatE(embedding_path)


    def set_rules(self, input):
        self.rules = list()
        if type(input) == list:
            for rule in input:
                rule_ = (rule[0], rule[1:])
                self.rules.append(rule_)
            logging.info('Predictor: read {} rules from list.'.format(len(self.rules)))
        elif type(input) == str:
            with open(input, 'r') as fi:
                for line in fi:
                    rule = line.strip().split()
                    rule = [int(_) for _ in rule]
                    rule_ = (rule[0], rule[1:])
                    self.rules.append(rule_)
            logging.info('Predictor: read {} rules from file.'.format(len(self.rules)))
        else:
            raise ValueError
        self.num_rules = len(self.rules)

        self.relation2rules = [[] for r in range(self.num_relations)]
        self.id2rule = {}
        for index, rule in enumerate(self.rules):
            relation = rule[0]
            self.relation2rules[relation].append([index, rule])
            self.id2rule[index] = rule
        
        self.rule_weights = torch.nn.parameter.Parameter(torch.zeros( self.num_rules))

    def forward(self, all_h, all_r, edges_to_remove, test_mode=False, outfolder='', iteration='', flag=[]):
        query_r = all_r[0].item()
        assert (all_r != query_r).sum() == 0
        device = all_r.device

        #########
        rule_score = torch.zeros(all_r.size(0),self.num_entities,self.num_rules)
        score = torch.zeros(all_r.size(0), self.num_entities, device=device)
        mask = torch.zeros(all_r.size(0), self.num_entities, device=device)
        #print(torch.nonzero(self.rule_weights))
        for index, (r_head, r_body) in self.relation2rules[query_r]: # For each rule
            assert r_head == query_r

            x = self.graph.grounding(all_h, r_head, r_body, edges_to_remove)
            #score_i = x
            #score += x

            score_i = x * self.rule_weights[index]
            score += x * self.rule_weights[index]

            if test_mode:
                rule_score[:,:,index] = score_i
            mask += x
        top_rules = []
        if mask.sum().item() == 0:
            if self.entity_feature == 'bias':
                return mask + self.bias.unsqueeze(0), (1 - mask).bool(), top_rules, []
                #return mask , (1 - mask).bool(), top_rules, []
            else:
                return mask - float('-inf'), mask.bool(), top_rules, []
        
        if self.entity_feature == 'bias':
            score = score + self.bias.unsqueeze(0)
            mask = torch.ones_like(mask).bool()
        else:
            mask = (mask != 0)
            score = score.masked_fill(~mask, float('-inf'))

        if test_mode:
            query_r = all_r[0].item()
            top_rules = get_main_rules(self, rule_score, n=10, k=20, score=score, all_h=all_h,outfolder=outfolder,
                                       query_r=query_r, edges_to_remove=edges_to_remove, iteration=iteration, flag=flag)
            #top_rules = get_main_rules(rule_score=rule_score,n=5,k=5,score = score,outfolder)
        return score, mask, top_rules, []



    def compute_H(self, all_h, all_r, all_t, edges_to_remove):
        query_r = all_r[0].item()
        assert (all_r != query_r).sum() == 0
        device = all_r.device

        rule_score = list()
        rule_index = list()
        mask = torch.zeros(all_r.size(0), self.num_entities, device=device)
        for index, (r_head, r_body) in self.relation2rules[query_r]:
            assert r_head == query_r

            x = self.graph.grounding(all_h, r_head, r_body, edges_to_remove)
            score = x * self.rule_weights[index]
            mask += x
            #explain = [all_h,all_r,all_t,score]
            rule_score.append(score)
            rule_index.append(index)

        rule_index = torch.tensor(rule_index, dtype=torch.long, device=device)
        pos_index = F.one_hot(all_t, self.num_entities).bool()
        if device.type == "cuda":
            pos_index = pos_index.cuda(device)
        neg_index = (mask != 0)

        if len(rule_score) == 0:
            return None, None

        rule_H_score = list()
        for score in rule_score:
            pos_score = (score * pos_index).sum(1) / torch.clamp(pos_index.sum(1), min=1)
            neg_score = (score * neg_index).sum(1) / torch.clamp(neg_index.sum(1), min=1)
            H_score = pos_score - neg_score
            rule_H_score.append(H_score.unsqueeze(-1))

        rule_H_score = torch.cat(rule_H_score, dim=-1)
        rule_H_score = torch.softmax(rule_H_score, dim=-1).sum(0)

        return rule_H_score, rule_index

    def evaluate_plus(self, all_h,all_r,all_t,device,test_mode=True):
        logits, mask, top_rules = self.forward(all_h, all_r, None, test_mode)
        a = 10

class PredictorPlus(torch.nn.Module):
    def __init__(self, graph, type='emb', num_layers=3, hidden_dim=16, entity_feature='bias', aggregator='sum', embedding_path=''):
        super(PredictorPlus, self).__init__()
        self.graph = graph

        self.type = type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.entity_feature = entity_feature
        self.aggregator = aggregator
        self.embedding_path = embedding_path

        self.num_entities = graph.entity_size
        self.num_relations = graph.relation_size
        self.padding_index = graph.relation_size

        self.vocab_emb = torch.nn.Embedding(self.num_relations + 1, self.hidden_dim, padding_idx=self.num_relations)

        if self.type == 'lstm':
            self.rnn = torch.nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        elif self.type == 'gru':
            self.rnn = torch.nn.GRU(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        elif self.type == 'rnn':
            self.rnn = torch.nn.RNN(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        elif self.type == 'emb':
            self.rule_emb = None
        else:
            raise NotImplementedError

        if aggregator == 'sum':
            self.rule_to_entity = FuncToNodeSum(self.hidden_dim)
        elif aggregator == 'pna':
            self.rule_to_entity = FuncToNode(self.hidden_dim)
        else:
            raise NotImplementedError

        self.relation_emb = torch.nn.Embedding(self.num_relations, self.hidden_dim)
        self.score_model = MLP(self.hidden_dim * 2, [128, 1]) # 128 for FB15k

        if entity_feature == 'bias':
            self.bias = torch.nn.parameter.Parameter(torch.zeros(self.num_entities))
        elif entity_feature == 'RotatE':
            self.RotatE = RotatE(embedding_path)

    def set_rules(self, input):
        self.rules = list()
        if type(input) == list:
            for rule in input:
                rule_ = (rule[0], rule[1:])
                self.rules.append(rule_)
            logging.info('Predictor+: read {} rules from list.'.format(len(self.rules)))
        elif type(input) == str:
            self.rules = list()
            with open(input, 'r') as fi:
                for line in fi:
                    rule = line.strip().split()
                    rule = [int(_) for _ in rule]
                    rule_ = (rule[0], rule[1:])
                    self.rules.append(rule_)
            logging.info('Predictor+: read {} rules from file.'.format(len(self.rules)))
        else:
            raise ValueError
        self.num_rules = len(self.rules)
        self.max_length = max([len(rule[1]) for rule in self.rules])

        self.relation2rules = [[] for r in range(self.num_relations)]
        self.id2rule = {}
        for index, rule in enumerate(self.rules):
            relation = rule[0]
            self.relation2rules[relation].append([index, rule])
            self.id2rule[index] = rule
        self.rule_features = []
        for rule in self.rules:
            rule_ = [rule[0]] + rule[1] + [self.padding_index for i in range(self.max_length - len(rule[1]))]
            self.rule_features.append(rule_)
        self.rule_features = torch.tensor(self.rule_features, dtype=torch.long)

        if self.type == 'emb':
            self.rule_emb = nn.parameter.Parameter(torch.zeros(self.num_rules, self.hidden_dim))
            nn.init.kaiming_uniform_(self.rule_emb, a=math.sqrt(5), mode="fan_in")

        #self.rule_weights = torch.nn.parameter.Parameter(torch.zeros(self.num_rules))
    def encode_rules(self, rule_features):
        rule_masks = rule_features != self.num_relations
        x = self.vocab_emb(rule_features)
        output, hidden = self.rnn(x)
        idx = (rule_masks.sum(-1) - 1).long()
        idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        rule_emb = torch.gather(output, 1, idx).squeeze(1)
        return rule_emb
    
    def forward(self, all_h, all_r, edges_to_remove,test_mode=False,outfolder='..',iteration= '', flag=[]):
        query_r = all_r[0].item()
        assert (all_r != query_r).sum() == 0
        device = all_r.device

        if device.type == "cuda":
            self.rule_features = self.rule_features.cuda(device)

        rule_index = list()
        rule_count = list()
        mask = torch.zeros(all_h.size(0), self.graph.entity_size, device=device)
        rule_flag = torch.zeros(all_r.size(0),self.num_entities,self.num_rules)
        my_count = 0
        for index, (r_head, r_body) in self.relation2rules[query_r]:
            my_count += 1
            assert r_head == query_r

            count = self.graph.grounding(all_h, r_head, r_body, edges_to_remove).float()
            mask += count
            rule_flag[:,:, index] = count
            rule_index.append(index)
            rule_count.append(count)
        #print(my_count)
        if mask.sum().item() == 0:
            if self.entity_feature == 'bias':

                return mask + self.bias.unsqueeze(0), (1 - mask).bool(),[],[]
                #return mask , (1 - mask).bool()
            elif self.entity_feature == 'RotatE':
                bias = self.RotatE(all_h, all_r)
                return mask + bias, (1 - mask).bool(),[],[]
            else:
                return mask - float('-inf'), mask.bool(),[],[]

        candidate_set = torch.nonzero(mask.view(-1), as_tuple=True)[0]
        batch_id_of_candidate = candidate_set // self.graph.entity_size

        rule_index = torch.tensor(rule_index, dtype=torch.long, device=device)
        rule_count = torch.stack(rule_count, dim=0)
        rule_count_ = rule_count.clone()
        rule_count = rule_count.reshape(rule_index.size(0), -1)[:, candidate_set]
        
        if self.type == 'emb':
            rule_emb = self.rule_emb[rule_index]
        else:
            rule_emb = self.encode_rules(self.rule_features[rule_index])

        output = self.rule_to_entity(rule_count, rule_emb, batch_id_of_candidate)
        
        rel = self.relation_emb(all_r[0]).unsqueeze(0).expand(output.size(0), -1)
        feature = torch.cat([output, rel], dim=-1)
        output = self.score_model(feature).squeeze(-1)

        score = torch.zeros(all_h.size(0) * self.graph.entity_size, device=device)
        score.scatter_(0, candidate_set, output) # Creo que esta es la agregación
        score = score.view(all_h.size(0), self.graph.entity_size)
        mask_ = mask.clone()
        if self.entity_feature == 'bias':
            score = score + self.bias.unsqueeze(0)
            mask = torch.ones_like(mask).bool()
        elif self.entity_feature == 'RotatE':
            bias = self.RotatE(all_h, all_r)
            score = score + 0.1*bias
            mask = torch.ones_like(mask).bool()
        else:
            mask = (mask != 0)
            score = score.masked_fill(~mask, float('-inf'))
        top_rules = []
        if test_mode:
            aaa= 3
            top_rules = self.get_main_rules_plus(rule_index, rule_count_.transpose(1,2).transpose(0,2), 10, 20,
                                                 score, all_h, query_r, save_evaluation_rules=True, outfolder=outfolder,
                                                 edges_to_remove=edges_to_remove, iteration=iteration, flag=flag)
        return score, mask, top_rules, mask_

    def get_main_rules_plus(self, rule_index, rule_score, n, k, score, all_h=[], query_r=[],
                            save_evaluation_rules=False, edges_to_remove=[],
                            outfolder='/home/ana/Proyectos/rnnlogic_2/RNNLogic+/output/metapaths',
                            iteration='', flag=[]):
        if save_evaluation_rules:
            outfolder = outfolder+'_'+iteration
            if not os.path.isdir(outfolder):
                os.mkdir(outfolder)
            query_relation = self.graph.id2relation[(query_r)]
            query_folder = outfolder +'/'+ query_relation
            if not os.path.isdir(query_folder):
                os.mkdir(query_folder)
        # Get top scored nodes for each query
        top_node_scores, top_nodes = torch.topk(score*flag, k, dim=1)
        # top_node_scores, top_nodes = torch.topk(score, k, dim=1)

        top_rules = []
        # Para cada query
        for query in range(rule_score.size(0)): # Para cada query
            # Select the rules that score each node
            top_nodes_rules = torch.index_select(rule_score[query].cuda(device=score.device), 0, top_nodes[query, :])
            # Select the most relevant rules for each node
            top_rule_score, top_rules_id = torch.topk(top_nodes_rules, n, dim=1)
            top_rules_id = rule_index[top_rules_id]
            top_rules.append([top_nodes.cpu().numpy()[query, :], top_rules_id.cpu().numpy()])

            if save_evaluation_rules:
                head = self.graph.id2entity[(all_h[query]).item()]
                query_file_name = query_folder + '/' + head.replace('::','_') + '.txt'
                if not os.path.isfile(query_file_name):
                    candidate_nodes = {}
                    for j in range(len(top_nodes[query])): # Para cada nodo
                    # for node in top_nodes[query]:
                        node = top_nodes[query][j]
                        # Append head, relation and node (e) to the file
                        tail_name = self.graph.id2entity[(node).item()]
                        e_triple = pd.DataFrame(['Prediction:',head, query_relation, self.graph.id2entity[(node).item()]])
                        if head == 'Battle_Battle_of_Riverrun' and query_relation== 'IS_IN':
                            if tail_name == 'Culture_braavosi':
                                aa = 10
                        rule_k = 0
                        top_rules_for_candidate = []
                        for rule in top_rules_id[j]:

                            score = top_rule_score[j][rule_k]
                            rule_k = rule_k+1
                            r_body = self.id2rule[rule.item()][1]
                            rule_body = ['Rule']
                            current_node = all_h[query].unsqueeze(0)
                            #parents_list = []
                            current_list = []
                            current_step = 0
                            subgraph_data = []

                            for step in r_body:
                                #step = r_body[k]
                                # Aquí es donde tengo que llamar a grounding y propagate
                                # Current node tiene que ser el id de los nodos de llegada
                                #count = self.graph.grounding(current_node, query_r, [step], edges_to_remove).float()
                                #prev_step = current_node
                                #path = torch.nonzero(count)
                                #path[:, 0] = prev_step[path[:, 0]]
                                #parent_nodes = path[:,0]
                                #current_node = torch.nonzero(count)[:,1]

                                # Sub-graph proposal
                                #heads = parent_nodes.detach().cpu().numpy()
                                #heads = [self.graph.id2entity[x] for x in heads]
                                #tails = current_node.detach().cpu().numpy()
                                #tails = [self.graph.id2entity[x] for x in tails]
                                #relation = np.ones(shape=current_node.size()) * step
                                #relation = np.repeat(self.graph.id2relation[(step)],current_node.size())
                                #sub_graph = np.concatenate(((np.asarray(heads))[:, None], relation[:, None], (np.asarray(tails))[:, None], (np.ones(shape=current_node.size()) * current_step)[:, None]), axis=1)
                                #subgraph_data.append(sub_graph)
                                ###############
                                #current_step += 1
                                #current_list.append(current_node)
                                #parents_list.append(parent_nodes)
                                rule_body.append(self.graph.id2relation[step])


                            rule_body.append(float(score.detach().cpu().numpy()))
                            top_rules_for_candidate.append(rule_body)
                            rule_text = pd.DataFrame(rule_body)
                            #with open(query_file_name, 'a') as f:
                                #e_triple.T.to_csv(f, header=None, index=None, sep='\t', mode='w', lineterminator='\n')
                                #rule_text.T.to_csv(f, header=None, index=None, sep='\t', mode='w', lineterminator='\n')

                        query_triplet = head + '->' + query_relation + '->' + tail_name
                        candidate_nodes[query_triplet] = top_rules_for_candidate
                        json_string = json.dumps(candidate_nodes)
                        with open(query_file_name, 'w') as outfile:
                            json.dump(candidate_nodes, outfile, indent=2)


        # Rules of the real target node
        return top_rules