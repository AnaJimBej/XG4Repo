import sys
import os
import os.path as osp
import logging
import argparse
import random
import json
import os
sys.path.insert(0, os.getcwd())
from easydict import EasyDict
import numpy as np
from datetime import datetime
import torch
import operator
from data import KnowledgeGraph, TrainDataset, ValidDataset, TestDataset, RuleDataset
from predictors import Predictor, PredictorPlus
from generators import Generator
from utils import load_config, save_config, set_logger, set_seed, parse_args
from trainer import TrainerPredictor, TrainerGenerator
import comm
import shutil



def save_rules(predictor, outpath):
    with open(outpath + '/relation2rules.json', "w") as outfile:
          outfile.write(json.dumps(predictor.relation2rules, indent=0))
    with open(outpath + '/rule_weights.json', "w") as outfile:
        outfile.write(json.dumps(list(np.float64(predictor.rule_weights.cpu().detach().numpy())), indent=0))


#     rules = {}
#     weight = {}
#     relation2rules = predictor.relation2rules
#     for relation in relation2rules:
#         for index, (r_head, r_body) in relation:
#             rules[index] = (r_head, r_body)
#     with open(final_rules, "w") as outfile:
#         outfile.write(json.dumps(rules, indent=0))
#     with open(final_rules_weights, "w") as outfile:
#         outfile.write(json.dumps(rules, indent=0))


def select_top_rules(predictor,dataset,top_k,rules):
    top_rules = []
    # Select a decaying number of Important rules

    # Organize rule by relation
    relation2rules = [[] for r in range(dataset.num_relations)]
    for index, rule in enumerate(rules):
        relation = rule[0]
        relation2rules[relation].append(rule)
    # For each relation, get top k
    for i in range(dataset.num_relations):
        r_rules = relation2rules[i]
        sorted_rules = sorted(r_rules, key=operator.itemgetter(-1))
        top_k_rules = sorted_rules[0:top_k]
        top_rules = top_rules + top_k_rules
    return top_rules

def filter_rules(path,rules):
    relation2head = json.load(open(path + '/r2head.json'))
    relation2tail = json.load(open(path + '/r2tail.json'))
    new_rules = []
    for rule in rules:
        if len(rule)>2:
            relation = rule[0]
            rule_head = relation2head[str(rule[1])]
            relation_head = relation2head[str(relation)]
            rule_tail = relation2tail[str(rule[-2])]
            relation_tail = relation2tail[str(relation)]
            if (len(set(rule_head).intersection(set(relation_head))) > 0) & (
                    len(set(rule_tail).intersection(set(relation_tail))) > 0):
                # Conservar regla
                new_rules.append(rule)
            #else:
                #print('mal')
        else:
            new_rules.append(rule)

    return new_rules


def main(args):
    cfgs, cfgs_save = load_config(args.config)
    cfg = cfgs[0]

    if cfg.save_path is None:
        cfg.save_path = os.path.join('../outputs', datetime.now().strftime('%Y%m-%d%H-%M%S'))


    smoothing_list = [0.8]
    lr_list = [1e-10]

    base_path = cfg.save_path
    for smoothing in smoothing_list:
        for lr in lr_list:

            #cfg.predictorplus.optimizer.lr = lr
            #cfg.predictorplus.train.smoothing = smoothing

            cfg.save_path = base_path + '/' + datetime.now().strftime('%Y%m%d-%H%M%S')+'_PLUSsmoothing_' +\
                            str(cfg.predictorplus.train.smoothing) + '_PLUSlr_' + str(cfg.predictorplus.optimizer.lr)
            cfg.out_metapath = cfg.save_path + '/metapath'

            if cfg.save_path and not os.path.exists(cfg.save_path):
                os.makedirs(cfg.save_path)

            # Change this to save properly
            save_config(cfgs_save, cfg.save_path)

            set_logger(cfg.save_path)
            set_seed(cfg.seed)

            graph = KnowledgeGraph(cfg.data.data_path)
            train_set = TrainDataset(graph, cfg.data.batch_size)
            valid_set = ValidDataset(graph, cfg.data.batch_size)
            test_set = TestDataset(graph, cfg.data.batch_size)

            dataset = RuleDataset(graph.relation_size, cfg.data.rule_file)

            if comm.get_rank() == 0:
                logging.info('-------------------------')
                logging.info('| Pre-train Generator')
                logging.info('-------------------------')
            generator = Generator(graph, **cfg.generator.model)
            solver_g = TrainerGenerator(generator, gpu=cfg.generator.gpu)
            solver_g.train(dataset, **cfg.generator.pre_train)

            replay_buffer = list()
            best_valid_mrr = 0.0
            test_mrr = 0.0
            best_em_iter = 0
            solver_g.save(os.path.join(cfg.save_path, 'generator_0.pt'))

            for k in range(cfg.EM.num_iters):
                if comm.get_rank() == 0:
                    logging.info('-------------------------')
                    logging.info('| EM Iteration: {}/{}'.format(k + 1, cfg.EM.num_iters))
                    logging.info('-------------------------')

                # Sample logic rules.
                sampled_rules = solver_g.sample(cfg.EM.num_rules, cfg.EM.max_length)
                #sampled_rules = solver_g.beam_search(cfg.EM.num_rules, cfg.EM.max_length)

                # sampled_rules = list()
                # for num_rules, max_length in zip(cfg.EM.num_rules,cfg.EM.max_length):
                #     sampled_rules_ = solver_g.beam_search(num_rules, max_length)
                #     # sampled_rules_ = filter_rules(cfg.data.data_path, sampled_rules_)
                #     sampled_rules += sampled_rules_


                print(len(sampled_rules))
                #sampled_rules = filter_rules(cfg.data.data_path,sampled_rules)

                prior = [rule[-1] for rule in sampled_rules]
                rules = [rule[0:-1] for rule in sampled_rules]

                # Train a reasoning predictor with sampled logic rules.
                predictor = Predictor(graph, **cfg.predictor.model)
                predictor.set_rules(rules)
                optim = torch.optim.Adam(predictor.parameters(), **cfg.predictor.optimizer)
                solver_p = TrainerPredictor(predictor, train_set, valid_set, test_set, optim, gpus=cfg.predictor.gpus)
                print(cfg.predictor.train)
                solver_p.train(cfg.save_path,**cfg.predictor.train)

                #########
                # gpus = cfg.predictorplus.gpus
                # device = torch.device(gpus[comm.get_rank() % len(gpus)])
                # predictor.evaluate_plus([0,1],[1,1],[2,3],device)
                # save_rules(predictor.relation2rules, cfg.save_path)  # Creo que el predictor final ya usa estas reglas
                ############

                solver_p.save(os.path.join(cfg.save_path, 'predictor_'+str(k + 1)+'.pt'))

                is_validation = True
                # print(cfg.out_metapath)
                if is_validation:
                    valid_mrr_iter = solver_p.evaluate('valid','EM',str(k + 1),cfg.save_path, cfg,expectation=cfg.predictor.eval.expectation,test_mode= False, scatter_plot=False,outfolder=cfg.out_metapath+'EM')
                    test_mrr_iter = solver_p.evaluate('test','EM',str(k + 1),cfg.save_path, cfg,expectation=cfg.predictor.eval.expectation,test_mode= True, scatter_plot=False,outfolder=cfg.out_metapath+'EM')

                # E-step: Compute H scores of logic rules.
                likelihood = solver_p.compute_H(**cfg.predictor.H_score)
                posterior = [l + p * cfg.EM.prior_weight for l, p in zip(likelihood, prior)]
                for i in range(len(rules)):
                    rules[i].append(posterior[i])
                replay_buffer += rules

                # rules = select_top_rules(predictor, dataset, cfg.EM.best_rules, rules)
                # M-step: Update the rule generator.
                dataset = RuleDataset(graph.relation_size, rules)
                solver_g.train(dataset, **cfg.generator.train)
                solver_g.save(os.path.join(cfg.save_path, 'generator_' + str(k + 1) + '.pt'))
                if is_validation:
                    if valid_mrr_iter > best_valid_mrr:
                        best_valid_mrr = valid_mrr_iter
                        test_mrr = test_mrr_iter
                        # Remove folder
                        if best_em_iter != 0:
                            dir2remove = cfg.out_metapath +'EM_'+str(best_em_iter)
                            shutil.rmtree(dir2remove)
                        best_em_iter = k + 1
                        solver_p.save(os.path.join(cfg.save_path, 'predictor.pt'))
                    elif os.path.isdir(cfg.out_metapath +'EM_'+str(k+1)):
                        dir2remove = cfg.out_metapath +'EM_'+str(k+1)
                        shutil.rmtree(dir2remove)

            if replay_buffer != []:
                if comm.get_rank() == 0:
                    logging.info('-------------------------')
                    logging.info('| Post-train Generator')
                    logging.info('-------------------------')
                dataset = RuleDataset(graph.relation_size, replay_buffer)
                solver_g.train(dataset, **cfg.generator.post_train)

            if comm.get_rank() == 0:
                logging.info('-------------------------')
                logging.info('| Beam Search Best Rules')
                logging.info('-------------------------')

            solver_g.load(cfg.save_path + '/generator_'+str(best_em_iter-1)+'.pt')
            sampled_rules = list()
            for num_rules, max_length in zip(cfg.final_prediction.num_rules, cfg.final_prediction.max_length):
                sampled_rules_ = solver_g.beam_search(num_rules, max_length)
                #sampled_rules_ = filter_rules(cfg.data.data_path, sampled_rules_)
                sampled_rules += sampled_rules_

            prior = [rule[-1] for rule in sampled_rules]
            rules = [rule[0:-1] for rule in sampled_rules]

            if comm.get_rank() == 0:
                logging.info('-------------------------')
                logging.info('| Train Final Predictor+')
                logging.info('-------------------------')

            predictor = PredictorPlus(graph, **cfg.predictorplus.model)
            predictor.set_rules(rules)
            optim = torch.optim.Adam(predictor.parameters(), **cfg.predictorplus.optimizer)
            # save_rules(predictor, cfg.save_path ) # Creo que el predictor final ya usa estas reglas

            solver_p = TrainerPredictor(predictor, train_set, valid_set, test_set, optim, gpus=cfg.predictorplus.gpus)

            best_valid_mrr = 0.0
            test_mrr = 0.0
            for k in range(cfg.final_prediction.num_iters):
                if comm.get_rank() == 0:
                    logging.info('-------------------------')
                    logging.info('| Iteration: {}/{}'.format(k + 1, cfg.final_prediction.num_iters))
                    logging.info('-------------------------')

                solver_p.train(cfg.save_path,**cfg.predictorplus.train)
                solver_p.save(os.path.join(cfg.save_path, 'predictor_plus_'+str(k + 1)+'.pt'))
                if is_validation:
                    valid_mrr_iter = solver_p.evaluate('valid', 'PLUS', str(k + 1), cfg.save_path, cfg,
                                                       expectation=cfg.predictorplus.eval.expectation)
                    test_mrr_iter = solver_p.evaluate('test', 'PLUS', str(k + 1), cfg.save_path, cfg,
                                                      expectation=cfg.predictorplus.eval.expectation, test_mode=True,
                                                      scatter_plot=False, outfolder=cfg.out_metapath+'PLUS')

                    if valid_mrr_iter > best_valid_mrr:
                        best_valid_mrr = valid_mrr_iter
                        test_mrr = test_mrr_iter
                        solver_p.save(os.path.join(cfg.save_path, 'predictor_plus.pt'))

            if comm.get_rank() == 0:
                logging.info('-------------------------')
                logging.info('| Final Test MRR: {:.6f}'.format(test_mrr))
                logging.info('-------------------------')

if __name__ == '__main__':
    main(parse_args())