import pandas as pd

import comm
from utils import *
import torch
from torch import distributed as dist
from torch import nn
from torch.utils import data as torch_data
from itertools import islice
from data import RuleDataset, Iterator
import pickle
import matplotlib.pyplot as plt
#import pandas as pd
from os.path import exists
import os
class TrainerPredictor(object):

    def __init__(self, model, train_set, valid_set, test_set, optimizer, scheduler=None, gpus=None, num_worker=0):
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.gpus = gpus
        self.num_worker = num_worker

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            if len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                logging.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comm.init_process_group(backend, init_method="env://")

        if self.rank == 0:
            logging.info("Preprocess training set")
        if self.world_size > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if self.device.type == "cuda":
            model = model.cuda(self.device)

        self.model = model
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, save_path,batch_per_epoch, smoothing, print_every):
        if comm.get_rank() == 0:
            logging.info('>>>>> Predictor: Training')
        self.train_set.make_batches()
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = torch_data.DataLoader(self.train_set, 1, sampler=sampler, num_workers=self.num_worker)
        batch_per_epoch = batch_per_epoch or len(dataloader)
        model = self.model
        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device], find_unused_parameters=True)
            else:
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.train()

        total_loss = 0.0
        total_size = 0.0

        sampler.set_epoch(0)

        for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
            all_h, all_r, all_t, target, edges_to_remove = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            target = target.squeeze(0)
            edges_to_remove = edges_to_remove.squeeze(0)
            target_t = torch.nn.functional.one_hot(all_t, self.train_set.graph.entity_size)
            
            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                target = target.cuda(device=self.device)
                edges_to_remove = edges_to_remove.cuda(device=self.device)
                target_t = target_t.cuda(device=self.device)
            
            target = target * smoothing + target_t * (1 - smoothing)

            logits, mask,_,_ = model(all_h, all_r, edges_to_remove)
            if mask.sum().item() != 0:
                logits = (torch.softmax(logits, dim=1) + 1e-8).log()
                loss = -(logits[mask] * target[mask]).sum() / torch.clamp(target[mask].sum(), min=1)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                total_size += mask.sum().item()
                
            if (batch_id + 1) % print_every == 0:
                self.save(os.path.join(save_path, 'predictor_' + str(batch_id + 1) + '.pt'))
                if comm.get_rank() == 0:
                    logging.info('{} {} {:.6f} {:.1f}'.format(batch_id + 1, len(dataloader), total_loss / print_every, total_size / print_every))
                total_loss = 0.0
                total_size = 0.0

        if self.scheduler:
            self.scheduler.step()
    
    @torch.no_grad()
    def compute_H(self, print_every):
        if comm.get_rank() == 0:
            logging.info('>>>>> Predictor: Computing H scores of rules')
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = torch_data.DataLoader(self.train_set, 1, sampler=sampler, num_workers=self.num_worker)
        model = self.model

        model.eval()
        all_H_score = torch.zeros(model.num_rules, device=self.device)
        for batch_id, batch in enumerate(dataloader):
            all_h, all_r, all_t, target, edges_to_remove = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            target = target.squeeze(0)
            edges_to_remove = edges_to_remove.squeeze(0)
            
            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                target = target.cuda(device=self.device)
                edges_to_remove = edges_to_remove.cuda(device=self.device)
            
            H, index = model.compute_H(all_h, all_r, all_t, edges_to_remove)
            if H != None and index != None:
                all_H_score[index] += H / len(model.graph.train_facts)
                
            if (batch_id + 1) % print_every == 0:
                if comm.get_rank() == 0:
                    logging.info('{} {}'.format(batch_id + 1, len(dataloader)))
        
        if self.world_size > 1:
            all_H_score = comm.stack(all_H_score)
            all_H_score = all_H_score.sum(0)
        
        return all_H_score.data.cpu().numpy().tolist()

    @torch.no_grad()
    def get_queries(self,split):
        if comm.get_rank() == 0:
            logging.info('>>>>> Predictor: Evaluating on {}'.format(split))
        test_set = getattr(self, "%s_set" % split)
        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
        dataloader = torch_data.DataLoader(test_set, 1, sampler=sampler, num_workers=self.num_worker)

        concat_logits = []
        concat_all_h = []
        concat_all_r = []
        concat_all_t = []
        concat_flag = []
        concat_mask = []
        concat_top_rules = []
        for batch in dataloader:
            all_h, all_r, all_t, flag = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
        return concat_all_h, concat_all_r, concat_all_t

    # @torch.no_grad()
    # def evaluate_save(self, split, predictor, iteration, path, cfg, expectation=True, test_mode=False,
    #                   scatter_plot=False,
    #                   outfolder='..'):
    #     if comm.get_rank() == 0:
    #         logging.info('>>>>> Predictor: Evaluating on {}'.format(split))
    #     test_set = getattr(self, "%s_set" % split)
    #     sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
    #     dataloader = torch_data.DataLoader(test_set, 1, sampler=sampler, num_workers=self.num_worker)
    #     model = self.model
    #
    #     model.eval()
    #     concat_logits = []
    #     concat_all_h = []
    #     concat_all_r = []
    #     concat_all_t = []
    #     concat_flag = []
    #     concat_mask = []
    #     concat_top_rules = []
    #     concat_mask_score = []
    #
    #     ranks = []
    #     query2LH = dict()
    #     save_rank = []
    #     hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
    #     for batch in dataloader:
    #         all_h, all_r, all_t, flag = batch
    #         all_h = all_h.squeeze(0)
    #         all_r = all_r.squeeze(0)
    #         all_t = all_t.squeeze(0)
    #         flag = flag.squeeze(0)
    #         if self.device.type == "cuda":
    #             all_h = all_h.cuda(device=self.device)
    #             all_r = all_r.cuda(device=self.device)
    #             all_t = all_t.cuda(device=self.device)
    #             flag = flag.cuda(device=self.device)
    #         logits, mask, all_top_rules, mask_score = model(all_h, all_r, None, test_mode, outfolder, iteration)
    #
    #         for k in range(len(all_h)):
    #             h = all_h[k]
    #             r = all_r[k]
    #             t = all_t[k]
    #             if mask[k, t].item() == True:
    #                 val = logits[k, t]
    #                 L = (logits[k][flag[k]] > val).sum().item() + 1
    #                 H = (logits[k][flag[k]] >= val).sum().item() + 2
    #             else:
    #                 L = 1
    #                 H = test_set.graph.entity_size + 1
    #             ranks += [[h, r, t, L, H]]
    #             query2LH[(h, r, t)] = (L, H)
    #             save_rank.append([split, h, self.model.graph.id2relation[r.detach().cpu().item()], t, L, H])
    #
    #             if expectation:
    #                 for rank in range(L, H):
    #                     if rank <= 1:
    #                         hit1 += 1.0 / (H - L)
    #                     if rank <= 3:
    #                         hit3 += 1.0 / (H - L)
    #                     if rank <= 10:
    #                         hit10 += 1.0 / (H - L)
    #                     mr += rank / (H - L)
    #                     mrr += 1.0 / rank / (H - L)
    #             else:
    #                 rank = H - 1
    #                 if rank <= 1:
    #                     hit1 += 1
    #                 if rank <= 3:
    #                     hit3 += 1
    #                 if rank <= 10:
    #                     hit10 += 1
    #                 mr += rank
    #                 mrr += 1.0 / rank
    #
    #     ranks = torch.tensor(ranks, dtype=torch.long, device=self.device)
    #
    #     if self.world_size > 1:
    #         ranks = comm.cat(ranks)
    #
    #     df_ranks = pd.DataFrame(save_rank)
    #
    #     file_name = '/' + str(cfg.model_name) + '_' + str(predictor) + '_' + str(iteration) + '_' + str(
    #         cfg.EM.num_rules) + '_' + str(cfg.EM.best_rules) + '_ranks_' + str(cfg.seed) + '.csv'
    #
    #     with open(path + file_name, 'a') as fd:
    #         df_ranks.to_csv(fd, header=None, index=None, sep=';', mode='w', lineterminator='\n')
    #     hit1 /= len(ranks)
    #     hit3 /= len(ranks)
    #     hit10 /= len(ranks)
    #     mr /= len(ranks)
    #     mrr /= len(ranks)
    #
    #     model_name = path.split('/')[-1]
    #     save_eval_metrics = [model_name, predictor, iteration, split, str(hit1), str(hit3), str(hit10), str(mr),
    #                          str(mrr)]
    #     df_metrics = pd.DataFrame(save_eval_metrics).transpose()
    #     file_name = '/metrics_result_' + str(cfg.model_name) + '_' + str(cfg.EM.num_rules) + '_' + str(
    #         cfg.EM.best_rules) + '_' + str(cfg.seed) + '.csv'
    #
    #
    #     # file_name = '/metrics_results_EMlr'+ str(cfg.predictor.optimizer.lr) + \
    #     #             '_EMsmoothing'+str(cfg.predictor.train.smoothing)+'_pluslr'+\
    #     #             str(cfg.predictorplus.optimizer.lr) + '_plussmoothing'+str(cfg.predictorplus.train.smoothing) + str(cfg.model_name)
    #     file_name_folder = '/home/ana/Proyectos/rnnlogic_2/RNNLogic+/output/metrics' + file_name
    #     print('Llegué a la parte conflictiva')
    #     print(file_name_folder)
    #     # with open(file_name_folder, 'a') as fd:
    #     #     df_metrics.to_csv(fd, header=None, index=None, sep=';', mode='w', lineterminator='\n')
    #     if comm.get_rank() == 0:
    #         logging.info('Data : {}'.format(len(query2LH)))
    #         logging.info('Hit1 : {:.6f}'.format(hit1))
    #         logging.info('Hit3 : {:.6f}'.format(hit3))
    #         logging.info('Hit10: {:.6f}'.format(hit10))
    #         logging.info('MR   : {:.6f}'.format(mr))
    #         logging.info('MRR  : {:.6f}'.format(mrr))
    #
    #     return mrr


    @torch.no_grad()
    def evaluate(self, split,predictor,iteration,path,cfg,expectation=True,test_mode=False,scatter_plot=False,outfolder='..'):
        if comm.get_rank() == 0:
            logging.info('>>>>> Predictor: Evaluating on {}'.format(split))
        test_set = getattr(self, "%s_set" % split)
        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
        dataloader = torch_data.DataLoader(test_set, 1, sampler=sampler, num_workers=self.num_worker)
        model = self.model

        model.eval()
        concat_logits = []
        concat_all_h = []
        concat_all_r = []
        concat_all_t = []
        concat_flag = []
        concat_mask = []
        concat_top_rules = []
        concat_mask_score =[]
        for batch in dataloader:
            all_h, all_r, all_t, flag, flag_list = batch
            all_h = all_h.squeeze(0)
            all_r = all_r.squeeze(0)
            all_t = all_t.squeeze(0)
            flag = flag.squeeze(0)
            flag_list = flag_list.squeeze(0)

            if self.device.type == "cuda":
                all_h = all_h.cuda(device=self.device)
                all_r = all_r.cuda(device=self.device)
                all_t = all_t.cuda(device=self.device)
                flag = flag.cuda(device=self.device)
                flag_list = flag_list.cuda(device=self.device)

            logits, mask, all_top_rules, mask_score = model(all_h, all_r, None, test_mode, outfolder, iteration, flag_list)

            concat_logits.append(logits)
            concat_all_h.append(all_h)
            concat_all_r.append(all_r)
            concat_all_t.append(all_t)
            concat_flag.append(flag)
            concat_mask.append(mask)
            concat_top_rules.append(all_top_rules)
            concat_mask_score.append(mask_score)
        # with open('final_top_rules.pickle', 'wb') as fp:
        #    pickle.dump(concat_top_rules, fp)
        concat_logits = torch.cat(concat_logits, dim=0)
        concat_all_h = torch.cat(concat_all_h, dim=0)
        concat_all_r = torch.cat(concat_all_r, dim=0)
        concat_all_t = torch.cat(concat_all_t, dim=0)
        concat_flag = torch.cat(concat_flag, dim=0)
        concat_mask = torch.cat(concat_mask, dim=0)

        ranks = []
        for k in range(concat_all_t.size(0)): # Para todas las tripletas de test
            h = concat_all_h[k]
            r = concat_all_r[k]
            t = concat_all_t[k]
            if concat_mask[k, t].item() == True:
                val = concat_logits[k, t]
                L = (concat_logits[k][concat_flag[k]] > val).sum().item() + 1
                H = (concat_logits[k][concat_flag[k]] >= val).sum().item() + 2
            else:
                L = 1
                H = test_set.graph.entity_size + 1
            ranks += [[h, r, t, L, H]]
        ranks = torch.tensor(ranks, dtype=torch.long, device=self.device)
            
        if self.world_size > 1:
            ranks = comm.cat(ranks)
        
        query2LH = dict()
        save_rank = []
        for h, r, t, L, H in ranks.data.cpu().numpy().tolist():
            query2LH[(h, r, t)] = (L, H)
            self.model.graph.id2relation[r]
            save_rank.append([split, h, self.model.graph.id2relation[r], t, L, H])

        df_ranks = pd.DataFrame(save_rank)

        file_name = '/' +str(cfg.model_name)+'_' +str(predictor) + '_' + str(iteration) +'_'+str(cfg.EM.num_rules) +'_' + str(cfg.EM.best_rules) +'_ranks_'+str(cfg.seed)+'.csv'

        with open(path + file_name, 'a') as fd:
            df_ranks.to_csv(fd, header=None, index=None, sep=';', mode='w', lineterminator='\n')
            
        hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        for (L, H) in query2LH.values():
            if expectation:
                for rank in range(L, H):
                    if rank <= 1:
                        hit1 += 1.0 / (H - L)
                    if rank <= 3:
                        hit3 += 1.0 / (H - L)
                    if rank <= 10:
                        hit10 += 1.0 / (H - L)
                    mr += rank / (H - L)
                    mrr += 1.0 / rank / (H - L)
            else:
                rank = H - 1
                if rank <= 1:
                    hit1 += 1
                if rank <= 3:
                    hit3 += 1
                if rank <= 10:
                    hit10 += 1
                mr += rank
                mrr += 1.0 / rank
            
        hit1 /= len(ranks)
        hit3 /= len(ranks)
        hit10 /= len(ranks)
        mr /= len(ranks)
        mrr /= len(ranks)

        model_name = path.split('/')[-1]
        save_eval_metrics = [model_name,predictor, iteration, split, str(hit1), str(hit3),
                             str(hit10), str(mr), str(mrr)]
        df_metrics = pd.DataFrame(save_eval_metrics).transpose()
        file_name = '/metrics_result_' + str(cfg.model_name)+'_' + str(cfg.EM.num_rules) + '_' + \
                    str(cfg.EM.best_rules) + '_' + str(cfg.seed)+'.csv'

        # file_name = '/metrics_results_EMlr'+ str(cfg.predictor.optimizer.lr) + '_EMsmoothing'+
        # str(cfg.predictor.train.smoothing)+'_pluslr'+str(cfg.predictorplus.optimizer.lr) +
        # '_EMsmoothing'+str(cfg.predictorplus.train.smoothing)
        with open('/home/ana/Proyectos/rnnlogic_2/RNNLogic+/output/metrics' + file_name, 'a') as fd:
            df_metrics.to_csv(fd, header=None, index=None, sep=';', mode='w', lineterminator='\n')
        if comm.get_rank() == 0:
            logging.info('Data : {}'.format(len(query2LH)))
            logging.info('Hit1 : {:.6f}'.format(hit1))
            logging.info('Hit3 : {:.6f}'.format(hit3))
            logging.info('Hit10: {:.6f}'.format(hit10))
            logging.info('MR   : {:.6f}'.format(mr))
            logging.info('MRR  : {:.6f}'.format(mrr))

        return mrr

    def load(self, checkpoint, load_optimizer=True):
        """
        Load a checkpoint from file.
        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
        """
        if comm.get_rank() == 0:
            logging.info("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(state["model"])

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        comm.synchronize()

    def save(self, checkpoint):
        """
        Save checkpoint to file.
        Parameters:
            checkpoint (file-like): checkpoint file
        """
        if comm.get_rank() == 0:
            logging.info("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }
            torch.save(state, checkpoint)

        comm.synchronize()

class TrainerGenerator(object):

    def __init__(self, model, gpu):
        self.model = model

        if gpu is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(gpu)

        model = model.cuda(self.device)
    
    def train(self, rule_set, num_epoch=10000, lr=1e-3, print_every=100, batch_size=512, weights_flag=True):
        if comm.get_rank() == 0:
            logging.info('>>>>> Generator: Training')
        model = self.model
        model.train()
        
        dataloader = torch_data.DataLoader(rule_set, batch_size, shuffle=True, collate_fn=RuleDataset.collate_fn)
        iterator = Iterator(dataloader)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        total_loss = 0.0
        for epoch in range(num_epoch):
            batch = next(iterator)
            inputs, target, mask, weight = batch
            hidden = self.zero_state(inputs.size(0))
            
            if self.device.type == "cuda":
                inputs = inputs.cuda(self.device)
                target = target.cuda(self.device)
                mask = mask.cuda(self.device)
                weight = weight.cuda(self.device)

            loss = model.loss(inputs, target, mask, weight, hidden, weights_flag)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if (epoch + 1) % print_every == 0:
                if comm.get_rank() == 0:
                    logging.info('{} {} {:.6f}'.format(epoch + 1, num_epoch, total_loss / print_every))
                total_loss = 0.0
    
    def zero_state(self, batch_size): 
        state_shape = (self.model.num_layers, batch_size, self.model.hidden_dim)
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False, device=self.device)
        return (h0, c0)
    
    @torch.no_grad()
    def log_probability(self, rules):
        if rules == []:
            return []
        
        model = self.model
        model.eval()

        rules = [rule + [model.ending_idx] for rule in rules]
        max_len = max([len(rule) for rule in rules])
        for k in range(len(rules)):
            rule_len = len(rules[k])
            for i in range(max_len - rule_len):
                rules[k] += [model.padding_idx]
        rules = torch.tensor(rules, dtype=torch.long, device=self.device)
        inputs = rules[:, :-1]
        target = rules[:, 1:]
        n, l = target.size(0), target.size(1)
        mask = (target != model.padding_idx)
        hidden = self.zero_state(inputs.size(0))
        logits, hidden = model(inputs, inputs[:, 0], hidden)
        logits = torch.log_softmax(logits, -1)
        logits = logits * mask.unsqueeze(-1)
        target = (target * mask).unsqueeze(-1)
        log_prob = torch.gather(logits, -1, target).squeeze(-1) * mask
        log_prob = log_prob.sum(-1)
        return log_prob.data.cpu().numpy().tolist()

    @torch.no_grad()
    def next_relation_log_probability(self, seq, temperature):
        model = self.model
        model.eval()

        inputs = torch.tensor([seq], dtype=torch.long, device=self.device)
        relation = torch.tensor([seq[0]], dtype=torch.long, device=self.device)
        hidden = self.zero_state(1)
        logits, hidden = model(inputs, relation, hidden)
        log_prob = torch.log_softmax(logits[0, -1, :] / temperature, dim=-1).data.cpu().numpy().tolist()
        return log_prob
    
    @torch.no_grad()
    def beam_search(self, num_samples, max_len, temperature=0.2):
        if comm.get_rank() == 0:
            logging.info('>>>>> Generator: Rule generation with beam search')
        model = self.model
        model.eval()
        
        max_len += 1
        all_rules = []
        for relation in range(model.num_relations):
            found_rules = []
            prev_rules = [[[relation], 0]]
            for k in range(max_len):
                current_rules = list()
                for _i, (rule, score) in enumerate(prev_rules):
                    assert rule[-1] != model.ending_idx
                    log_prob = self.next_relation_log_probability(rule, temperature)
                    for i in (range(model.label_size) if (k + 1) != max_len else [model.ending_idx]):
                        new_rule = rule + [i]
                        new_score = score + log_prob[i]
                        (current_rules if i != model.ending_idx else found_rules).append((new_rule, new_score))
                    
                prev_rules = sorted(current_rules, key=lambda x:x[1], reverse=True)[:num_samples]
                found_rules = sorted(found_rules, key=lambda x:x[1], reverse=True)[:num_samples]

            ret = [rule[0:-1] + [score] for rule, score in found_rules]
            all_rules += ret
        return all_rules
    
    @torch.no_grad()
    def sample(self, num_samples, max_len, temperature=1.0):
        if comm.get_rank() == 0:
            logging.info('>>>>> Generator: Rule generation with sampling')
        model = self.model
        model.eval()

        all_rules = []
        for relation in range(model.num_relations):
            rules = torch.zeros([num_samples, max_len + 1], dtype=torch.long, device=self.device) + model.ending_idx
            log_probabilities = torch.zeros([num_samples, max_len + 1], device=self.device)
            head = torch.tensor([relation for k in range(num_samples)], dtype=torch.long, device=self.device)

            rules[:, 0] = relation
            hidden = self.zero_state(num_samples)

            for pst in range(max_len):
                inputs = rules[:, pst].unsqueeze(-1)
                logits, hidden = model(inputs, head, hidden)
                logits /= temperature
                log_probability = torch.log_softmax(logits.squeeze(1), dim=-1)
                probability = torch.softmax(logits.squeeze(1), dim=-1)
                sample = torch.multinomial(probability, 1)
                log_probability = log_probability.gather(1, sample)

                mask = (rules[:, pst] != model.ending_idx)
                
                rules[mask, pst + 1] = sample.squeeze(-1)[mask]
                log_probabilities[mask, pst + 1] = log_probability.squeeze(-1)[mask]

            length = (rules != model.ending_idx).sum(-1).unsqueeze(-1) - 1
            formatted_rules = torch.cat([length, rules], dim=1)

            log_probabilities = log_probabilities.sum(-1)

            formatted_rules = formatted_rules.data.cpu().numpy().tolist()
            log_probabilities = log_probabilities.data.cpu().numpy().tolist()
            for k in range(num_samples):
                length = formatted_rules[k][0]
                formatted_rules[k] = formatted_rules[k][1: 2 + length] + [log_probabilities[k]]

            rule_set = set([tuple(rule) for rule in formatted_rules])
            formatted_rules = [list(rule) for rule in rule_set]

            all_rules += formatted_rules

        return all_rules
    
    def load(self, checkpoint):
        """
        Load a checkpoint from file.
        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
        """
        if comm.get_rank() == 0:
            logging.info("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state["model"])

    def save(self, checkpoint):
        """
        Save checkpoint to file.
        Parameters:
            checkpoint (file-like): checkpoint file
        """
        if comm.get_rank() == 0:
            logging.info("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = {
            "model": self.model.state_dict()
        }
        torch.save(state, checkpoint)    
