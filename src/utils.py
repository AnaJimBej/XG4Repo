import sys
import os
import logging
import argparse
import random
import json
import yaml
import easydict
import numpy as np
import torch
import jinja2

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='RNNLogic',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--config', default='../rnnlogic.yaml', type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args(args)

def load_config(cfg_file):
    with open(cfg_file, "r") as fin:
        raw_text = fin.read()

    if "---" in raw_text:
        configs = []
        grid, template = raw_text.split("---")
        grid = yaml.safe_load(grid)
        template = jinja2.Template(template)
        configs_save = []
        for hyperparam in meshgrid(grid):
            config_i = yaml.safe_load(template.render(hyperparam))
            config = easydict.EasyDict(config_i)
            configs_save.append(config_i)
            configs.append(config)
    else:
        configs_save = yaml.safe_load(raw_text)
        configs = [easydict.EasyDict(configs_save)]
        configs_save = [configs_save]

    return configs, configs_save

def save_config(cfg, path,i=0):
    with open(os.path.join(path, 'config.yaml'), 'w') as fo:
        yaml.dump(dict(cfg[i]), fo)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

def save_model(model, optim, args):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    params = {
        'model': model.state_dict(),
        'optim': optim.state_dict()
    }

    torch.save(params, os.path.join(args.save_path, 'checkpoint'))

def load_model(model, optim, args):
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])

def set_logger(save_path):
    log_file = os.path.join(save_path, 'run.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)