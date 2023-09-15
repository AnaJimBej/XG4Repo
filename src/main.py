import pandas as pd
from prepare_data import adapt_data, get_vocab
from utils import  parse_args, load_config
import os

def main(args):
    # Load configuration
    cfgs, cfgs_save = load_config(args.config)
    cfg = cfgs[0]
    # We asume the data is splited into train, validation, test and graph files
    dataset_folder = cfg.data.data_path

    train_file =  os.path.join(cfg.data.data_path, 'train.txt')
    test_file = os.path.join(cfg.data.data_path, 'test.txt')
    dev_file = os.path.join(cfg.data.data_path, 'valid.txt')
    graph_file = os.path.join(cfg.data.data_path, 'graph.txt')
    files = [train_file, test_file, dev_file, graph_file]
    # Get vocabulary files
    get_vocab(dataset_folder, files)


if __name__ == '__main__':
    main(parse_args())