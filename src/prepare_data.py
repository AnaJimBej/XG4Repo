import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.getcwd())
import csv
import prepare_rules
from utils import parse_args, load_config


"""
    Prepare the graph to be the input of the model
"""

def adapt_data(file):
    # Modify this depending on the original dataset
    df_origin = pd.read_csv(file, sep='\t',names=['head_prime','relation','tail_prime'])
    df_no_space = df_origin.copy()
    df_no_space['head_prime'] = df_no_space['head_prime'].str.replace('::', '__')
    df_no_space['head_prime'] = df_no_space['head_prime'].str.replace(':', '-')
    df_no_space['head_prime'] = df_no_space['head_prime'].str.replace(' ', '_')
    df_no_space['tail_prime'] = df_no_space['tail_prime'].str.replace('::', '__')
    df_no_space['tail_prime'] = df_no_space['tail_prime'].str.replace(':', '-')
    df_no_space['tail_prime'] = df_no_space['tail_prime'].str.replace(' ', '_')
    with open(file, 'w') as f:
        df_no_space.to_csv(f, header=None, index=None, sep='\t', mode='w', line_terminator='\n')
def create_graph_inv(input_file, output_file):
    df_origin = pd.read_csv(input_file, sep='\t', names=['head', 'relation', 'tail'])

    df_not_inv = df_origin[['tail', 'relation', 'head']]
    df_not_inv['relation'] = '_' + df_origin['relation'].astype(str)
    df_not_inv = df_not_inv.rename(columns={"head": "tail", "tail": "head"})
    df_all = pd.concat([df_origin, df_not_inv])
    df_all = df_all.sample(frac=1)
    with open(output_file, 'a') as f:
        df_all.to_csv(f, header=None, index=None, sep='\t', mode='w', line_terminator='\n')

def split_train_test(train_file,dev_file,test_file,graph_file,data_file,relation_i=None):
    data_csv = pd.read_csv(data_file, sep=',')

    if relation_i is not None:
        df = data_csv[data_csv['relation'] == relation_i]
    else:
        df = data_csv.copy()
    train = df.sample(frac=0.8, random_state=23)
    test_val = df.drop(train.index)
    test = test_val.sample(frac=0.5, random_state=23)
    val = test_val.drop(test.index)
    graph = data_csv.drop(test_val.index)
    with open(train_file, 'w') as f:
        train.to_csv(train_file, header=None, index=None, sep='\t', mode='w', lineterminator='\n')
    with open(dev_file, 'w') as f:
        val.to_csv(f, header=None, index=None, sep='\t', mode='w', lineterminator='\n')
    with open(test_file, 'w') as f:
        test.to_csv(f, header=None, index=None, sep='\t', mode='w', lineterminator='\n')
    with open(graph_file, 'w') as f:
        graph.to_csv(f, header=None, index=None, sep='\t', mode='w', lineterminator='\n')


def filter_relations_in_df_files(relations,infile,name):
    df = pd.read_csv(infile, sep='\t',names=['head_prime','relation','tail_prime'])

    df_filter = df[df['relation'] == relations[0]]
    if len(relations)>1:
        for relation in relations[1:]:
            df_filter_r = df[df['relation'] == relation]
            df_filter = df_filter.append(df_filter_r, ignore_index=True)

    outfile = infile.split('.')[0] + '_' +name +'.txt'
    with open(outfile, 'w') as f:
        df_filter.to_csv(f, header=None, index=None, sep='\t', mode='w', lineterminator='\n')


def sample_df(file,frac):
    df = pd.read_csv(file, sep='\t', names=['head_prime', 'relation', 'tail_prime'])

    df_sampled = df.sample(frac=frac)
    outfile = file.split('.')[0] + '_frac_'+str(frac) +'.txt'
    with open(outfile, 'w') as f:
        df_sampled.to_csv(f, header=None, index=None, sep='\t', mode='w', lineterminator='\n')


def get_vocab(dataset_dir, files):
    entity_vocab = {}
    relation_vocab = {}

    entity_counter = len(entity_vocab)
    relation_counter = len(relation_vocab)

    for f in files:
        with open(dataset_dir + f) as raw_file:
            csv_file = csv.reader(raw_file, delimiter='\t')
            for line in csv_file:
                e1, r, e2 = line
                if e1 not in entity_vocab:
                    entity_vocab[e1] = entity_counter
                    entity_counter += 1
                if e2 not in entity_vocab:
                    entity_vocab[e2] = entity_counter
                    entity_counter += 1
                if r not in relation_vocab:
                    relation_vocab[r] = relation_counter
                    relation_counter += 1

    df_entity = pd.DataFrame.from_dict(entity_vocab, orient='index').reset_index()
    df_entity = df_entity[[df_entity.columns[1], df_entity.columns[0]]]
    with open(
            dataset_dir+ '/entities.dict',
            'w') as f:
        df_entity.to_csv(f, header=None, index=None, sep='\t', mode='w', lineterminator='\n')
    df_relation = pd.DataFrame.from_dict(relation_vocab, orient='index').reset_index()
    df_relation = df_relation[[df_relation.columns[1], df_relation.columns[0]]]
    with open(dataset_dir+ '/relations.dict',
              'w') as f:
        df_relation.to_csv(f, header=None, index=None, sep='\t', mode='w', lineterminator='\n')


def remove_relation_type(file, relations_to_remove,outfile):
    df = pd.read_csv(file, sep='\t', names=['head_prime', 'relation', 'tail_prime'])

    for relation in relations_to_remove:
        df.drop(df[df['relation'] ==relation ].index, inplace=True)

    with open(outfile, 'w') as f:
        df.to_csv(f, header=None, index=None, sep='\t', mode='w', lineterminator='\n')

def remove_relation_from_rule(r_dict_file,rule_file, relations2remove, remove_inverse):
    r_dict =  pd.read_csv(r_dict_file, sep='\t',names=['id','relation'])
    id2r = r_dict.set_index('id')
    id2r = id2r.to_dict()['relation']
    r2id = r_dict.set_index('relation')
    r2id = r2id.to_dict()['id']
    id2remove = []
    for relation in r2id.keys():
        if remove_inverse:
            if '_' in relation:
                id2remove.append(r2id[relation])
        if relation in relations2remove:
            id2remove.append(r2id[relation])

    id2remove = list(set(id2remove))
    outfile = rule_file.split('.')[0]+'_modified_rules.txt'

    n_original_rules = 0
    n_removed_rules = 0
    with open(rule_file, 'r') as fi:
        for line in fi:
            n_original_rules = n_original_rules+1
            rule = line.strip().split()
            to_drop = False
            for step in rule[:-1]:
                if int(step) in id2remove:
                    to_drop = True
            if to_drop:
                n_removed_rules = n_removed_rules+1
                print('Drop rule')
            else:
                prepare_rules.append_rule(rule, outfile)

    print('Number of original rules: %d, Number of removed rules %d' % (n_original_rules, n_removed_rules))


def main(args):
    cfgs, cfgs_save = load_config(args.config)
    cfg = cfgs[0]
    # The objective of this script is to prepare the data to be used in the models.
    # join os.join
    dataset_dir = cfg.data.data_path
    train_file = dataset_dir + 'train.txt'
    test_file = dataset_dir + 'test.txt'
    dev_file = dataset_dir + 'valid.txt'
    graph_file = dataset_dir + 'graph.txt'

    r_dict_file = dataset_dir +'relations.dict'
    data_file = configuration_file.got_csv
    files = [train_file, test_file, dev_file]

    relation_i = 'CtD'
    # Test section

    # Step 1: Obtain train, validation and test
    no_train_test = False
    if no_train_test:
        split_train_test(train_file, dev_file, test_file, graph_file, data_file, relation_i)
        print('Split in train, validation and test done')

    # Step 2: Adapt the format
    adapt_files = False
    if adapt_files:
        for file in files:
            adapt_data(file)
        print('Adapt files done')

    # Step 3: Obtain inverse
    inverse = False
    if inverse:
        create_graph_inv(graph_file, graph_file)

    # Step 4: Obtain vocabulary
    vocab = True
    if vocab:
        get_vocab(dataset_dir,['train.txt', 'valid.txt', 'test.txt','graph.txt'])
        print('Vocabulary created')

if __name__ == "__main__":
    main(parse_args())




