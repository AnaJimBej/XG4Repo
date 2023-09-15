import json
import pandas as pd
import numpy as np
import pandas as pd
import json
import csv
from operator import itemgetter
import preprocess_rules

"""
    Adapt the rules from AnyBURL to be the priors for RNNLogic
"""


def get_vocab(dataset_dir):
    entity_vocab = {}
    relation_vocab = {}
    entity_counter = len(entity_vocab)
    relation_counter = len(relation_vocab)
    for f in ['train.txt', 'dev.txt', 'test.txt']:
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
    return relation_vocab, entity_vocab


def append_rule(new_rule, outfile):
    new_rule_df = pd.DataFrame(new_rule).transpose()
    with open(outfile, 'a') as fd:
        new_rule_df.to_csv(fd, header=None, index=None, sep=' ', mode='w',line_terminator='\n')


def save_unique_rules(unique_rules,outfile,criterion='avg'):
    for rule in unique_rules.keys():
        scores = np.array(unique_rules[rule]).astype('float')
        if criterion == 'avg':
            score = np.average((scores))
        elif criterion == 'max':
            score = np.max(scores)
        rule = rule.replace('[','')
        rule = rule.replace(']', '')
        rule = rule.replace(',', '')
        rule = rule.split(' ')
        rule = list(map(int, rule))

        score = (score - 1e-5)/1000
        rule.append(str(score))
        append_rule(rule, outfile)


def main():

    dataset_dir = '/home/ana/Proyectos/rnnlogic_2/RNNLogic+/data/20230904_1800k_mj_rules/'
    rule_file = dataset_dir + 'HETIONET_rules-1000'
    anyburl_readable_rules = dataset_dir + 'HETIONET_rules_read.txt'
    outfile = dataset_dir + 'mined_rules_HETIONET.txt'
    r_vocab = dataset_dir + 'relations.dict'

    preprocess_rule_list_from_anyburl_longer_paths.main(rule_file,anyburl_readable_rules)
    rule_file = anyburl_readable_rules
    r_df = pd.read_csv(r_vocab,header=None,sep='\t')

    # Vocabulary
    # relation_vocab, entity_vocab = get_vocab(dataset_dir)
    relation_vocab = dict(zip(r_df[1], r_df[0]))

    seen_rules = []
    all_rules = []
    unique_rules = {}

    with open(rule_file, 'r') as f:
        rules = json.load(f)
    for rule in rules:
        rule_path = rule[:-1]
        new_rule = []
        print(len(rule_path))
        for step in rule_path:
            new_rule.append(relation_vocab[step[:-5]])

        rule_str = ''.join(str(new_rule))
        all_rules.append(rule_str)
        if rule_str not in seen_rules:
            seen_rules.append(rule_str)
            unique_rules[rule_str] = [rule[-1]]
        else:
            unique_rules[rule_str].append(rule[-1])
        new_rule.append(rule[-1])
        # new_rule_df = pd.DataFrame(new_rule).transpose()
        # with open(outfile, 'a') as fd:
        #     new_rule_df.to_csv(fd, header=None, index=None, sep=' ', mode='w',line_terminator='\n')

    print('Number of unique rules')
    print(len(unique_rules))
    save_unique_rules(unique_rules, outfile, criterion='avg')


if __name__ == "__main__":
    main()
