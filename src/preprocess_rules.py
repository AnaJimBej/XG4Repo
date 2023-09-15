import anyburl2rnnlogic
import re
import json
import pandas as pd
import numpy as np
"""
    Prepare AnyBURL rules for the models (longer paths)
"""

def reverse_relation(relation_name,first_letter,second_letter):
    if '_' in relation_name:
        correct_relation = relation_name.replace('_', '')
    else:
        correct_relation = '_' + relation_name
    correct_letters = '(' + second_letter + ',' + first_letter + ')'
    return correct_relation, correct_letters



def main(rule_file,outfile):
    with open(rule_file, 'r') as f:
        rules = f.readlines()

    preprocessed_rules = []
    short_preprocessed_rules = []
    for rule in rules:
        rule_elements = rule.split('\t')
        score = float(rule_elements[2])

        rule_head = rule_elements[-1].split('<=')[0].replace(' ','')
        rule_body = rule_elements[-1].split('<=')[1]
        rule_body = rule_body.replace('\n','')
        rule_relations = rule_body.split(', ')

        head_letters = rule_head[-5:]
        if head_letters != '(X,Y)':
            print('There is a problem with the head')
            if head_letters == '(Y,X)':
                print('The rule is upside down')
        else:
            correct_rule = []
            correct_rule.append(rule_head)
            for relation in rule_relations:
                relation_name = relation[:-5].replace(' ','')
                relation_letters = relation[-5:]
                first_letter = relation_letters[1]
                second_letter = relation_letters[3]

                if 'X' in relation_letters:
                    if first_letter == 'X':
                        correct_relation = relation_name
                        correct_letters = relation_letters
                    else:
                        correct_relation,correct_letters = reverse_relation(relation_name,first_letter,second_letter)

                elif 'Y' in relation_letters:
                    if second_letter == 'Y':
                        correct_relation = relation_name
                        correct_letters = relation_letters
                    else:
                        correct_relation,correct_letters = reverse_relation(relation_name,first_letter,second_letter)
                else:
                    if first_letter<second_letter:
                        correct_relation = relation_name
                        correct_letters = relation_letters
                    else:
                        correct_relation,correct_letters = reverse_relation(relation_name,first_letter,second_letter)

                correct_rule.append(correct_relation+correct_letters)
            correct_rule.append(score)
            preprocessed_rules.append(correct_rule)
            #anyburl2rnnlogic.append_rule(correct_rule,outfile)
    with open(outfile, 'w') as f:
        json.dump(preprocessed_rules, f)


if __name__ == '__main__':
    dataset_dir = 'C:/Users/anaji/OneDrive - Universidad Politécnica de Madrid/MUIT_OneDrive/TFM_GAPS/20230126_hetionet/18k/'
    rule_file = dataset_dir + 'rules-1000_lenght6'
    outfile = dataset_dir +'test_rule_file.txt'
    main(rule_file,outfile)

