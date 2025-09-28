import argparse
import torch
import numpy as np


from utils_data_building import (
    Person, 
    Rule, 
)

from utils_metric import (
    check_if_in_deductive_closure, 
    get_minimal_nec_unlearn_and_not_included_unlearn, 
    get_prec_rec_acc,  
    get_valid_unlearn_general,
    get_edge_id,
    get_deductive_closure,
)

parser = argparse.ArgumentParser(description='calculate the recall and accuracy')
parser.add_argument('--unlearn_data_id', type=int, default=None, help="id of the fact to unlearn")
parser.add_argument('--input_dir', type=str, default=None, help="directory that saves the rettained knowledge base")
args = parser.parse_args()

# (69, 67), father, Sloane Lee, <utils_data_building.Person object at 0x7e13e265ea80>: 'age', 'birthplace', 'children', 'father', 'gender', 'generation', 'husband', 'if_build', 'job', 'mother', 'name', 'wife'
(edge_list, edge_type_list, fixed_names, person_list) = torch.load("synthetic_data/family-200-graph.pt")
# <class 'utils_data_building.Rule'>:  [(0, 'wife', 1)], (1, 'husband', 0)
rule_list = torch.load("synthetic_data/family_rule.pt")
# вывод на основании правил и имеющихся фактов всех возможных следствий для полноты картины
dc_edge_list, dc_edge_type_list = get_deductive_closure(edge_list, edge_type_list, rule_list, person_list)
# a random subset of size 55 from the facts in family relationship to evaluate the deep unlearning 
shuffled_edge_id_list = torch.load("synthetic_data/subsample.pt")
# 267  - это номер грани? Тогда где указание на то, какой родственной связи соответствует?
# args - обработанные аргументы командной строки
shuffled_unlearn_data_id = shuffled_edge_id_list[args.unlearn_data_id]

# Сюда попадаем, когда не указан input_dir (предвычисления)
if args.input_dir is None:
    print("pre-compute the minimal deep unlearning set only")
    precision_list, recall_list, accuracy_list, minimal_unlearn_list = get_valid_unlearn_general(
        shuffled_unlearn_data_id, # 267 (номер факта из relationships)
        edge_list, # (69, 67), ... (здесь все грани из поданных синтетических данных family-200-graph.pt)
        edge_type_list, # father, ...(здесь все названия граней из поданных синтетических данных family-200-graph.pt)
        dc_edge_list, # здесь к edge_list добавлены все возможные следствия на основании rules
        dc_edge_type_list, # здесь к edge_type_list добавлены все выведенные названия
        np.zeros(len(edge_list)), # массив из нулей размером 400, т.к. 400 relationships
        rule_list, # 48 правил выведения родственных связей в формате left_tuples, right_tuple
        num_seed=100)
    exit()

# Если указан input_dir, попадаем сюда (полная оценка)    
rel_ind = np.asarray(torch.load(f"{args.input_dir}/relationships_correct.pt")).astype(np.float32)
unlearn_ind = 1 - rel_ind
bio_ind = torch.load(f"{args.input_dir}/biographies_correct.pt")

precision_list, recall_list, accuracy_list, minimal_unlearn_list = get_valid_unlearn_general(
    shuffled_unlearn_data_id, 
    edge_list, 
    edge_type_list, 
    dc_edge_list, 
    dc_edge_type_list, 
    unlearn_ind, 
    rule_list, 
    num_seed=100)

rec = max(recall_list)
argmax = np.asarray(recall_list).argmax()
acc_rel = accuracy_list[argmax]
acc_bio = np.asarray(bio_ind).mean()

num_rel = len(rel_ind)
num_bio = len(bio_ind)
size_mul = len(list(minimal_unlearn_list)[argmax])
acc_all = ((acc_bio * num_bio) + accuracy_list[argmax] * ( num_rel - size_mul)) / (num_bio + num_rel - size_mul)
print(("recall", "accuracy of relationships", "accuracy of biographies", "accurcy of all knowledge base"))
print((rec, acc_rel, acc_bio, acc_all))
torch.save((rec, acc_rel, acc_bio, acc_all), f"{args.input_dir}/rec_acc.pt")