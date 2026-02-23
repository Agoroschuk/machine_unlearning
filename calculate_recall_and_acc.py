import argparse
import torch
import numpy as np


from utils_data_building import (
    Person, 
    Rule, 
)

from utils_metric import (
    check_if_in_deductive_closure, 
    get_minimal_nec_unlearn_and_not_included_unlearn,  # создает мин.мн-во для забывания целевого факта
    get_prec_rec_acc,   # рассчитывает метрики
    get_valid_unlearn_general,
    get_edge_id, # выдает id ребра
    get_deductive_closure, # на основании поданных фактов и известных правил выводит все, что еще не выведены
)

parser = argparse.ArgumentParser(description='calculate the recall and accuracy')
parser.add_argument('--unlearn_data_id', type=int, default=None, help="id of the fact to unlearn")
parser.add_argument('--input_dir', type=str, default=None, help="directory that saves the retained knowledge base")
# в объект args кладется unlearn_data_id и input_dir
args = parser.parse_args()

# 400 relationships 
# (69, 67), father, Sloane Lee, <utils_data_building.Person object at 0x7e13e265ea80>: 'age', 'birthplace', 'children', 'father', 'gender', 'generation', 'husband', 'if_build', 'job', 'mother', 'name', 'wife'
# читать так: 67 = отец для 69, справа налево то есть
(edge_list, edge_type_list, fixed_names, person_list) = torch.load("synthetic_data/family-200-graph.pt")
# <class 'utils_data_building.Rule'>:  [(0, 'wife', 1)], (1, 'husband', 0)]
rule_list = torch.load("synthetic_data/family_rule.pt")
# вывод на основании правил и имеющихся 400 фактов всех возможных следствий (доп.фактов помимо 400)
dc_edge_list, dc_edge_type_list = get_deductive_closure(edge_list, edge_type_list, rule_list, person_list)
# a random subset of size 55 from the facts in family relationship to evaluate the deep unlearning 
shuffled_edge_id_list = torch.load("synthetic_data/subsample.pt")
# args - обработанные аргументы командной строки
shuffled_unlearn_data_id = shuffled_edge_id_list[args.unlearn_data_id]

# Сюда попадаем, когда не указан input_dir
if args.input_dir is None:
    print("pre-compute the minimal deep unlearning set only")
    precision_list, recall_list, accuracy_list, minimal_unlearn_list = get_valid_unlearn_general(
        shuffled_unlearn_data_id, # 267 (номер факта из relationships)
        edge_list, # (69, 67), ... (здесь все грани из поданных синтетических данных family-200-graph.pt)
        edge_type_list, # father, ...(здесь все названия граней из поданных синтетических данных family-200-graph.pt)
        dc_edge_list, # здесь к edge_list добавлены все возможные следствия на основании rules
        dc_edge_type_list, # здесь к edge_type_list добавлены все выведенные названия
        np.zeros(len(edge_list)), # массив из нулей размером 400, т.к. 400 relationships (видимо его будем заполнять 1 и 0 после unlearning)
        rule_list, # 48 правил выведения родственных связей в формате left_tuples, right_tuple
        num_seed=100)
    exit()  # весь код ниже не выполняется, если не передана директория с результатами забывания

# Если указан input_dir, попадаем сюда
# В rel_ind 1 у сохранившихся после unlearning фактов, массив из 1 и 0 размером 400
# relationships_correct.pt и biographies_correct.pt получены в vllm_eval.py, это булевы массивы размером 400 и 300, соответственно, 
# где True, если модель после забывания смогла сгенерировать верный ответ
rel_ind = np.asarray(torch.load(f"{args.input_dir}/relationships_correct.pt")).astype(np.float32)
# обратный массив из 0 и 1 размером 400
# Если в rel_ind 1, значит, факт не забылся, значит unlearn_ind будет 0. В unlearn_ind 1 у забытых фактов
unlearn_ind = 1 - rel_ind
bio_ind = torch.load(f"{args.input_dir}/biographies_correct.pt")

# в каждом массиве набор метрик от разных minimal_set для забывания shuffled_unlearn_data_id
# (так как min_set не единственно, максимум = 17 таких множеств согласно статье) 
precision_list, recall_list, accuracy_list, minimal_unlearn_list = get_valid_unlearn_general(
    shuffled_unlearn_data_id, 
    edge_list, 
    edge_type_list, 
    dc_edge_list, 
    dc_edge_type_list, 
    unlearn_ind, 
    rule_list, 
    num_seed=100)

# лучший recall
rec = max(recall_list) 
# его индекс
argmax = np.asarray(recall_list).argmax()
# accuracy для этого индекса
acc_rel = accuracy_list[argmax]
# просто среднее число 1 в результате забывания на фактах биографии
acc_bio = np.asarray(bio_ind).mean()

num_rel = len(rel_ind) # 400
num_bio = len(bio_ind) # 300
# Берем то мин.мн-во для забывания, которое дало лучший recall и считаем его длину
size_mul = len(list(minimal_unlearn_list)[argmax])
# чисто идейно не очень понятно, зачем нужен смешанный accuracy
acc_all = ((acc_bio * num_bio) + accuracy_list[argmax] * ( num_rel - size_mul)) / (num_bio + num_rel - size_mul)
print(("recall", "accuracy of relationships", "accuracy of biographies", "accuracy of all knowledge base"))
print((rec, acc_rel, acc_bio, acc_all))

# Сохранение нужно продумать по папочкам (метод/модель/id данных/чекпоинт) и сохранять на google drive в каком-то формате, в котором будут результаты
# ('recall', 'accuracy of relationships', 'accuracy of biographies', 'accuracy of all knowledge base')
# (0.75, 0.25, 0.023333333333333334, 0.15229885057471265)
torch.save((rec, acc_rel, acc_bio, acc_all), f"{args.input_dir}/rec_acc.pt")