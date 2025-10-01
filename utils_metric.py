import os
import torch
import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy

def check_if_in_deductive_closure(unlearn_data_id, minimal_set, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list):
    cur_minimal_set = set(list(deepcopy(minimal_set)) + list(range(len(edge_list), len(dc_edge_list))))
    
    new_added_id_list = []
    t = 0
    while len(new_added_id_list) > 0 or t == 0:
        new_added_id_list = []
        t = t + 1
        for cur_unlearn_data_id in cur_minimal_set:
            unlearn_edge = dc_edge_list[cur_unlearn_data_id]
            unlearn_edge_type = dc_edge_type_list[cur_unlearn_data_id]
            rule_set_related = [rule for rule in rule_list if rule.right_tuple[1] == unlearn_edge_type]
            if_deducted = False
            for rule in rule_set_related:
                if if_deducted:
                    break
                up_edges_list = rule.get_up_edges_list(dc_edge_list, dc_edge_type_list, unlearn_edge, unlearn_edge_type)
                for up_edges in up_edges_list:
                    up_edges_if_deducted = True
                    for up_edge in up_edges:
                        ind = get_edge_id((up_edge[0], up_edge[2]), dc_edge_list)
                        if ind in cur_minimal_set:
                            up_edges_if_deducted = False
                            break
                    if up_edges_if_deducted:
                        if_deducted = True
                        new_added_id_list.append(cur_unlearn_data_id)
                        break
        for new_added_id in new_added_id_list:
            cur_minimal_set.remove(new_added_id)
    if unlearn_data_id in cur_minimal_set:
        return False
    else:
        return True              
                
    
def get_minimal_nec_unlearn_and_not_included_unlearn(unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list, seed=0):
    np.random.seed(seed)
    random.seed(seed)
    
    minimal_set = set([])
    minimal_set_unverified = set([unlearn_data_id])

    
    #Find a valid unlearning set expanded from the given unlearning result.
    while len(minimal_set_unverified) >= 1:
#         print(minimal_set_unverified)
        # sorted, т.к. random.sample требует упорядоченность, потом список из 1 эл-та и сам эл-т [0]
        cur_unlearn_data_id = random.sample(sorted(minimal_set_unverified), 1)[0]
        # удаляем id этого элемента, minimal_set_unverified становится пустым в 1 проход
        minimal_set_unverified.remove(cur_unlearn_data_id)
        # и добавляем id рассматриваемый факта в minimal_set
        minimal_set.add(cur_unlearn_data_id)

        # получаем tuple, отражающий связь для рассматриваемого id (69, 67)
        unlearn_edge = dc_edge_list[cur_unlearn_data_id]
        # получаем название связи-ребра (father)
        unlearn_edge_type = dc_edge_type_list[cur_unlearn_data_id]
        # если рассматриваемый факт = следствие в правиле, то эти ассоциированные с фактом правила сохраняем
        rule_set_related = [rule for rule in rule_list if rule.right_tuple[1] == unlearn_edge_type]

        # по следствию из каждого правила ищем причины. У каждого правила несколько 
        for rule in rule_set_related:
            # причины [(10, 'father', 20), (20, 'brother', 40)],
            # [(10, 'father', 20), (20, 'brother', 30), (30, 'brother', 40)]
            up_edges_list = rule.get_up_edges_list(dc_edge_list, dc_edge_type_list, unlearn_edge, unlearn_edge_type)
            for up_edges in up_edges_list:
                if_suf = 0
                for up_edge in up_edges:
                    # получаем id ребра, которое сформировало причину
                    ind = get_edge_id((up_edge[0], up_edge[2]), dc_edge_list)
                    if (ind in minimal_set) or (ind in minimal_set_unverified):
                        if_suf = 1
                        break
                if if_suf == 0:
                    rand_edge = random.sample(up_edges, 1)[0]
                    rand_ind = get_edge_id((rand_edge[0], rand_edge[2]), dc_edge_list)
                    minimal_set_unverified.add(rand_ind)
        
    minimal_set = set([i for i in minimal_set if i < len(edge_list)])
    #Prune the valid unlearning set by removing redundant element from the extended part
    
    C = []
    t = 0
    while len(C) != 0 or t==0:
        C = []
        t = t+1
        shuffled_minimal_set = np.asarray(list(minimal_set))[np.random.permutation(len(minimal_set))]
        for data_id in shuffled_minimal_set:
            minimal_set.remove(data_id)
            if not check_if_in_deductive_closure(unlearn_data_id, minimal_set, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list):
                C.append(data_id)
            else:
                minimal_set.add(data_id)
    return minimal_set

    
def get_prec_rec_acc(minimal_set, unlearn_ind):
    minimal_set_ind = np.zeros(len(unlearn_ind))
    minimal_set_ind[list(minimal_set)] = 1
    prec = (minimal_set_ind * unlearn_ind).sum() / max(unlearn_ind.sum(), 1e-8)
    rec = (minimal_set_ind * unlearn_ind).sum() / minimal_set_ind.sum()
    acc = 1 - (unlearn_ind * (1 - minimal_set_ind)).sum() / (len(unlearn_ind) - len(minimal_set))
    return prec, rec, acc
    
# python calculate_recall_and_acc.py --unlearn_data_id 3 --input_dir example_for_evaluation    
def get_valid_unlearn_general(unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, unlearn_ind, rule_list, num_seed=10, 
                              save_dir="synthetic_data/unlearn_minimal_set" # здесь почти 400 .pt 
                              ):
    # если для данного unlearn_data_id уже есть minimal_unlearn_set в save_dir, загружаем
    if os.path.exists(f"{save_dir}/{unlearn_data_id}.pt"):
        minimal_unlearn_set = torch.load(f"{save_dir}/{unlearn_data_id}.pt")
    else:
        minimal_unlearn_list = []
        for seed in tqdm(range(num_seed)):
            minimal_set = get_minimal_nec_unlearn_and_not_included_unlearn(unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list, seed)
            minimal_unlearn_list.append(minimal_set)
        # frozenset, чтобы сделать set неизменяемым и поместить в другое множество мин.множество для забывания 
        minimal_unlearn_set = set([frozenset(minimal_set) for minimal_set in minimal_unlearn_list])
        # сохраняем множество из минимальных множеств для забывания конкретного факта
        torch.save(minimal_unlearn_set, f"{save_dir}/{unlearn_data_id}.pt")
    minimal_unlearn_set = list(minimal_unlearn_set)
    precision_list = []
    recall_list = []
    acc_list = []
    for minimal_set in minimal_unlearn_set:
        # unlearn_ind = np.array из 0 и 1, где 0 значит, что факт не забыт
        prec, rec, acc = get_prec_rec_acc(minimal_set, unlearn_ind)
        precision_list.append(prec)
        recall_list.append(rec)
        acc_list.append(acc)
    
    return precision_list, recall_list, acc_list, minimal_unlearn_set

def get_edge_id(edge, edge_list):
    for i, _edge in enumerate(edge_list):
        if _edge == edge:
            return i
        
# edge_list, edge_type_list, rule_list, person_list = (69, 67), father, Sloane Lee, <utils_data_building.Person object at 0x7e13e265ea80>    
def get_deductive_closure(edge_list, edge_type_list, rule_list, person_list):
    """
    Если я правильно понимаю, функция работает, пока не найдет все факты, которые только можно логически вывести из уже имеющихся
    И в dc_edge_list складывает все возможные связи (как старые, так и снова выведенные), (69, 67)
    В dc_edge_type_list аналогично хранятся названия типов связей, 'father'
    """
    dc_edge_list, dc_edge_type_list = deepcopy(edge_list), deepcopy(edge_type_list)
    new_edge_list = []
    new_edge_type_list = []
    cur_iter=0
    while len(new_edge_list) > 0 or cur_iter == 0:
        new_edge_list = [] 
        new_edge_type_list = []
        for rule in rule_list:
            # обходим все rule и выходим, все факты вывели т.о. (факты из имеющихся выводятся в get_dc_edges_list)
            _new_edge_list, _new_edge_type_list = rule.get_dc_edges_list(dc_edge_list, dc_edge_type_list, person_list)
            dc_edge_list = dc_edge_list + _new_edge_list
            dc_edge_type_list = dc_edge_type_list + _new_edge_type_list
            
            new_edge_list = new_edge_list + _new_edge_list
            new_edge_type_list = new_edge_type_list + _new_edge_type_list
            
        cur_iter += 1
    return dc_edge_list, dc_edge_type_list