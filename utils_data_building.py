# Система для работы с семейными отношениями через логические правила
import numpy as np
from copy import deepcopy

class Person:
    """
    Хранит данные о людях
    """
    def __init__(self):
        self.name = None
        self.gender = gender
        self.father = None # ссылка на отца
        self.mother = None
        self.children = None
        self.husband = None
        self.wife = None


class Rule:
    """
    Хранит логику вывода связей между людьми
    """
    def __init__(self, left_tuples, right_tuple):
        """
        num_var - количество переменных в правиле, aka число вершин

        Пример:
        all_tuples = left_tuples + [right_tuple]
        [(0, 'father', 1), (1, 'brother', 2), (0, 'uncle', 2)]
        num_var = 3 
        Я так понимаю, что left_tuples и right_tuple должны уже характеризовать 
        корректное правило
        """
        self.left_tuples = left_tuples
        self.right_tuple = right_tuple
        self.num_var = max(max(tup[0], tup[2]) for tup in left_tuples + [right_tuple]) + 1
        
    def get_up_edges_list(self, edge_list, edge_type_list, unlearn_edge, unlearn_edge_type):
        """
        Восходящий поиск

        Если (X отец Y) и (Y отец Z), то (X дед Z)
        Мы знаем: (0 дед 2) - хотим найти КАК это получилось
        edge_list - существующие связи: [(0,1), (1,2), (3,4)]
        edge_type_list - типы связей: ['father', 'brother', 'husband']
        unlearn_edge - конкретная связь которую объясняем: (0, 2) 
        unlearn_edge_type - тип этой связи: 'uncle'
        """
        # пример ниже расписан для данных из головы, не данных из docstring выше
        source_type_dict = {}
        type_target_dict = {}
        for edge, edge_type in zip(edge_list, edge_type_list):
            source_type = (edge[0], edge_type) #(0, father)
            if source_type in source_type_dict.keys():
                source_type_dict[source_type].append(edge[1]) #(0, father): 2 '2 = отец для 0'
                # но тогда у 0 есть 2 отца ))))))
            else:
                source_type_dict[source_type] = [edge[1]] # (0, father): 1 '1 = отец для 0' 
                
            type_target = (edge_type, edge[1]) # (father, 1)
            if type_target in type_target_dict.keys():
                type_target_dict[type_target].append(edge[0]) # (father, 1): 7 '1 = отец для 7'
            else:
                type_target_dict[type_target] = [edge[0]] # (father, 1): 0  '1 = отец для 0'
        
        var_value = -np.ones(self.num_var) # делает массив из -1 размером num_var
        var_value[self.right_tuple[0]] = unlearn_edge[0] # 0
        var_value[self.right_tuple[2]] = unlearn_edge[1] # 2
        
        dc_var_value_list = []
        
        # Вроде как ищем все, что приводит к забываемому факту здесь
        def _get_up_edges_list(cur_var_value):
            if (cur_var_value == -1).sum() == 0:
                for tup in self.left_tuples + [self.right_tuple]:
                    if (cur_var_value[tup[0]], tup[1]) not in source_type_dict.keys():
                        return
                    if cur_var_value[tup[2]] not in source_type_dict[(cur_var_value[tup[0]], tup[1])]:
                        return
                
                if not any(np.array_equal(cur_var_value, unique_arr) for unique_arr in dc_var_value_list):
                    dc_var_value_list.append(cur_var_value)
                return
            
            for tup in self.left_tuples:
                if cur_var_value[tup[0]] == -1 and cur_var_value[tup[2]] != -1:
                    if (tup[1], cur_var_value[tup[2]]) in type_target_dict.keys():
                        for potential_tup0_val in type_target_dict[(tup[1], cur_var_value[tup[2]])]:
                            new_cur_var_value = deepcopy(cur_var_value)
                            new_cur_var_value[tup[0]] = potential_tup0_val
                            _get_up_edges_list(new_cur_var_value)
                elif cur_var_value[tup[2]] == -1 and cur_var_value[tup[0]] != -1:
                    if (cur_var_value[tup[0]], tup[1]) in source_type_dict.keys():
                        for potential_tup0_val in source_type_dict[(cur_var_value[tup[0]], tup[1])]:
                            new_cur_var_value = deepcopy(cur_var_value)
                            new_cur_var_value[tup[2]] = potential_tup0_val
                            _get_up_edges_list(new_cur_var_value)
        
        _get_up_edges_list(var_value)
        
        up_edges_list = []
        for dc_var_value in dc_var_value_list:
            up_edges = []
            for tup in self.left_tuples:
                up_edges.append((dc_var_value[tup[0]], tup[1], dc_var_value[tup[2]]))
            up_edges_list.append(up_edges)
            
        return up_edges_list
    
    def get_dc_edges_list(self, edge_list, edge_type_list, person_list):
        """
        Нисходящий поиск
        """
        source_type_dict = {}
        type_target_dict = {}
        for edge, edge_type in zip(edge_list, edge_type_list):
            source_type = (edge[0], edge_type)
            if source_type in source_type_dict.keys():
                source_type_dict[source_type].append(edge[1])
            else:
                source_type_dict[source_type] = [edge[1]]
                
            type_target = (edge_type, edge[1])
            if type_target in type_target_dict.keys():
                type_target_dict[type_target].append(edge[0])
            else:
                type_target_dict[type_target] = [edge[0]]
        
        dc_var_value_list = []
        def _get_right_edges_list(cur_var_value):
            if (cur_var_value == -1).sum() == 0:
                for tup in self.left_tuples:
                    if (cur_var_value[tup[0]], tup[1]) not in source_type_dict.keys():
                        return
                    if cur_var_value[tup[2]] not in source_type_dict[(cur_var_value[tup[0]], tup[1])]:
                        return
                if not any(np.array_equal(cur_var_value, unique_arr) for unique_arr in dc_var_value_list):
                    dc_var_value_list.append(cur_var_value)
                return
            
            for tup in self.left_tuples:
                if cur_var_value[tup[0]] == -1 and cur_var_value[tup[2]] != -1:
                    if (tup[1], cur_var_value[tup[2]]) in type_target_dict.keys():
                        for potential_tup0_val in type_target_dict[(tup[1], cur_var_value[tup[2]])]:
                            new_cur_var_value = deepcopy(cur_var_value)
                            new_cur_var_value[tup[0]] = potential_tup0_val
                            _get_right_edges_list(new_cur_var_value)
                elif cur_var_value[tup[2]] == -1 and cur_var_value[tup[0]] != -1:
                    if (cur_var_value[tup[0]], tup[1]) in source_type_dict.keys():
                        for potential_tup0_val in source_type_dict[(cur_var_value[tup[0]], tup[1])]:
                            new_cur_var_value = deepcopy(cur_var_value)
                            new_cur_var_value[tup[2]] = potential_tup0_val
                            _get_right_edges_list(new_cur_var_value)
        
        for edge, edge_type in zip(edge_list, edge_type_list):
            if edge_type == self.left_tuples[0][1]:
                var_value = -np.ones(self.num_var)
                var_value[self.left_tuples[0][0]] = edge[0]
                var_value[self.left_tuples[0][2]] = edge[1]
                _get_right_edges_list(var_value)
        
        
        new_edge_list = []
        new_edge_type_list = []
        
        for dc_var_value in dc_var_value_list:
            new_edge = (dc_var_value[self.right_tuple[0]], dc_var_value[self.right_tuple[2]])
            new_edge_type = (self.right_tuple[1])
            
            if (dc_var_value[self.right_tuple[0]], self.right_tuple[1]) in source_type_dict.keys():
                if dc_var_value[self.right_tuple[2]] in source_type_dict[(dc_var_value[self.right_tuple[0]], self.right_tuple[1])]:
                    continue
            
            if self.right_tuple[1] in ["husband", "uncle", "father", "brother", "nephew"]:
                if person_list[int(dc_var_value[self.right_tuple[2]])].gender != "male":
                    continue
                if self.right_tuple[1] == "husband" and person_list[int(dc_var_value[self.right_tuple[0]])].gender != "female":
                    continue
                    
            elif self.right_tuple[1] in ["wife", "aunt", "mother", "sister", "niece"]:
                if person_list[int(dc_var_value[self.right_tuple[2]])].gender != "female":
                    continue
                if self.right_tuple[1] == "wife" and person_list[int(dc_var_value[self.right_tuple[0]])].gender != "male":
                    continue
            if dc_var_value[self.right_tuple[0]] == dc_var_value[self.right_tuple[2]]:
                continue
            
            new_edge_list.append(new_edge)
            new_edge_type_list.append(new_edge_type)
            
        return new_edge_list, new_edge_type_list