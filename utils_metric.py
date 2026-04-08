import os
import torch
import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy

# minimal_set включает сам unlearn_data_id
def check_if_in_deductive_closure(unlearn_data_id, minimal_set, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list):
    """
    Функция проверяет, можно ли вывести целевой факт из минимального мн-ва minimal_set после удаления оттуда data_id. 
    Передаваемое minimal_set уже построено по конкретному unlearn_data_id для забывания
    """
    # создается мн-во из id фактов в minimal_set + id выведенных искуственно с пом. get_dc_edges_list фактов (range(start, stop)), напр. range(400, 410)
    cur_minimal_set = set(list(deepcopy(minimal_set)) + list(range(len(edge_list), len(dc_edge_list))))
    
    new_added_id_list = []
    t = 0 # чтобы 1-й раз зайти в цикл 
    while len(new_added_id_list) > 0 or t == 0:
        new_added_id_list = []
        t = t + 1
        # Перебор id фактов из cur_minimal_set
        for cur_unlearn_data_id in cur_minimal_set:
            unlearn_edge = dc_edge_list[cur_unlearn_data_id] # получаем из dc_edge_list номер грани в формате (10, 20)
            unlearn_edge_type = dc_edge_type_list[cur_unlearn_data_id] # получаем тип грани в формате father
            # для каждого правила проверяем, является ли father следствием в правиле, если да, складываем в rule_set_related
            rule_set_related = [rule for rule in rule_list if rule.right_tuple[1] == unlearn_edge_type]  
            if_deducted = False
            # по следствию каждого правила ищем причины
            for rule in rule_set_related:
                if if_deducted:
                    break
                # на основании каких цепочек причин выводится текущий факт, допустим (10, 'uncle', 30), выясняем с помощью правил
                # (может быть несколько цепочек с причинами, как здесь [[(10, father, 20), (20, brother, 30)], [(10, 'brother', 15), (15, 'father', 30)]] ->)
                up_edges_list = rule.get_up_edges_list(dc_edge_list, dc_edge_type_list, unlearn_edge, unlearn_edge_type)
                # for [(10, father, 20), (20, brother, 30)] in [[(10, father, 20), (20, brother, 30)], [(10, 'brother', 15), (15, 'father', 30)]]:
                for up_edges in up_edges_list:
                    up_edges_if_deducted = True
                    # for (10, father, 20) in [(10, father, 20), (20, brother, 30)]:
                    for up_edge in up_edges:
                        # ind = get_edge_id((10, 20), dc_edge_list)
                        ind = get_edge_id((up_edge[0], up_edge[2]), dc_edge_list)
                        # Если этот факт есть в cur_minimal_set, (наверное это значит, что его можно вывести и он не необходим в cur_minimal_set), не попадем в if ниже и
                        # прекращаем обход текущей цепочки, напр., [(10, father, 20), (20, brother, 30)]
                        if ind in cur_minimal_set:
                            up_edges_if_deducted = False
                            break
                    # если ни одного факта из цепочки нет в cur_minimal_set, значит, их можно вывести из других, (а почему если нет в cur_minimal_set, значит что можно вывести из других?) попадаем сюда
                    # Если нашли id факта, которого еще не было в minimal_set, то по нему можно вывести cur_unlearn_data_id
                    # Кажется, тут возможен случай, когда unlearn_data_id будет выброшен как один из data_id
                    if up_edges_if_deducted:
                        if_deducted = True
                        # и добавляем его в пустой специальный список, затем по break выходим из цикла обхода цепочек причин, поднимаемся на for rule in rule_set_related: и там снова break
                        # итак, new_added_id_list содержит предпосылки правил, по которым можно вывести разные cur_unlearn_data_id из cur_minimal_set, сконструированный под конкретный unlearn_data_id
                        new_added_id_list.append(cur_unlearn_data_id)
                        break

        # Если факт добавился к множеству для забывания и был найден в new_added_id_list, он удаляется из cur_minimal_set
        # потому что в new_added_id_list попадают только те факты cur_unlearn_data_id, которые удалось вывести по правилам, но их еще не было в cur_minimal_set => они избыточны, раз выводимы
        for new_added_id in new_added_id_list:
            cur_minimal_set.remove(new_added_id)
    
    # если мы покинули while, значит, удалили все избыточные (выводимые из cur_minimal_set) факты
    # Кажется, этот участок нужен на случай, если в результате перебора data_id в первичном minimal_set был удален unlearn_data_id, чтобы вернуть его обратно
    if unlearn_data_id in cur_minimal_set:
        return False
    else:
        # если целевого факта нет в финальном cur_minimal_set и он удален перебором data_id в get_minimal_nec_unlearn_and_not_included_unlearn, вернуть его (см get_minimal_nec_unlearn_and_not_included_unlearn)
        return True              
                
    
def get_minimal_nec_unlearn_and_not_included_unlearn(unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list, seed=0):
    """
    Создает минимальное множество фактов для глубокого забывания целевого факта. То есть ищет факты, без которых
    целевой факт не вывести. Множество минимально, т.к. не содержит все вообще факты для вывода целевого,
    а содержит их случайный минимальный набор. Для поиска случ. мин. набора в коде прим-ся random.sample

    В идеале стоит разобрать на реальном ulearn_data_id
    """
    np.random.seed(seed)
    random.seed(seed)
    
    minimal_set = set([])
    minimal_set_unverified = set([unlearn_data_id]) # очередь на обработку

    
    #Find a valid unlearning set expanded from the given unlearning result.
    while len(minimal_set_unverified) >= 1: # а когда мы выйдем вообще из этого цикла?
        # print(minimal_set_unverified)
        # sorted, т.к. random.sample требует упорядоченность, потом список из 1 эл-та и сам эл-т [0]
        # во время второго прохода здесь 1 случайный id из цепочки причин
        cur_unlearn_data_id = random.sample(sorted(minimal_set_unverified), 1)[0] # именно здесь random.sample приводит к тому, что получаются разные minimal_unlearn_sets для 1 и того же факта
        # удаляем id этого элемента, minimal_set_unverified становится пустым в первый проход, после второго прохода тоже остается пустым
        minimal_set_unverified.remove(cur_unlearn_data_id)
        # и добавляем id рассматриваемого факта в minimal_set, minimal_set каждый проход while пополняется на 1 факт, т.к. из minimal_set ничего не удаляется
        minimal_set.add(cur_unlearn_data_id)
        # получаем tuple, отражающий связь для рассматриваемого id (69, 67), интересно, что ищем не в edge_list, а именно в dc_edge_list, чтобы работать со всеми фактами из замыкания
        unlearn_edge = dc_edge_list[cur_unlearn_data_id]
        # получаем название связи-ребра (father)
        unlearn_edge_type = dc_edge_type_list[cur_unlearn_data_id]
        # если рассматриваемый факт = следствие в правиле, то эти ассоциированные с фактом правила сохраняем, пример right_tuple: (1, 'husband', 0)
        # если текущий cur_unlearn_data_id не является следствием в каком-то правиле, rule_set_related = [], в коде ниже в очередь minimal_set_unverified не добавится новый rand_ind, len(minimal_set_unverified), произойдет выход из while
        rule_set_related = [rule for rule in rule_list if rule.right_tuple[1] == unlearn_edge_type] 

        # по следствию из каждого правила ищем причины. Каждое следствие можно вывести разными способами
        for rule in rule_set_related:
            # Почему здесь именно rule.get_up_edges_list, а не просто get_up_edges_list? Т.к. класс Rule не импортирован
            # Смотрим, что могло привести к факту, основываясь на разных правилах
            # То есть те факты, на основании которых можно восстановить целевой (10, 'uncle', 30), ищет
            # напр. [[(10, father, 20), (20, brother, 30)], [(10, 'brother', 15), (15, 'father', 30)]]
            up_edges_list = rule.get_up_edges_list(dc_edge_list, dc_edge_type_list, unlearn_edge, unlearn_edge_type) 
            for up_edges in up_edges_list: # for [(10, father, 20), (20, brother, 30)] in [[...]]
                if_suf = 0
                # Если хотя бы 1 составляющая цепочки причины уже есть в minimal_set или minimal_set_unverified, if_suf = 1 и ни один id из цепочки не добавляется в minimal_set_unverified
                # Это способ не добавить id в minimal_set_unverified, если id уже есть в minimal_set_unverified или в minimal_set, а из minimal_set_unverified факт все равно попадет в minimal_set в начале while
                for up_edge in up_edges: # for (10, father, 20) in [(10, father, 20), (20, brother, 30)]
                    # получаем id ребра, которое сформировало причину, из dc_edge_list расширенного мн-ва
                    ind = get_edge_id((up_edge[0], up_edge[2]), dc_edge_list) # up_edge[0] = 10, up_edge[2] = 20
                    if (ind in minimal_set) or (ind in minimal_set_unverified): # если это тот же id, что у unlearn_data_id или id из minimal_set_unverified(что значит??)
                        if_suf = 1
                        # (BREAK = защита от повторной обработки фактов, избыточности в minimal_set, бесконечных циклов while, если в графе есть циклы)
                        break # если эта причина уже есть в мин.мн-ве, выходим из цикла обхода этой причины 
                if if_suf == 0:
                    # random.sample нужен для получения разных минимальных множеств (для более честной оценки, как я понимаю)
                    # взять из up_edges = [(10, father, 20), (20, brother, 30)] 1 случайный элемент 
                    # (напр. (10, father, 20) из [(10, father, 20), (20, brother, 30)]), [0] из [(10, father, 20)] делает (10, father, 20)
                    rand_edge = random.sample(up_edges, 1)[0]
                    # получаем id факта из dc_edge_list 
                    rand_ind = get_edge_id((rand_edge[0], rand_edge[2]), dc_edge_list)
                    # кладем id причины для забываемого факта, но только один случайный из цепочки причин
                    minimal_set_unverified.add(rand_ind) # потом вернемся с этим фактом в начало while, начнем заново и начнем смотреть, что привело к этому факту


    # здесь по индексу фильтруются только оригинальные факты, выведенные логически и дополняющие edge_list до dc_edge_list исключаются   
    minimal_set = set([i for i in minimal_set if i < len(edge_list)])
    #Prune the valid unlearning set by removing redundant element from the extended part
    
    # Финальная минимизация множества - удаление избыточных фактов 
    C = []
    t = 0
    # когда len(C) == 0, выходим из while, удалили все возможные факты, 
    # t нужен, чтобы 1-й раз зайти в цикл
    while len(C) != 0 or t==0:
        C = []
        t = t+1
        # получение случайной перестановки эл-тов в minimal_set
        shuffled_minimal_set = np.asarray(list(minimal_set))[np.random.permutation(len(minimal_set))]
        # Далее выясняем, можно ли целевой факт unlearn_data_id вывести без data_id. Если можно, то удаляем data_id (?? это мб не совсем верная трактовка происходящего)
        # среди shuffled_minimal_set есть и unlearn_data_id
        for data_id in shuffled_minimal_set:
            # временное удаление факта из minimal_set
            minimal_set.remove(data_id)
            # если мы здесь, то проверяем, можно ли вывести unlearn_data_id без data_id
            if not check_if_in_deductive_closure(unlearn_data_id, minimal_set, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list):
                # значит, data_id избыточен и из minimal_set его можно удалить навсегда
                C.append(data_id)
            else:
                # если мы здесь, data_id не избыточен и остается в minimal_set (ВРОДЕ ТУТ ВЕРНЕТСЯ УДАЛЕННЫЙ UNLEARN_DATA_ID)
                minimal_set.add(data_id)
    # таким образом, minimal_set здесь точно не станет больше, но может уменьшиться
    return minimal_set

# unlearn_ind = массив из 0 и 1 размером 400 (т.к. строится на основе rel_ind, а relationships 400), где 1 = факт успешно забыт после unlearning
def get_prec_rec_acc(minimal_set, unlearn_ind):
    # создаем np.array из 0 размером 400
    minimal_set_ind = np.zeros(len(unlearn_ind))
    # 1 ставим у элементов, номера которых есть в minimal_set, т.е. которые должны быть забыты
    minimal_set_ind[list(minimal_set)] = 1
    prec = (minimal_set_ind * unlearn_ind).sum() / max(unlearn_ind.sum(), 1e-8)
    # minimal_set_ind * unlearn_ind показывает, сколько забыто фактов из мин.мн-ва для забывания
    # То есть по поданному id забываемого факта формируется мин.мн-во для забывания и далее смотрим, что забылось из того, что должно
    # В unlearn_ind 1 на тех местах, которые забылись, в minimal_set_ind 1 на тех местах, которые должны забыться
    # то есть на позициях minimal_set_ind
    rec = (minimal_set_ind * unlearn_ind).sum() / minimal_set_ind.sum()
    # 1 - (сколько забыто среди того, что не надо забывать/размер мн-ва того, что забывать не надо)
    # Чем больше лишнего забыто, тем ниже accuracy
    acc = 1 - (unlearn_ind * (1 - minimal_set_ind)).sum() / (len(unlearn_ind) - len(minimal_set))
    return prec, rec, acc
      
def get_valid_unlearn_general(unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, unlearn_ind, rule_list, num_seed=10, 
                              save_dir="synthetic_data/unlearn_minimal_set" # здесь почти 400 .pt отдельных файлов
                              ):
    # если для данного unlearn_data_id уже есть minimal_unlearn_set в save_dir, загружаем
    if os.path.exists(f"{save_dir}/{unlearn_data_id}.pt"):
        minimal_unlearn_set = torch.load(f"{save_dir}/{unlearn_data_id}.pt", weights_only=False)
    # иначе получаем мин.мн-во для забывания требуемого факта
    else:
        minimal_unlearn_list = []
        # Мы получаем 10 мин мн-в для забывания одного факта
        for seed in tqdm(range(num_seed)):
            minimal_set = get_minimal_nec_unlearn_and_not_included_unlearn(unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list, seed)
            minimal_unlearn_list.append(minimal_set)
        # frozenset, чтобы сделать set неизменяемым и поместить в другое множество мин.множество для забывания 
        # далеем оставляем лишь уникальные множества для забывания
        minimal_unlearn_set = set([frozenset(minimal_set) for minimal_set in minimal_unlearn_list])
        # сохраняем множество из минимальных множеств для забывания конкретного факта
        torch.save(minimal_unlearn_set, f"{save_dir}/{unlearn_data_id}.pt")
    minimal_unlearn_set = list(minimal_unlearn_set)
    precision_list = []
    recall_list = []
    acc_list = []
    for minimal_set in minimal_unlearn_set:
        # unlearn_ind = np.array из 0 и 1, где 1 значит, что факт забыт
        prec, rec, acc = get_prec_rec_acc(minimal_set, unlearn_ind)
        precision_list.append(prec)
        recall_list.append(rec)
        acc_list.append(acc)
    
    return precision_list, recall_list, acc_list, minimal_unlearn_set

# ind = get_edge_id((up_edge[0], up_edge[2]), dc_edge_list)
def get_edge_id(edge, edge_list):
    # edge = (10, 20), edge_list = [(69, 67), (10, 20), ...]
    # for 0, (69, 67), ...
    for i, _edge in enumerate(edge_list):
        if _edge == edge:
            return i
        
# edge_list, edge_type_list, rule_list, person_list = (69, 67), father, Sloane Lee, <utils_data_building.Person object at 0x7e13e265ea80>
# but for all 400 relationships 
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
        # Если какой-то факт добавился в new_edge_list в процессе обхода правил, то процедура проверки повторяется
        new_edge_list = [] 
        new_edge_type_list = []
        for rule in rule_list:
            # обходим все 48 rule и выходим, все факты вывели т.о. (факты из имеющихся выводятся в get_dc_edges_list)
            # То есть может стать не 400 фактов relationships, а больше, если логически можно вывести из 400 дополнительные факты
            _new_edge_list, _new_edge_type_list = rule.get_dc_edges_list(dc_edge_list, dc_edge_type_list, person_list)
            dc_edge_list = dc_edge_list + _new_edge_list
            dc_edge_type_list = dc_edge_type_list + _new_edge_type_list
            
            # new_edge_list вроде вводится для доп проверки. Если при обходе последнего правила в цикле for rule in rule_list выведены новые факты,
            # т.е. len(new_edge_list) > 0 при проверке окажется, то обход правил начнется заново. Иначе выйдем из while
            new_edge_list = new_edge_list + _new_edge_list
            new_edge_type_list = new_edge_type_list + _new_edge_type_list
            
        cur_iter += 1
    return dc_edge_list, dc_edge_type_list