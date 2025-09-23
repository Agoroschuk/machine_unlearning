#This is the implementation from muse_bench (https://github.com/swj0419/muse_bench/blob/main/baselines/baselines/task_vector.py)
# Для расчета разницы между overfitted and finetuned models
from transformers import AutoModelForCausalLM

import torch


def load_model(model_dir: str, **kwargs) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        **kwargs
    )


def compare(model1, model2) -> bool:
    """Compares two models. Поэлементно сравнивает тензоры весов

    Args:
        model1 (_type_): _description_
        model2 (_type_): _description_

    Returns:
        bool: _description_
    """
    dict1, dict2 = model1.state_dict(), model2.state_dict()
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1.keys():
        if not torch.equal(dict1[key], dict2[key]):
            return False
    return True


def unlearn(
    model_dir: str,
    out_dir: str | None = None,
    some_pt_model_dir: str | None = None,
    some_ft_model_dir: str | None = None,
    alpha: float = 1.0
):
    """
    новые_веса = веса_модели - α × (веса_дообученной - веса_предобученной)
    foriginal − α · (foverfit − foriginal)
    """
    if some_pt_model_dir is None or some_ft_model_dir is None:
        raise ValueError("Task vector (ilharco2023) requires some pretrained & finetuned models!")

    # 1. Создаем вектор задачи: что модель "научилась" при дообучении
    task_vector = TaskVector(
        pretrained_state_dict=load_model(some_pt_model_dir).state_dict(),
        finetuned_state_dict=load_model(some_ft_model_dir).state_dict()
    )

    if not task_vector.is_nonzero():
        raise ValueError("Zero task vector encountered!")

    # 2. Инвертируем вектор: получаем "вектор забывания"
    neg_task_vector = -task_vector # Здесь вызывается __neg__, без него бы python не понял, что делать
    #print("NEGATIVE VECTOR VALUE: ", type(task_vector))
    
    # 3. Загружаем целевую модель
    model = load_model(model_dir)
    # 4. Применяем отрицательный вектор с коэффициентом alpha
    new_state_dict = neg_task_vector.apply_to(pretrained_model=model, scaling_coef=alpha, in_place=False)
    del model
    # 5. Создаем новую модель с обновленными весами
    new_model = load_model(model_dir, state_dict=new_state_dict, device_map='auto')

    if out_dir is not None:
        # Сохраняем модель только если указана выходная директория
        new_model.save_pretrained(out_dir, state_dict=new_state_dict)
    return new_model


class TaskVector():
    """
    разница между finetuned и overfitted
    """
    def __init__(self,
                 pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None,
                 pretrained_state_dict=None, finetuned_state_dict=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector # если уже готов вектор
        else: # Вычисляет разницу между дообученной и предобученной моделью
            assert (
                (pretrained_checkpoint is not None and finetuned_checkpoint is not None)
                or
                (pretrained_state_dict is not None and finetuned_state_dict is not None)
            ) # если не пройти assert, код дальше не выполнится
            with torch.no_grad(): # загрузка весов модели
                if pretrained_state_dict is None:
                    pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                if finetuned_state_dict is None:
                    finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                # перебор каждого параметра отдельно
                for key in pretrained_state_dict:
                    # пропуск целочисленных весов (метаданные о размерах, идентификаторы)
                    # таким образом оставляем только обучаемые параметры
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    # ОСНОВНАЯ ФОРМУЛА: вектор_задачи = finetuned - pretrained
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    
    def __add__(self, other): # магич.метод, __add__ = операторная перегрузка для +
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other): # магич.метод
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self): # магич.метод, то же самое, что -
        """Negate a task vector.Меняет знак всех элементов"""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def is_nonzero(self): # проверка, чтобы вектор был ненулевым (защита от бессмысленных операций)
        return any([(self.vector[key] != 0).any() for key in self.vector])

    def apply_to(self, pretrained_model, scaling_coef=1.0, in_place=False):
        # scaling_coef=1.0 - это не слишком много?
        """Apply a task vector to a pretrained model."""
        print('scaling_coef:',scaling_coef)
        with torch.no_grad():
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                # Основная формула: новые_веса = старые_веса + коэффициент × вектор
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
            #torch.save(new_state_dict, "new_state_dict")
        if in_place: # Либо модифицирует существующую модель
            pretrained_model.load_state_dict(new_state_dict, strict=False)
        return new_state_dict # Либо возвращает новые веса