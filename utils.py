import yaml
import copy
import numpy as np
from scipy.stats import sem, hmean, ks_2samp
from natsort import natsorted
def get_model_identifiers_from_yaml(model_family, config_path="config"):
    #path is model_configs.yaml
    '''
    models:
        llama2-7b:
            hf_key: "NousResearch/Llama-2-7b-chat-hf"  # Ключ на HuggingFace
            question_start_tag: "[INST] " # Тег начала вопроса
            question_end_tag: " [/INST] " # Тег конца вопроса
            answer_tag: ""                # Тег, добавляемый перед ответом
            start_of_sequence_token: "<s>" # Токен начала последовательности
    '''
    model_configs  = {}
    with open(f"{config_path}/model_config.yaml", "r") as f:
        # model_configs = словарь {модель:ее параметры из конфиг.файла}
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]

def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)
    return dataset