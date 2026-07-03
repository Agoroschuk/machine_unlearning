import yaml
import copy
import numpy as np

def get_model_identifiers_from_yaml(model_family, config_path="config"):
    model_configs  = {}
    with open(f"{config_path}/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]

def add_dataset_index(dataset): # dataset - объект HuggingFace Dataset
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)
    return dataset