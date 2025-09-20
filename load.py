import torch
from datasets import Dataset
dataset_relationships = Dataset.from_dict(torch.load("synthetic_data/family_relationships.pt")) #load the facts in family relationships
dataset_biographies = Dataset.from_dict(torch.load("synthetic_data/family_biographies.pt")) #load the facts in biographies


# print("Relationships dataset:", len(dataset_relationships))
# print("Biographies dataset:", len(dataset_biographies))
# print("Sample:", dataset_relationships[0])

from utils_data_building import Person, Rule
rule_list = torch.load("synthetic_data/family_rule.pt")
(edge_list,relation_list, _, _) = torch.load("synthetic_data/family-200-graph.pt") #edge_list is a list of pairs of two people; relation_list is a list of relationthips in string, e.g. child.
