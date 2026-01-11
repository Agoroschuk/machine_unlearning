import argparse
import datasets
import gc
import torch
import numpy as np
# pip show vllm (for info), https://github.com/vllm-project/vllm
from vllm import LLM  # vllm - высокопроизводительная библиотека для инференса больших языковых моделей
from vllm.distributed.parallel_state import destroy_model_parallel
from pathlib import Path
from datasets import Dataset

from utils import get_model_identifiers_from_yaml
from evaluate_util import eval_qa_vllm

parser = argparse.ArgumentParser(description='evaluate llm by vllm')
parser.add_argument('--curr_save_dir', type=str, default=None)
parser.add_argument('--model_family', type=str, default="llama2-7b")
parser.add_argument('--clean_cache', type=str, default="false")
parser.add_argument('--config_path', type=str, default="config/") # т.к. config_path из .sh файла не передан, значение его берется из default
args = parser.parse_args()

# curr_save_dir = конкретный чекпоинт в save_path=/content/drive/MyDrive/Unlearning/models/unlearning_checkpoint/ga/${model}/${unlearn_data_id}
curr_save_dir = args.curr_save_dir
# model_cfg - словарь с параметрами модели из model_config.yaml
model_cfg = get_model_identifiers_from_yaml(args.model_family, config_path=args.config_path)
model_id = model_cfg["model_id"]

#load model via llm from checkpoint, model is ready for text generation
model_eval = LLM(curr_save_dir, tokenizer=model_id, device="auto")
# eval_dataset = datasets.load_from_disk(curr_save_dir+"/eval.hf")
eval_dataset_list = [Dataset.from_dict(torch.load("synthetic_data/family_relationships.pt", weights_only=False)), Dataset.from_dict(torch.load("synthetic_data/family_biographies.pt", weights_only=False))]
eval_dataset_name_list = ["relationships_", "biographies_"]

#remove local model
if args.clean_cache == "true":
    import shutil # (shell utilities, python модуль для высокоуровневых файловых операций)
    shutil.rmtree(curr_save_dir) # удаление всей папки curr_save_dir и всего содержимого

Path(curr_save_dir).mkdir(parents=True, exist_ok=True)

# Здесь происходит инференс чекпоинтов на разных стадиях забывания (получение biographies/relationships_correct.pt)
for eval_dataset, eval_dataset_name in zip(eval_dataset_list, eval_dataset_name_list):
    with torch.no_grad():
        # correct - булев массив, где, если я правильно понимаю, True = факт сохранился
        # responses - vllm объекты с подробностями о том, как на них работало забывание (ценно)
        correct, responses = eval_qa_vllm(
            eval_dataset, # факты из биографии  (300) или взаимоотношения(400) в виде вопрос-ответ 
            model_eval, # готовая модель из чекпоинта для генерации текста
            qk="question4", 
            ak="answer4", 
            question_start_tag=model_cfg["question_start_tag"], 
            question_end_tag=model_cfg["question_end_tag"], 
            answer_tag=model_cfg["answer_tag"])
        # eval_qa_vllm формирует промпт с тегами и генерирует ответ с помощью модели model_eval
        # сохранение ответа True/False (True = модель дала правильный ответ)
        torch.save(correct, f"{curr_save_dir}/{eval_dataset_name}correct.pt")
        # подробные результаты генерации 
        # ['request_id', 'prompt', 'prompt_token_ids', 'prompt_logprobs', 'outputs', 'finished', 'metrics', 'lora_request', 'encoder_prompt', 'encoder_prompt_token_ids']
        torch.save(responses, f"{curr_save_dir}/{eval_dataset_name}responses.pt")
        # если проводить параллель с calculate_recall_and_acc.py, то правильно считать таким способом только accuracy для biographies, для relationships логика сложнее
        # и самое важное здесь = relationships_correct.pt = булев массив из того, что осталось после unlearning незабытым
        # при сопоставлении его с minimal_set для забывания этого конкретного факта можем выяснить acc_relationships, recall_relationships
        acc = np.asarray(correct).astype(np.float32).mean()
        print(f"{eval_dataset}accuracy: {acc}")

# освобождение gpu памяти и завершение параллельного окружения vLLM
destroy_model_parallel()
del model_eval
gc.collect()
torch.cuda.empty_cache()