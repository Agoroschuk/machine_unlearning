# Настройка данных, модели, параметров
# Но НЕ сама логика забывания!
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

import hydra #для конфигуцраций
import transformers
from datasets import Dataset
import os
import gc
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np

from data_module import FamilyForgetDataset, custom_data_collator, custom_data_collator_npo
from unlearn_trainer import CustomFamilyTrainerForgetting
from utils import get_model_identifiers_from_yaml

# подсчет числа обучаемых параметров
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel() # для матрицы 100 x 10 вернет 1000
        if param.requires_grad: # особенно актуально, если какие-то слои заморожены
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# декоратор гидра делает из config/forget.yaml объект cfg
@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    # здесь нужно монтировать гугл диск, чтобы использовать finetuned модели
    # остальное вроде не трогать
    num_devices = int(os.environ.get('WORLD_SIZE', 1)) # число устройств gpu
    print(f"num_devices: {num_devices}")

    # настройка распределенного обучения, LOCAL_RANK - номер текущего gpu
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    set_seed(cfg.seed) # для одинаковой инициализации весов, dropout и тд

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family, cfg.config_path)
    model_id = model_cfg["model_id"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]

    # куда будут сохраняться результаты эксперимента
    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")


    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    #get the the unlearn_data_i in shuffled id, ПОДГОТОВКА ДАТАСЕТА для забывания
    subsample = torch.load(cfg.subsample_path)
    if "family" in cfg.data_path:
        if cfg.unlearn_data_id != -1:
            shuffled_unlearn_data_id = int(subsample[cfg.unlearn_data_id])
            # FamilyForgetDataset возвращает датасет для забывания
            torch_format_dataset = FamilyForgetDataset(
                cfg.data_path, tokenizer=tokenizer, 
                model_configs=model_cfg, 
                max_length=500, 
                unlearn_data_id=shuffled_unlearn_data_id, 
                question_key='question4', 
                answer_key='answer4')
        else:
            torch_format_dataset = FamilyForgetDataset(cfg.data_path, tokenizer=tokenizer, model_configs=model_cfg, max_length=500, unlearn_data_id=subsample, question_key='question4', answer_key='answer4')
    elif "mquake" in cfg.data_path:
        torch_format_dataset = FamilyForgetDataset(cfg.data_path, tokenizer=tokenizer, model_configs=model_cfg, max_length=500, unlearn_data_id=subsample, question_key='question', answer_key='answer')
        
    
    if cfg.lr is None:
        if cfg.forget_loss == "ga":
            lr = float(model_cfg["ga_lr"])
        elif cfg.forget_loss == "npo":
            lr = float(model_cfg["npo_lr"])
    else:
        lr = float(cfg.lr)
    
    # настройка параметров обучения
    if cfg.forget_loss == "ga":
        num_epochs = model_cfg["ga_num_epochs"]
    elif cfg.forget_loss == "npo":
        num_epochs = model_cfg["npo_num_epochs"]
    
    # расчет числа шагов обучения для правильной настройки max_steps
    batch_size = cfg.batch_size
    # чтобы обновлять веса каждые gradient_accumulation_steps, а не на каждом шаге
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)
    max_steps = int(num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")
    print(f"steps_per_epoch: {steps_per_epoch}")
    
    # создание аргументов для тренировки
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size, #кол-во примеров на трейне для оценки
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, # накопление градиентов перед обновлением весов
        warmup_steps=max(1, steps_per_epoch),
        max_steps=max_steps,
        learning_rate=lr,
        bf16=True, # точность меньше, чем у fp32, но выше, чем у fp16 => скорость, градиенты сохраняются лучше
        bf16_full_eval=True,
        logging_steps=max(1,max_steps//20),
        logging_dir=f'{cfg.save_dir}/logs',
        output_dir=cfg.save_dir,
        # optim не определяет ф-цию потерь, а лишь оптимизирует ее
        optim="paged_adamw_32bit", #разница с AdamW только в уменьшенном использовании памяти
        save_strategy="no",
        ddp_find_unused_parameters= False,
        deepspeed='config/ds_config.json',
        weight_decay = cfg.weight_decay, #l2-рег.
        eval_steps = 1,
        evaluation_strategy = "steps",
        seed=cfg.seed,
        lr_scheduler_type="linear", # линейное уменьшение lr
    )
    
    
    #first get the base model architectur2e
    #if there is a pytorch*.bin file in the model path, then load that. use regex there can be anything in between pytorch and .bin
    # проверка, ли существует ли чекпоинт модели в указанном пути cfg.model_path
    # отсюда же берутся частично обученные сохраненные части модели
    import re
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search("pytorch.*\.bin", file):
            path_found = True
            break
        
        if re.search("model-*\.safetensors", file):
            path_found = True
            break


    if path_found:
        # Загружает конфигурацию модели ИЗ Hugging Face, даже если веса берутся локально!
        config = AutoConfig.from_pretrained(model_id)

        print("Loading from checkpoint")
        # загрузка модели
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, # ← ЛОКАЛЬНЫЙ путь к весам "ft_model_checkpoint/ft_phi"
            config=config, # конфигурация модели из HF, хоть и есть локальная. Из HF будет полнее
            use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
            torch_dtype=torch.bfloat16, token=os.environ['HF_TOKEN'], 
            trust_remote_code = True)
    else:
        print("checkpoint not found")
        exit()
    
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    # выбор случайного токена при генерации на основе распределения вероятностей, 
    # а не токена с максимальной вероятностью, ссылка на статью, где указано, что в llama жадный алгоритм по умолчанию
    # жадная генерация дает более однообразные ответы, из-за отсутствия разнообразия оценка м.б. занижена
    # ???
    model.generation_config.do_sample = True
    
    #now we have a HuggingFace model 
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    # кастомный тренер, он только для ga и npo
    trainer = CustomFamilyTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        compute_metrics=None,
        args=training_args,
        data_collator=custom_data_collator if not cfg.forget_loss == "npo" else custom_data_collator_npo,
        forget_loss = cfg.forget_loss, # метод забывания ga/npo
        save_step_pattern=cfg.save_step_pattern,
        save_dir=cfg.save_dir
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    
    # особенность для npo - предварительное вычисление reference логитов
    if cfg.forget_loss == "npo":
        outputs_f_ref_dir = f"{cfg.save_dir}/outputs_f_ref.pt"
        if not os.path.exists(outputs_f_ref_dir):
            ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, token=os.environ['HF_TOKEN'], trust_remote_code = True)
            deepspeed_ref_model = trainer.e_prepare_deepspeed(ref_model)
            
            with torch.no_grad():
                outputs_f_ref_logit_list = []
                for data_id in tqdm(range(len(trainer.train_dataset))):
                    inputs = trainer.train_dataset[data_id]
                    input_ids, labels, attention_mask = inputs[0], inputs[1], inputs[2]
                    input_ids, labels, attention_mask = input_ids.unsqueeze(0).to(local_rank), labels.unsqueeze(0).to(local_rank), attention_mask.unsqueeze(0).to(local_rank)
                    outputs_f_ref_logit = deepspeed_ref_model(input_ids, labels=labels, attention_mask=attention_mask).logits.cpu()
                    outputs_f_ref_logit_list.append(outputs_f_ref_logit)
            outputs_f_ref_logits = torch.cat(outputs_f_ref_logit_list)
                    
            deepspeed_ref_model.destroy()
            del deepspeed_ref_model
            del ref_model
            gc.collect()
            torch.cuda.empty_cache()
            torch.save(outputs_f_ref_logits, outputs_f_ref_dir)
        trainer.train_dataset.outputs_f_ref_logits = torch.load(outputs_f_ref_dir)
    
    # запуск обучения, train() наследуется от Trainer, вызывает кастомные evaluate(), compute_loss() 
    trainer.train()

    #delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    # удаление временных файлов после обучения
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)



if __name__ == "__main__":
    main()

