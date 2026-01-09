# Настройка данных, модели, параметров
# Но НЕ сама логика забывания!
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

import hydra #для конфигураций
import re
import transformers
from datasets import Dataset
import os
import gc
from tqdm import tqdm
from pathlib import Path # замена os.path для большей читаемости и удобства
from omegaconf import OmegaConf
import numpy as np

from data_module import FamilyForgetDataset, custom_data_collator, custom_data_collator_npo
from unlearn_trainer import CustomFamilyTrainerForgetting
from utils import get_model_identifiers_from_yaml

# подсчет числа обучаемых параметров
# если print_trainable_parameters(model) = 0, значит, все слои заморожены
def print_trainable_parameters(model):  # единственная оставленная утилитарная ф-ция
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    # named_parameters() - метод pytorch, наследуется от nn.Module
    for _, param in model.named_parameters():
        # numel() возвращает общее число эл-тов в тензоре
        all_param += param.numel() # для матрицы 100 x 10 вернет 1000
        if param.requires_grad: # особенно актуально, если какие-то слои заморожены
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# декоратор гидра делает обычной функции конфигурируемое приложение
# с управлением через конфиги и cmd, из config/forget_family.yaml (forget.yaml переопределяется на forget_family.yaml в ga.sh и др. .sh) объект cfg
# гидра забирает параметры из командной строки/bash файла или из спец. конфиг.файла .yaml (здесь смесь методов)
@hydra.main(version_base=None, config_path="config", config_name="forget_family")
def main(cfg):
    num_devices = int(os.environ.get('WORLD_SIZE', 1)) # число устройств gpu
    print(f"num_devices: {num_devices}")

    # настройка распределенного обучения, LOCAL_RANK - номер текущего gpu
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank} # чтобы каждый процесс (0 или 1) получил свою копию модели на своем gpu

    set_seed(cfg.seed) # для одинаковой инициализации весов, dropout и тд

    # os.environ - словарь с переменными окружения опер. с-мы
    os.environ["WANDB_DISABLED"] = "true"
    # model_cfg - словарь с конфигами конкретной модели из model_config.yaml
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family, cfg.config_path)
    model_id = model_cfg["model_id"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]

    os.makedirs(cfg.save_dir, exist_ok=True)
    # куда будут сохраняться результаты эксперимента
    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")


    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    #get the the unlearn_data_id in shuffled id, ПОДГОТОВКА ДАТАСЕТА для забывания
    subsample = torch.load(cfg.subsample_path, weights_only = False)
    if "family" in cfg.data_path:
        if cfg.unlearn_data_id != -1:
            # subsample содержит все 55 фактов
            shuffled_unlearn_data_id = int(subsample[cfg.unlearn_data_id])
            # FamilyForgetDataset возвращает датасет для забывания
            torch_format_dataset = FamilyForgetDataset(
                cfg.data_path, tokenizer=tokenizer, 
                model_configs=model_cfg, 
                max_length=500, # макс.длина послед-ти токенов для модели, остальное ОБРЕЗАЕТСЯ
                unlearn_data_id=shuffled_unlearn_data_id, 
                question_key='question4', 
                answer_key='answer4')
        else:
            # забывание сразу всего subsample из 55 фактов
            torch_format_dataset = FamilyForgetDataset(
                cfg.data_path, 
                tokenizer=tokenizer, 
                model_configs=model_cfg, 
                max_length=500, # этот max_length переопределит max_length в data_module
                unlearn_data_id=subsample, 
                question_key='question4', 
                answer_key='answer4')
    elif "mquake" in cfg.data_path:
        torch_format_dataset = FamilyForgetDataset(
            cfg.data_path, 
            tokenizer=tokenizer, 
            model_configs=model_cfg, 
            max_length=500, 
            unlearn_data_id=subsample, 
            question_key='question', 
            answer_key='answer')
        
    
    if cfg.lr is None:
        if cfg.forget_loss == "ga":
            lr = float(model_cfg["ga_lr"])
        elif cfg.forget_loss == "npo":
            lr = float(model_cfg["npo_lr"])
    else:
        lr = float(cfg.lr)
    
    if cfg.forget_loss == "ga":
        num_epochs = model_cfg["ga_num_epochs"]
    elif cfg.forget_loss == "npo":
        num_epochs = model_cfg["npo_num_epochs"]
    
    # расчет числа шагов обучения для правильной настройки max_steps
    batch_size = cfg.batch_size
    # чтобы обновлять веса каждые gradient_accumulation_steps, а не после каждого батча (градиенты, само собой, накапливаются после каждого батча)
    gradient_accumulation_steps = cfg.gradient_accumulation_steps

    # len(torch_format_dataset) = число примеров в забываемом датасете (1 у авторов)
    # У меня steps_per_epoch = 1, что практически отключает разогрев в обучении, но наверное есть смысл делать steps_per_epoch > 1, 
    # т.к. это определит число шагов в warmup (эвристика)
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)
    # max_steps тоже эвристика
    max_steps = int(num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")
    print(f"steps_per_epoch: {steps_per_epoch}")
    # папка logs создается через раз и всегда пустая. Чем должна быть заполнена?
    os.makedirs(f'{cfg.save_dir}/logs', exist_ok=True)
    
    # задание параметров для обучения (если передать неизвестный аргумент, будет ошибка)
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size, #кол-во примеров на трейне для оценки
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps, # накопление градиентов перед обновлением весов, дает возможность обновления раз в несколько батчей
        warmup_steps=max(1, steps_per_epoch), # постепенный рост lr к максимальному за warmup_steps
        max_steps=max_steps, # число вызовов optimizer.step(), при сложном обучении недостаточно лишь num_epochs, и этот параметр важен (но до конца не ясно)
        # ДОРАЗОБРАТЬ, ЗАЧЕМ MAX_STEPS НУЖЕН И В ЧЕМ ОТЛИЧИЕ ОТ N_EPOCHS 
        learning_rate=lr,
        bf16=True, # точность меньше, чем у fp32, но выше, чем у fp16 => скорость, градиенты сохраняются лучше
        bf16_full_eval=True, # использовать bf16 и на evaluation
        logging_steps=max(1,max_steps//20),
        logging_dir=f'{cfg.save_dir}/logs',
        output_dir=cfg.save_dir,
        # optim не определяет ф-цию потерь, а лишь оптимизирует ее
        optim="paged_adamw_32bit", #разница с AdamW только в уменьшенном использовании памяти
        save_strategy="no", # не сохранять промежуточные состояния модели каждые save_steps шагов
        ddp_find_unused_parameters= False,
        deepspeed='config/ds_config.json',
        weight_decay = cfg.weight_decay, #l2-рег. (штраф за большие веса для предотвращения переобучения)
        eval_steps = 1,
        eval_strategy = "steps",
        seed=cfg.seed,
        lr_scheduler_type="linear", # линейное уменьшение lr
    )
    
    
    #first get the base model architectur2e
    #if there is a pytorch*.bin file in the model path, then load that. use regex there can be anything in between pytorch and .bin
    # проверка, существует ли чекпоинт модели в указанном пути cfg.model_path
    # отсюда же берутся частично обученные сохраненные части модели
    path_found = False
    # cfg.model_path переопределяется в вызове .sh соответствующего метода
    # CUDA_VISIBLE_DEVICES=${devices} torchrun --nproc_per_node=1 ..... forget_loss=${forget_loss} model_path=${model_path}; 
    for file in os.listdir(cfg.model_path):
        if re.search(r"pytorch.*\.bin", file):
            path_found = True
            break
        
        if re.search(r"model-*\.safetensors", file):
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
            # use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
            attn_implementation="flash_attention_2",  #для скорости и уменьшения затрат памяти (оптимизированный мех-м внимания)
            torch_dtype=torch.bfloat16, 
            token=os.environ['HF_TOKEN'], 
            trust_remote_code = True)
    else:
        print("checkpoint not found")
        exit()  # глобально завершит весь процесс, не только forget.py
    
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    # выбор случайного токена при генерации на основе распределения вероятностей, 
    # а не токена с максимальной вероятностью, ссылка на статью, где указано, что в llama жадный алгоритм по умолчанию
    # жадная генерация дает более однообразные ответы, из-за отсутствия разнообразия оценка м.б. занижена
    # ???
    model.generation_config.do_sample = True
    
    #now we have a HuggingFace model 
    if model_cfg["gradient_checkpointing"] == "true": # стратегия сохранения памяти и не сохранения лишних градиентов в случае больших моделей (вместо сохранения пересчитывается, когда нужно)
        model.gradient_checkpointing_enable()

    # кастомный тренер (исполнитель обучения), он только для ga и npo
    trainer = CustomFamilyTrainerForgetting(
        # здесь все передаваемые аргументы = *kwargs (именованные)
        # т.к. формат name=value, а не просто value 
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        compute_metrics=None,
        args=training_args, # здесь о том, как нужно обучать
        data_collator=custom_data_collator if not cfg.forget_loss == "npo" else custom_data_collator_npo,
        # дальше именованные аргументы *kwargs (т.к. неизвестны для Trainer)
        forget_loss = cfg.forget_loss, # метод забывания ga/npo
        save_step_pattern=cfg.save_step_pattern,
        save_dir=cfg.save_dir
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    
    # особенность для npo - предварительное вычисление reference логитов
    if cfg.forget_loss == "npo":
        outputs_f_ref_dir = f"{cfg.save_dir}/outputs_f_ref.pt"
        if not os.path.exists(outputs_f_ref_dir):
            ref_model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,  # на предобученной модели делаем инференс 
                config=config, 
                use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
                torch_dtype=torch.bfloat16, 
                token=os.environ['HF_TOKEN'], 
                trust_remote_code = True)
            deepspeed_ref_model = trainer.e_prepare_deepspeed(ref_model)
            
            with torch.no_grad():
                outputs_f_ref_logit_list = []
                # для каждой пары вопрос-ответ, преобразованный к виду, доступному llm (train_dataset)
                for data_id in tqdm(range(len(trainer.train_dataset))):
                    inputs = trainer.train_dataset[data_id]
                    input_ids, labels, attention_mask = inputs[0], inputs[1], inputs[2]
                    input_ids, labels, attention_mask = input_ids.unsqueeze(0).to(local_rank), labels.unsqueeze(0).to(local_rank), attention_mask.unsqueeze(0).to(local_rank)
                    # здесь происходит инференс и выделение логитов предсказаний модели, сохраняется на cpu для сохранения памяти gpu
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
    # удаление файлов, начинающихся с global_step, после обучения
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)



if __name__ == "__main__": # чтобы forget.py выполнялся только при прямом запуске и не выполнялся при импортах (бывает в основных скриптах)
    main() # здесь декоратор гидра подставляет cfg автоматически

