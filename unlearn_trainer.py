import torch
from transformers import Trainer
import torch.nn.functional as F
import os
import time
import shutil
import copy
import numpy as np

import deepspeed
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available

class CustomTrainer(Trainer): # должен подходить для обычного обучения и finetuning
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        # logits = outputs.get("logits")
        loss = outputs.loss
        # # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad(): # экономия памяти, не создается граф вычислений, также на инференсе делают
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)


# ядро забывания
class CustomFamilyTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.save_dir = kwargs.pop('save_dir')
        self.save_step_pattern = kwargs.pop('save_step_pattern')
        self.last_epoch = 0
        # наследование от родительского класса нужно, чтобы все параметры CustomFamilyTrainerForgetting инициализировались
        # Чтобы работало то, что наследуется от Trainer
        super(CustomFamilyTrainerForgetting, self).__init__(*args, **kwargs) 
        
        if self.loss_type == "npo":
            self.beta = 0.1

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_type == "ga":
            forget_inputs = inputs
            input_ids, labels, attention_mask = inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1 # за счет *-1 мы увеличиваем потери
            loss = forget_loss
            # в GD было бы
            # input_ids, labels, attention_mask = inputs
            # outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            # loss = outputs.loss  # ← БЕРЕМ ПОТЕРИ КАК ЕСТЬ
            
        elif self.loss_type == 'npo':
            # идея npo - наказывать модель за сходство со старыми логитами
            forget_inputs = inputs
            input_ids, labels, attention_mask, outputs_f_ref_logits = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            
            neg_log_ratio = outputs_f_ref_logits.to(outputs.logits.device) - outputs.logits
            loss = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        return (loss, outputs) if return_outputs else loss
        
    # инференс на 1 батче данных
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad(): # не строится граф вычислений, ускорение
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def reliable_save_model(self, curr_save_dir, max_retries:int=10, sleep_time: int=5):
        # список не полный, главное - model.safetensors
        required_files = [
            'config.json',
            'generation_config.json',
            'model.safetensors',
            'training_args.bin'
        ]
        os.makedirs(curr_save_dir, exist_ok=True)

        for attempt in range(1, max_retries + 1):
            print(f'[ReliableSave] Attempt {attempt}/{max_retries} -> Saving model to {curr_save_dir}...')
            self.save_model(curr_save_dir)
            os.sync() # просьба к ядру linux синхронизировать (записать) все буферы записи на гугл диск
            time.sleep(sleep_time) # приостановка текущего потока программы на sleep_time секунд (чтобы FUSE-драйвер google drive закончил синхронизацию)

            existing_files = os.listdir(curr_save_dir)
            missing_files = [f for f in required_files if f not in existing_files]

            if not missing_files:
                print(f'[ReliableSave] All required files found in {curr_save_dir}')
                return True
            else:
                print(f'[ReliableSave] Missing files after attempt {attempt}:{missing_files}')
                try:
                    # вспомогат.процесс для помощи FUSE
                    shutil.copy2(os.path.join(curr_save_dir, 'config.json'), curr_save_dir)
                except Exception:
                    pass
        print(f'[ReliableSave] Failed to fully save_model after {max_retries} attempts')
        print(f'Succeeded to save: {os.listdir(curr_save_dir)}')
        return False


    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        """
        Сохранение чекпоинтов модели. Обычно evaluate для оценки метрик используется
        predictions = model.predict(eval_dataset)
        calculate_perplexity(predictions) и т.д.
        Но здесь оценка по всей видимости по чекпоинтам модели делается
        """
        if self.save_step_pattern == "log":
            curr_step = self.state.global_step
            import math
            # сохранение модели на этих шагах
            if curr_step not in [1, 2, 4, 8, 16, 32]: 
                return
            # сохранение 
            curr_save_dir = os.path.join(self.save_dir, f"checkpoint-{curr_step}")
            # self.save_model(curr_save_dir)
            # более надежное сохранение на google drive, с повторениями и ожиданиями
            self.reliable_save_model(curr_save_dir)
        # сохранение модели в конце каждой эпохи
        elif self.save_step_pattern == "every_epoch":
            curr_epoch = self.state.epoch
            import math
            if int(curr_epoch) <= self.last_epoch: 
                print(curr_epoch)
                return
#             print(int(curr_epoch+0.5))
            curr_save_dir = os.path.join(self.save_dir, f"checkpoint-{int(curr_epoch)}")
            self.last_epoch = int(curr_epoch)
            # self.save_model(curr_save_dir)
            self.reliable_save_model(curr_save_dir)
        return

        
                        
    def e_prepare_deepspeed(self, model): # для эффективного распределенного обучения
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model