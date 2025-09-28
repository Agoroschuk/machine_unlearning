#!/bin/bash

# Пайплайн для ga unlearning
master_port=18765;devices="0,1" # 2gpu
model=$1
unlearn_data_id=$2
model_path=ft_model_checkpoint/ft_${model} # путь к предобученной модели
forget_loss=ga

# Забывание запускается отдельно для каждого unlearn_data_id
save_path=unlearning_checkpoint/ga/${model}/${unlearn_data_id}
mkdir -p $save_path  # создание папки для сохранения результатов

# запуск скрипта забывания на 2 gpu процессах
# torchrun - launcher для распределенного обучения PyTorch
CUDA_VISIBLE_DEVICES=${devices} torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget_family.yaml model_family=${model} unlearn_data_id=${unlearn_data_id} forget_loss=${forget_loss} model_path=${model_path}; 

# для каждой поддиректории с чекпоинтами
for cur_save_dir in ${save_path}/*/; do
    # оценка на 1 из 4 моделей с пом. vllm_eval.py
    CUDA_VISIBLE_DEVICES=${devices} python vllm_eval.py --curr_save_dir ${cur_save_dir} --model_family $model --clean_cache false; 
    
    # Маппинг имен моделей
    declare -A model_to_modelid=( ["llama2-7b"]="meta-llama/Llama-2-7b" ["llama3-8b"]="meta-llama/Meta-Llama-3-8B" ["gpt2-xl"]="openai-community/gpt2-xl" ["phi"]="microsoft/phi-1_5")
    model_id="${model_to_modelid[$model]}"
    
    # Оценка способностей модели (LM-eval)
    CUDA_VISIBLE_DEVICES=${devices} lm_eval --model vllm \
        --model_args pretrained=${cur_save_dir},tokenizer=${model_id},tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
        --tasks piqa,race,mmlu \ # Тесты на здравый смысл, чтение, знания
        --batch_size auto \
        --output_path ${cur_save_dir}
    # Очистка весов моделей (сохраняем только метрики и логи)
    rm ${cur_save_dir}/*.safetensors
    rm ${cur_save_dir}/*.json
    rm ${cur_save_dir}/*.bin
    
    
done
