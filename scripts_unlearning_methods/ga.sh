#!/bin/bash  # нужно, чтобы выполнился script.sh

# sudo apt-get update && sudo apt-get install -y trash-cli # для удаления минуя корзину
# Пусть, вызвали 
# bash scripts_unlearning_methods/${unlearning_methods}.sh $target_model $unlearn_target_data_id
# bash scripts_unlearning_methods/ga.sh gpt2_xl 1

# Пайплайн для ga unlearning
# Задание переменных: номер порта для распределенного обучения и номера устройств gpu
master_port=18765;
# devices="0,1" # 2gpu
devices="0" 
model=$1 # 1-й аргумент команд. строки (= gpt2-xl)
unlearn_data_id=$2 # 2-й аргумент команд. строки (= 1)
# model_path=ft_model_checkpoint/ft_${model}
model_path=/content/drive/MyDrive/Unlearning/models/ft_model_checkpoint/ft_${model} # путь к предобученной модели
forget_loss=ga

# Забывание запускается отдельно для каждого unlearn_data_id
# save_path=unlearning_checkpoint/ga/${model}/${unlearn_data_id}
save_path_log_checkpoints=/content/drive/MyDrive/Unlearning/models/unlearning_checkpoint/log_checkpoints/ga/${model}/${unlearn_data_id}
save_path_full_checkpoints=/content/drive/MyDrive/Unlearning/models/unlearning_checkpoint/full_checkpoints/ga/${model}/${unlearn_data_id}
save_path=save_path_full_checkpoints
mkdir -p $save_path  # папка для сохранения результатов д.б. создана в google drive (parents директории создаются в случае необходимости)

# запуск скрипта забывания на 2 gpu процессах
# torchrun - launcher для распределенного обучения PyTorch
# -- отделяются опции
# Переопределяет "forget" на "forget_family.yaml"
CUDA_VISIBLE_DEVICES=${devices} torchrun --nproc_per_node=1 --master_port=$master_port forget.py --config-name=forget_family.yaml model_family=${model} unlearn_data_id=${unlearn_data_id} forget_loss=${forget_loss} model_path=${model_path}; 

# для каждой поддиректории с чекпоинтами (для каждого чекпоинта посчитать метрики)
# Тут уже должны рассчитываться метрики => проблема д.б. с сохранением checkpoint в forget.py
for cur_save_dir in ${save_path}/*/; do
    # оценка на 1 из 4 моделей с пом. vllm_eval.py
    CUDA_VISIBLE_DEVICES=${devices} python vllm_eval.py --curr_save_dir ${cur_save_dir} --model_family $model --clean_cache false; 
    
    # Маппинг имен моделей
    declare -A model_to_modelid=( ["llama2-7b"]="meta-llama/Llama-2-7b" ["llama3-8b"]="meta-llama/Meta-Llama-3-8B" ["gpt2-xl"]="openai-community/gpt2-xl" ["phi"]="microsoft/phi-1_5")
    model_id="${model_to_modelid[$model]}"
    
    # Оценка способностей модели (LM-eval) - вроде пока вообще не работает
    # tasks piqa,race,mmlu  - Тесты на здравый смысл, чтение, знания
    CUDA_VISIBLE_DEVICES=${devices} lm_eval --model vllm \
        --model_args pretrained=${cur_save_dir},tokenizer=${model_id},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
        --tasks piqa,race,mmlu \
        --batch_size auto \
        --output_path ${cur_save_dir}
    # Очистка весов моделей (сохраняем только метрики и логи)
    rm ${cur_save_dir}/*.safetensors
    rm ${cur_save_dir}/*.json
    rm ${cur_save_dir}/*.bin
done # конец цикла
    # Попробовать, чтобы удалялось сразу, а не попадало в корзину google drive 
    # trash-put -f ${cur_save_dir}/*.safetensors
    # trash-put -f ${cur_save_dir}/*.json
    # trash-put -f ${cur_save_dir}/*.bin

