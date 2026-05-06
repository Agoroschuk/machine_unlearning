#!/bin/bash  # нужно, чтобы предсказуемо выполнился script.sh

# bash scripts_unlearning_methods/${unlearning_methods}.sh $target_model $unlearn_target_data_id
# bash scripts_unlearning_methods/ga.sh gpt2_xl 1  (was for 0_dropped layers)

# New call
# bash scripts_unlearning_methods/ga.sh gpt2_xl 0_dropped 25_freezed 1

# Задание переменных: номер порта для распределенного обучения и номера устройств gpu
master_port=18765;
# master_port=18764;
# devices="0,1" # 2gpu
devices="0" 
model=$1 # 1-й аргумент команд. строки (= gpt2-xl)
percent_blocks_dropped=$2 # 25% удаленных блоков например
percent_blocks_freezed=$3
unlearn_data_id=$4 # 3-й аргумент команд. строки (= 1)
# model_path=ft_model_checkpoint/ft_${model}
model_path=/content/drive/MyDrive/Unlearning/miscellaneous/ft_model_checkpoint/ft_${model}/${percent_blocks_dropped} # путь к предобученной модели
forget_loss=ga

# Забывание запускается отдельно для каждого unlearn_data_id
save_path=/content/drive/MyDrive/Unlearning/miscellaneous/unlearning_checkpoint/${forget_loss}/${model}/${percent_blocks_dropped}/${percent_blocks_freezed}/${unlearn_data_id}
mkdir -p $save_path  # создание папки для сохранения результатов

# torchrun - launcher для распределенного обучения PyTorch
# -- отделяются длинные опции, (- для коротких, пр. -la)
# Переопределяет "forget" на "forget_family.yaml", используется hydra
CUDA_VISIBLE_DEVICES=${devices} torchrun --nproc_per_node=1 --master_port=$master_port forget.py --config-name=forget_family.yaml model_family=${model} \
    unlearn_data_id=${unlearn_data_id} forget_loss=${forget_loss} model_path=${model_path} percent_blocks_dropped=${percent_blocks_dropped} \
    percent_blocks_freezed=${percent_blocks_freezed};

# для каждой поддиректории с чекпоинтами (для каждого чекпоинта посчитать с пом.vllm _responses.pt, _correct.pt)
# /*/ <=> поиск в save_path (/) директорий (/) с любым названием (*)
for cur_save_dir in ${save_path}/*/; do
    # оценка на 1 из 4 моделей с пом. vllm_eval.py
    CUDA_VISIBLE_DEVICES=${devices} python vllm_eval.py --curr_save_dir ${cur_save_dir} --model_family $model --clean_cache false; 
    
    # Маппинг имен моделей (короткая версия: полная версия с HF), -A создает ассоциативный массив model_to_modelid, далее он заполняется по принципу ключ: значение, 
    declare -A model_to_modelid=( ["llama2-7b"]="meta-llama/Llama-2-7b" ["llama3-8b"]="meta-llama/Meta-Llama-3-8B" ["gpt2_xl"]="openai-community/gpt2-xl" ["phi"]="microsoft/phi-1_5")
    model_id="${model_to_modelid[$model]}" # доступ к эл-ту ассоц.массива
    
    # Оценка способностей модели (lm-evaluation-harness) # без race
    # tasks piqa,race,mmlu  - Тесты на здравый смысл, чтение, знания
    # CUDA_VISIBLE_DEVICES=${devices} lm_eval --model vllm \
    #     --model_args pretrained=${cur_save_dir},tokenizer=${model_id},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1,max_num_seqs=256 \
    #     --tasks race,mmlu \
    #     --batch_size auto \
    #     --output_path ${cur_save_dir}
    # Очистка весов моделей (сохраняем только метрики и логи)
    rm ${cur_save_dir}/*.safetensors
    rm ${cur_save_dir}/*.json
    rm ${cur_save_dir}/*.bin
done




