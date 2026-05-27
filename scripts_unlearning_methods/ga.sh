#!/bin/bash

# bash scripts_unlearning_methods/${unlearning_methods}.sh $target_model $unlearn_target_data_id

# single call
# bash scripts_unlearning_methods/ga.sh gpt2_xl 0_dropped 0_freezed 0
# multiple call
# bash scripts_unlearning_methods/ga.sh gpt2_xl 0_dropped 25_freezed -1 31


master_port=18765;
# master_port=18764;
# devices="0,1" # 2gpu
devices="0" 
model=$1
percent_blocks_dropped=$2
percent_blocks_freezed=$3
unlearn_data_id=$4
unlearn_data_count=$5

model_path=/content/drive/MyDrive/Unlearning/miscellaneous/ft_model_checkpoint/ft_${model}/${percent_blocks_dropped}
forget_loss=ga

# Забывание запускается либо отдельно для одного unlearn_data_id, либо сразу для нескольких unlearn_data_id
if [ "${unlearn_data_id}" = "-1" ] && [ -n "${unlearn_data_count}" ]; then
    run_name=first_${unlearn_data_count}
    extra_args="unlearn_data_count=${unlearn_data_count}"
else
    run_name=${unlearn_data_id}
    extra_args=""
fi

save_path=/content/drive/MyDrive/Unlearning/miscellaneous/unlearning_checkpoint/${forget_loss}/${model}/${percent_blocks_dropped}/${percent_blocks_freezed}/${run_name}
mkdir -p $save_path

# torchrun - launcher для распределенного обучения PyTorch
# -- отделяются длинные опции, (- для коротких, пр. -la)
# Переопределяет "forget" на "forget_family.yaml", используется hydra, далее часть аргументов меняется через override 
CUDA_VISIBLE_DEVICES=${devices} torchrun --nproc_per_node=1 --master_port=$master_port forget.py \
    --config-name=forget_family.yaml \
    model_family=${model} \
    unlearn_data_id=${unlearn_data_id} \
    forget_loss=${forget_loss} \
    model_path=${model_path} \
    percent_blocks_dropped=${percent_blocks_dropped} \
    percent_blocks_freezed=${percent_blocks_freezed} \
    save_dir=${save_path} ${extra_args};


# Маппинг имен моделей для lm-eval (короткая версия: полная версия с HF), -A создает ассоциативный массив model_to_modelid, далее он заполняется по принципу ключ: значение, 
declare -A model_to_modelid=( ["llama2-7b"]="meta-llama/Llama-2-7b" ["llama3-8b"]="meta-llama/Meta-Llama-3-8B" ["gpt2_xl"]="openai-community/gpt2-xl" ["phi"]="microsoft/phi-1_5")
model_id="${model_to_modelid[$model]}" # доступ к эл-ту ассоц.массива

# для каждой поддиректории с чекпоинтами (для каждого чекпоинта посчитать с пом.vllm _responses.pt, _correct.pt)
# /*/ <=> поиск в save_path (/) директорий (/) с любым названием (*)
for cur_save_dir in ${save_path}/*/; do
    # оценка на 1 из 4 моделей с пом. vllm_eval.py
    CUDA_VISIBLE_DEVICES=${devices} python vllm_eval.py --curr_save_dir ${cur_save_dir} --model_family $model --clean_cache false; 
    
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




