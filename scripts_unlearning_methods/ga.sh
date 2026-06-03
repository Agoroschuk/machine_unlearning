#!/bin/bash
# bash scripts_unlearning_methods/${unlearning_methods}.sh $target_model $unlearn_target_data_id

# single call
# bash scripts_unlearning_methods/ga.sh gpt2_xl 0_dropped 0_freezed 0
# multiple call
# ga
# bash scripts_unlearning_methods/ga.sh gpt2_xl 0_dropped 0_freezed -1 31
# ga + rt
# bash scripts_unlearning_methods/ga.sh gpt2_xl 0_dropped 0_freezed -1 31 combined

master_port=18765;
# master_port=18764;
# devices="0,1" # 2gpu
devices="0" 
model=$1
percent_blocks_dropped=$2
percent_blocks_freezed=$3
unlearn_data_id=$4
unlearn_data_count=$5
retain_mode=${6:-none}

model_path=/content/drive/MyDrive/Unlearning/ft_model_checkpoint/ft_${model}/${percent_blocks_dropped}
forget_loss=ga

method_name=${forget_loss}

if [ "${retain_mode}" != "none" ]; then method_name=${forget_loss}_rt
fi

# Забывание запускается либо отдельно для одного unlearn_data_id, либо сразу для нескольких unlearn_data_id
if [ "${unlearn_data_id}" = "-1" ] && [ -n "${unlearn_data_count}" ]; then
    run_name=first_${unlearn_data_count}
    extra_args="unlearn_data_count=${unlearn_data_count}"
else
    run_name=${unlearn_data_id}
    extra_args=""
fi

save_path=/content/drive/MyDrive/Unlearning/unlearning_checkpoint/${method_name}/${model}/${percent_blocks_dropped}/${percent_blocks_freezed}/${run_name}
mkdir -p $save_path

timing_file=${save_path}/runtime_seconds.tsv
echo -e "stage\tseconds" > ${timing_file}


# torchrun - launcher для распределенного обучения PyTorch
# -- отделяются длинные опции, (- для коротких, пр. -la)
# Переопределяет "forget" на "forget_family.yaml", используется hydra, далее часть аргументов меняется через override
train_start_time=$(date +%s)
CUDA_VISIBLE_DEVICES=${devices} torchrun --nproc_per_node=1 --master_port=$master_port forget.py \
    --config-name=forget_family.yaml \
    retain_mode=${retain_mode} \
    model_family=${model} \
    unlearn_data_id=${unlearn_data_id} \
    forget_loss=${forget_loss} \
    model_path=${model_path} \
    percent_blocks_dropped=${percent_blocks_dropped} \
    percent_blocks_freezed=${percent_blocks_freezed} \
    save_dir=${save_path} ${extra_args};

train_end_time=$(date +%s)
echo -e "forget\t$((train_end_time - train_start_time))" >> ${timing_file}

weight_change_file=${save_path}/weight_change_seconds.tmp
if [ -f "${weight_change_file}" ]; then
    echo -e "weight change\t$(cat "${weight_change_file}")" >> "${timing_file}"
else
    echo -e "weight change\tNA" >> "${timing_file}"
fi

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
    rm -f ${cur_save_dir}/*.safetensors
    rm -f ${cur_save_dir}/*.json
    rm -f ${cur_save_dir}/*.bin
done




