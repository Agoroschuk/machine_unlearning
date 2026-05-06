#!/bin/bash

# bash scripts_unlearning_methods/${unlearning_methods}.sh $target_model $unlearn_target_data_id
# bash scripts_unlearning_methods/npo.sh gpt2_xl 0_dropped 25_freezed 1 # здесь важно подать model в том же виде, что и в declare -A

model=$1
percent_blocks_dropped=$2 
percent_blocks_freezed=$3 # если передать 0_freezed, заморозки не будет, иначе будет указанный процент (учтено в forget.py)
unlearn_data_id=$4

master_port=16704
# devices="0,1"
devices='0'

# model_path=ft_model_checkpoint/ft_${model}
model_path=/content/drive/MyDrive/Unlearning/miscellaneous/ft_model_checkpoint/ft_${model}/${percent_blocks_dropped}
forget_loss=npo

# save_path=unlearning_checkpoint/${forget_loss}/${model}/${unlearn_data_id}
# save_path=/content/drive/MyDrive/Unlearning/miscellaneous/unlearning_checkpoint/${forget_loss}/${model}/${unlearn_data_id}
save_path=/content/drive/MyDrive/Unlearning/miscellaneous/unlearning_checkpoint/${forget_loss}/${model}/${percent_blocks_dropped}/${percent_blocks_freezed}/${unlearn_data_id}
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=${devices} torchrun --nproc_per_node=1 --master_port=$master_port forget.py --config-name=forget_family.yaml model_family=${model} \
    unlearn_data_id=${unlearn_data_id} forget_loss=${forget_loss} model_path=${model_path} percent_blocks_dropped=${percent_blocks_dropped} \
    percent_blocks_freezed=${percent_blocks_freezed};
 
for cur_save_dir in ${save_path}/*/; do
    CUDA_VISIBLE_DEVICES=${devices} python vllm_eval.py --curr_save_dir $cur_save_dir --model_family $model --clean_cache false;
    declare -A model_to_modelid=( ["llama2-7b"]="meta-llama/Llama-2-7b" ["llama3-8b"]="meta-llama/Meta-Llama-3-8B" ["gpt2_xl"]="openai-community/gpt2-xl" ["phi"]="microsoft/phi-1_5")
    model_id="${model_to_modelid[$model]}"

    # CUDA_VISIBLE_DEVICES=${devices} lm_eval --model vllm \
    #     --model_args pretrained=${cur_save_dir},tokenizer=${model_id},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1,max_num_seqs=256 \
    #     --tasks race,mmlu \
    #     --batch_size auto \
    #     --output_path ${cur_save_dir}
    rm ${cur_save_dir}/*.safetensors
    rm ${cur_save_dir}/*.json
    rm ${cur_save_dir}/*.bin
done