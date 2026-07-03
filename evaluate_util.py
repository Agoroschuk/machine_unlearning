from vllm import SamplingParams
import torch.nn.functional as F
import torch
from tqdm import tqdm

def eval_qa_vllm(
    dataset, # biographies of relationships in text q-a format (can be also pretokenized, vllm uses HF tokenizer)
    model_eval,
    qk="question", 
    ak="answer",
    question_start_tag="[INST] ", 
    question_end_tag=" [/INST]", 
    answer_tag=""
):  
    prompts = [question_start_tag + data[qk] + question_end_tag for data in dataset]
    sampling_params = SamplingParams(temperature=0, top_p=0.6, max_tokens=10)
    # responses - <class 'vllm.outputs.RequestOutput'>, содержит сам промпт, ответ outputs, 
    # напр. outputs: [CompletionOutput(index=0, text='1908fatherfatherfatherfatherfatherfatherfatherfather', token_ids=array('l', [1129, 2919, 11358, 11358, 11358, 11358, 11358, 11358, 11358, 11358]), cumulative_logprob=None, logprobs=None, finish_reason=length, stop_reason=None)]
    responses = model_eval.generate(prompts, sampling_params)
    outputs = [response.outputs[0].text for response in responses]
    # булев массив, рез-т проверки, есть ли, напр., верный ответ = 1908 в сгенерированном тексте [text='1908fatherfatherfatherfatherfatherfatherfatherfather']
    correct = [data[ak].lower() in output.lower() for data, output in zip(dataset, outputs)]
    return correct, responses


