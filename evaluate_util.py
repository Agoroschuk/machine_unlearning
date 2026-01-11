from vllm import SamplingParams
import torch.nn.functional as F
import torch
from tqdm import tqdm

def eval_qa_vllm(
    dataset, # biographies of relationships in text q-a format
    model_eval, # ready for inference (prepared by vllm) model from checkpoint after unlearning
    qk="question", # значение по умолчанию, будет переопределено вызовом с question4 из vllm_eval.py
    ak="answer", # аналогично, любые параметры могут быть перопределены
    question_start_tag="[INST] ", 
    question_end_tag=" [/INST]", 
    answer_tag=""
):  
    # формирование списка промптов к каждому факту из relationships или biographies facts, значения тегов берем из конфига
    prompts = [question_start_tag + data[qk] + question_end_tag for data in dataset]
    # настройка параметров генерации
    # temperature = 0 <=> выбор токена с макс.вер-тью
    # top_p:берем мин. кол-во наиб. вероятных токенов, кумулятивная вероятность которых >= 0.6
    # max_tokens: в ответе максимум 10 токенов
    sampling_params = SamplingParams(temperature=0, top_p=0.6, max_tokens=10)
    # генерация ответов на промпты в соответствии с настройкой
    # model_eval - объект LLM, который получает и параллельно обрабатывает список промптов
    # responses - <class 'vllm.outputs.RequestOutput'>, содержит сам промпт, ответ outputs, 
    # напр. outputs: [CompletionOutput(index=0, text='1908fatherfatherfatherfatherfatherfatherfatherfather', token_ids=array('l', [1129, 2919, 11358, 11358, 11358, 11358, 11358, 11358, 11358, 11358]), cumulative_logprob=None, logprobs=None, finish_reason=length, stop_reason=None)]
    responses = model_eval.generate(prompts, sampling_params)
    # извлечение текста из каждого response
    outputs = [response.outputs[0].text for response in responses]
    # проверяется, есть ли в сгенерированном ответе (text) правильный ответ, correct = булев массив
    # здесь например, прав. ответ = 1908 [text='1908fatherfatherfatherfatherfatherfatherfatherfather']
    correct = [data[ak].lower() in output.lower() for data, output in zip(dataset, outputs)]
    return correct, responses


def eval_qa_whp(dataset, whp_model, tokenizer, max_new_tokens=10, qk="question", ak="answer", question_start_tag = "[INST] ", question_end_tag = " [/INST]", answer_tag=""):
    prompts = [question_start_tag + data[qk] + question_end_tag for data in dataset]
    output_list = []
    
    for i,prompt in enumerate(tqdm(prompts)):
        inputs = tokenizer(prompt, return_tensors="pt").to(whp_model.device)
        outputs = whp_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_list.append(predicted_text)
        
    correct= [data[ak].lower() in output.lower() for data, output in zip(dataset, output_list)]
    return correct, output_list