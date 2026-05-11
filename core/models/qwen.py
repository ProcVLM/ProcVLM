import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Union, Tuple, Dict, Any


### use vLLM backend ###
def batch_generate_with_vllm(
    questions: List[str],
    model_path: str,
    system_prompt: str = "You are a helpful assistant",
    max_tokens: int = 1024,
    temperature: float = 0.1,
    tp: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> List[str]:
    """
    Generate text for a batch of questions using vLLM backend.
    """
    from core.backends.vllm import SamplingParams, get_vllm
    llm, processor = get_vllm(model_path, tp, gpu_memory_utilization)
    llm_inputs_batch = []
    for question in questions:
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        prompt_str = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        current_llm_input = {
            "prompt": prompt_str,
        }
        llm_inputs_batch.append(current_llm_input)
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    outputs = llm.generate(llm_inputs_batch, sampling_params=sampling_params, use_tqdm=False)
    results = []
    for batch_output in outputs:
        thinking_content = ""
        content = ""
        # cut <think> ... </think>
        text = batch_output.outputs[0].text
        if "<think>" in text and "</think>" in text:
            thinking_content = text.split("<think>")[1].split("</think>")[0].strip()
            content = text.split("</think>")[-1].strip()
        else:
            # remove possible incomplete <think> or </think>
            if "<think>" in text:
                text = text.split("<think>")[0]
            if "</think>" in text:
                text = text.split("</think>")[-1]
            content = text.strip()
        results.append(content)
        # print("text: ", text, "\nthinking: ", thinking_content, "\ncontent: ", content)
    return results

def generate_with_vllm(
    question: str,
    model_path: str,
    system_prompt: str = "You are a helpful assistant",
    max_tokens: int = 1024,
    temperature: float = 0.1,
    tp: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> str:
    return batch_generate_with_vllm(
        [question],
        model_path,
        system_prompt,
        max_tokens,
        temperature,
        tp, gpu_memory_utilization
    )[0]



### use Hugging Face Transformers ###
_cached_model = None
_cached_tokenizer = None
def get_model(model_path: str) -> tuple:
    """
    Return cached tokenizer and model. Load only once.
    """
    global _cached_model, _cached_tokenizer
    if _cached_model is None or _cached_tokenizer is None:
        _cached_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _cached_model = AutoModelForCausalLM.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _cached_model.to(device)
    return _cached_tokenizer, _cached_model

def batch_generate(
    questions: List[str],
    model_path: str,
    system_prompt: str = "You are a helpful assistant",
    max_tokens: int = 1024,
    temperature: float = 0.1,
    enable_thinking: bool = False,
) -> List[str]:
    """
    Generate text for a batch of questions using Hugging Face model (PyTorch backend).
    """
    tokenizer, model = get_model(model_path)
    device = next(model.parameters()).device
    results = []
    for q in questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q}
        ]
        text = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        results.append(content)
    return results

def generate(
    question: str,
    model_path: str,
    system_prompt: str = "You are a helpful assistant",
    max_tokens: int = 1024,
    temperature: float = 0.1,
    enable_thinking: bool = False,
) -> str:
    """
    Generate text for a single question using Hugging Face model (PyTorch backend).
    """
    return batch_generate(
        [question],
        model_path,
        system_prompt,
        max_tokens,
        temperature,
        enable_thinking
    )[0]