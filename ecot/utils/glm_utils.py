import torch
import json
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from typing import List, Optional, Union, Tuple, Dict, Any
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from core.utils.common import batch_multiturn_generate as __batch_multiturn_generate

### use vLLM backend ###
_cached_vllm_instance = None
_cached_vllm_processor = None
def get_vllm(model_path: str, tp: int) -> LLM:
    """
    Return a cached vLLM instance. Load model only once.
    """
    global _cached_vllm_instance, _cached_vllm_processor
    if _cached_vllm_instance is None:
        _cached_vllm_instance = LLM(model=model_path, tensor_parallel_size=tp, trust_remote_code=True)
    if _cached_vllm_processor is None:
        _cached_vllm_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return _cached_vllm_instance, _cached_vllm_processor

def batch_generate_with_vllm(
    images: List[Union[List[Image.Image], Image.Image]],
    questions: List[str],
    model_path: str,
    system_prompt: str = "You are a helpful assistant",
    max_tokens: int = 1024,
    temperature: float = 0.1,
    tp: int = 1,
) -> List[str]:
    """
    Generate text for a batch of questions using vLLM backend.
    """
    llm, processor = get_vllm(model_path, tp)
    pvi_func = process_vision_info
    # 构造 prompts
    llm_inputs_batch = []
    for image, question in zip(images, questions):
        # Construct messages in the format expected by Qwen-VL processor and process_vision_info
        if not isinstance(image, list):
            image = [image]
        message = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ] + [
                    {
                        "type": "image",
                        "image": img,
                        # Optional: min_pixels/max_pixels if your process_vision_info requires/supports them
                        # "min_pixels": 224 * 224,
                        # "max_pixels": 1280 * 28 * 28, # Example values
                    } for img in image
                ],
            },
        ]
        prompt_str = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = pvi_func(message, return_video_kwargs=True)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        current_llm_input = {
            "prompt": prompt_str,
            "multi_modal_data": mm_data,
        }
        if video_kwargs:
            current_llm_input["mm_processor_kwargs"] = video_kwargs
        llm_inputs_batch.append(current_llm_input)
    # 设置采样参数
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    outputs = llm.generate(llm_inputs_batch, sampling_params=sampling_params)
    results = []
    for batch_output in outputs:
        results.append(batch_output.outputs[0].text)
    return results

def batch_multiturn_generate_with_vllm(
    images: List[Image.Image],
    questions: List[List[str]],
    model_path: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> List[str]:
    return __batch_multiturn_generate(
        images, questions,
        model_path=model_path,
        batch_generate_func=batch_generate_with_vllm,
        max_tokens=max_tokens,
        temperature=temperature,
        __debug_mode=True,
    )

### use Hugging Face Transformers ###
# _cached_model = None
# _cached_tokenizer = None
# def get_model(model_path: str) -> tuple:
#     """
#     Return cached tokenizer and model. Load only once.
#     """
#     global _cached_model, _cached_tokenizer
#     if _cached_model is None or _cached_tokenizer is None:
#         _cached_tokenizer = AutoTokenizer.from_pretrained(model_path)
#         _cached_model = AutoModelForCausalLM.from_pretrained(model_path)
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         _cached_model.to(device)
#     return _cached_tokenizer, _cached_model
# def batch_generate(
#     questions: List[str],
#     system_prompt: str = "You are a helpful assistant",
#     model_path: str = "Qwen/Qwen3-0.6B",
#     max_tokens: int = 1024,
#     temperature: float = 0.1,
#     enable_thinking: bool = False,
# ) -> List[str]:
#     """
#     Generate text for a batch of questions using Hugging Face model (PyTorch backend).
#     """
#     tokenizer, model = get_model(model_path)
#     device = next(model.parameters()).device
#     results = []
#     for q in questions:
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": q}
#         ]
#         text = tokenizer.apply_chat_template(
#             messages, 
#             add_generation_prompt=True,
#             tokenize=False,
#             enable_thinking=enable_thinking
#         )
#         model_inputs = tokenizer([text], return_tensors="pt").to(device)
#         generated_ids = model.generate(
#             **model_inputs,
#             max_new_tokens=max_tokens,
#             temperature=temperature
#         )
#         output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
#         # parsing thinking content
#         try:
#             # rindex finding 151668 (</think>)
#             index = len(output_ids) - output_ids[::-1].index(151668)
#         except ValueError:
#             index = 0
#         # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
#         content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
#         results.append(content)
#     return results