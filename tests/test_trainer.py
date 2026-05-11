"""
python tests/test_trainer.py
"""

import torch
import sys
from pathlib import Path

# ===== Setup project root =====
project_root = Path(__file__).parent
sys.path.append(str(project_root))

print("=== Step 1: Import check ===")

try:
    import transformers
    from transformers import AutoTokenizer, AutoProcessor
    from transformers import Qwen3VLForConditionalGeneration
    print("Transformers import: OK")
except Exception as e:
    print("Transformers import: FAIL")
    raise e

try:
    import deepspeed
    print("DeepSpeed import: OK")
except Exception as e:
    print("DeepSpeed import: FAIL")
    raise e

try:
    import flash_attn
    print("FlashAttention import: OK")
except Exception as e:
    print("FlashAttention import: FAIL")
    raise e

try:
    from evqa.model import ProcVLMWithValueHead
    from evqa.train.trainer import replace_qwen2_vl_attention_class, restore_qwen_attention_class
    print("Custom modules import: OK")
except Exception as e:
    print("Custom modules import: FAIL")
    raise e

# ===== Config =====
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"  # 或替换为你的本地 checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"\nUsing device: {device}, dtype: {dtype}")

# ===== Load tokenizer & processor =====
print("\n=== Step 2: Load tokenizer & processor ===")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("Tokenizer / Processor: OK")

# ===== Load model =====
print("\n=== Step 3: Load model ===")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)

if device == "cpu":
    model = model.to(device)

model.eval()

print("Base model load: OK")

# ===== Wrap ProcVLM =====
print("\n=== Step 4: Wrap value head ===")

model = ProcVLMWithValueHead(
    model,
    tokenizer,
    value_loss_weight=1.0,
    value_dropout=0.0,
    value_noise_std=0.0,
)

model.eval()

print("ProcVLM wrapping: OK")

# ===== Dummy input =====
print("\n=== Step 5: Build dummy input ===")

inputs = tokenizer(
    "Describe the image.",
    return_tensors="pt",
)

inputs = {k: v.to(device) for k, v in inputs.items()}

inputs["labels"] = inputs["input_ids"].clone()
inputs["progress_gt"] = torch.zeros_like(
    inputs["input_ids"], dtype=torch.float32
).to(device)

print("Dummy input: OK")

# ===== Forward pass =====
print("\n=== Step 6: Forward pass test ===")

with torch.no_grad():
    outputs = model(**inputs)

print("Forward pass: OK")
print("Loss:", outputs.loss.item())

# ===== Generation test =====
print("\n=== Step 7: Generation test ===")

with torch.no_grad():
    generated = model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=5,
        use_cache=True,
    )

print("Generation: OK")
print("Generated shape:", generated.shape)

# ===== CUDA info =====
print("\n=== Step 8: CUDA check ===")

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Device:", torch.cuda.get_device_name(0))

print("\n=== All checks passed ✅ ===")