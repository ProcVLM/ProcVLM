"""
CUDA_VISIBLE_DEVICES=0,1,2,3 pytest -q tests/test_model_loading.py
"""

import os

import pytest

from evqa.model import load_procvlm


def _get_env(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    return None


def test_load_procvlm_cached():
    model_path = _get_env("PROCVLM_MODEL_PATH")
    if not model_path:
        pytest.skip("PROCVLM_MODEL_PATH is not set")

    print(f"Testing model loading cache with model path: {model_path}")
    model_a, processor_a = load_procvlm(model_path)

    print("Loading the model again to test caching...")
    model_b, processor_b = load_procvlm(model_path)

    assert model_a is model_b
    assert processor_a is processor_b


if __name__ == "__main__":
    if _get_env("PROCVLM_MODEL_PATH") is None:
        raise SystemExit("PROCVLM_MODEL_PATH is not set")
    test_load_procvlm_cached()
    print("Model loading cache test passed.")
