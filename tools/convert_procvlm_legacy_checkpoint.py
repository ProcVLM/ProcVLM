#!/usr/bin/env python3
"""Convert legacy ProcVLM checkpoints to a vLLM-compatible layout.

This script converts old ProcVLM checkpoint directories (legacy key namespace)
into a new layout:
- main weights: pure Qwen-style keys (no `base_model.` wrapper)
- value head weights: split into `procvlm_extra/procvlm_head.pt`

Supported input formats:
1) `procvlm_model.bin`
2) `pytorch_model.bin`
3) `pytorch_model.bin.index.json` + sharded `pytorch_model-*.bin`

Output layout:
- `pytorch_model.bin` or sharded `pytorch_model-*.bin` + new index
- `procvlm_extra/procvlm_head.pt` (if pooler/value_head params exist)
- copied non-weight metadata/tokenizer files
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch


WEIGHT_SINGLE_FILES = {
    "procvlm_model.bin",
    "pytorch_model.bin",
    "pytorch_model.safetensors",
}

WEIGHT_INDEX_FILES = {
    "pytorch_model.bin.index.json",
    "model.safetensors.index.json",
}

DEFAULT_HEAD_RELATIVE_PATH = Path("procvlm_extra") / "procvlm_head.pt"


def remap_legacy_checkpoint_key(key: str) -> str:
    """Normalize legacy ProcVLM checkpoint keys to native Qwen-compatible keys."""
    while key.startswith("base_model."):
        key = key.replace("base_model.", "", 1)

    while key.startswith("model.model."):
        key = key.replace("model.", "", 1)

    if key.startswith("model.lm_head."):
        key = key.replace("model.", "", 1)

    # Normalize legacy ProcVLM head namespace under model.*
    if key.startswith("model.pooler."):
        key = key.replace("model.", "", 1)
    if key.startswith("model.value_head."):
        key = key.replace("model.", "", 1)

    return key


def is_value_head_key(key: str) -> bool:
    return (
        key.startswith("pooler.")
        or key.startswith("value_head.")
        or key.startswith("model.pooler.")
        or key.startswith("model.value_head.")
    )


def _assert_no_head_keys(keys: List[str], context: str) -> None:
    leaked = [k for k in keys if is_value_head_key(k)]
    if leaked:
        preview = ", ".join(leaked[:5])
        raise RuntimeError(
            f"Found ProcVLM head keys in base weights ({context}): {preview}. "
            "This would break vLLM loading."
        )


_SENSITIVE_FP32_PATTERNS = (
    re.compile(r"(^|\.)norm(\.|$)"),
    re.compile(r"layernorm", re.IGNORECASE),
    re.compile(r"rms_norm", re.IGNORECASE),
    re.compile(r"(^|\.)ln_[0-9]+(\.|$)", re.IGNORECASE),
    re.compile(r"inv_freq", re.IGNORECASE),
)


def _is_floating_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype.is_floating_point


def _should_keep_fp32(key: str, policy: str) -> bool:
    if policy == "all":
        return False
    if policy == "keep_sensitive":
        return any(p.search(key) for p in _SENSITIVE_FP32_PATTERNS)
    raise ValueError(f"Unsupported cast_policy: {policy}")


def transform_tensor(
    key: str,
    tensor: torch.Tensor,
    target_dtype: str,
    cast_policy: str,
    compact_storage: bool,
) -> torch.Tensor:
    """Apply dtype transform and storage compaction to a tensor.

    Order is: optional cast -> optional clone/contiguous compaction.
    """
    out = tensor

    if target_dtype in {"bf16", "fp16"} and _is_floating_tensor(out):
        if not _should_keep_fp32(key, cast_policy):
            out = out.to(dtype=torch.bfloat16 if target_dtype == "bf16" else torch.float16)

    if compact_storage:
        # Break shared/oversized storage and save as a compact contiguous tensor.
        out = out.detach().contiguous().clone()

    return out


def split_state_dict(
    state_dict: Dict[str, torch.Tensor],
    target_dtype: str,
    cast_policy: str,
    compact_storage: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    base_state: Dict[str, torch.Tensor] = {}
    head_state: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        new_key = remap_legacy_checkpoint_key(key)
        new_value = transform_tensor(
            key=new_key,
            tensor=value,
            target_dtype=target_dtype,
            cast_policy=cast_policy,
            compact_storage=compact_storage,
        )
        if is_value_head_key(new_key):
            head_state[new_key] = new_value
        else:
            base_state[new_key] = new_value

    return base_state, head_state


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def copy_non_weight_files(src_dir: Path, out_dir: Path) -> None:
    """Copy top-level non-weight files needed for HF/vLLM loading."""
    for item in src_dir.iterdir():
        if item.is_dir():
            continue

        name = item.name
        if name in WEIGHT_SINGLE_FILES or name in WEIGHT_INDEX_FILES:
            continue
        if name.startswith("pytorch_model-") and name.endswith(".bin"):
            continue
        if name.startswith("model-") and name.endswith(".safetensors"):
            continue
        if name in {"procvlm_head.bin", "procvlm_head.pt"}:
            continue

        shutil.copy2(item, out_dir / name)


def detect_input_layout(src_dir: Path) -> Tuple[str, Path]:
    procvlm_bin = src_dir / "procvlm_model.bin"
    if procvlm_bin.exists():
        return "single", procvlm_bin

    single_bin = src_dir / "pytorch_model.bin"
    if single_bin.exists():
        return "single", single_bin

    index_bin = src_dir / "pytorch_model.bin.index.json"
    if index_bin.exists():
        return "sharded", index_bin

    raise FileNotFoundError(
        "No supported weight files found. Expected one of: "
        "procvlm_model.bin / pytorch_model.bin / pytorch_model.bin.index.json"
    )


def convert_single_file(
    weight_file: Path,
    out_dir: Path,
    head_relpath: Path,
    target_dtype: str,
    cast_policy: str,
    compact_storage: bool,
) -> Dict[str, int]:
    state_dict = torch.load(weight_file, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected checkpoint type in {weight_file}: {type(state_dict)}")

    base_state, head_state = split_state_dict(
        state_dict,
        target_dtype=target_dtype,
        cast_policy=cast_policy,
        compact_storage=compact_storage,
    )

    torch.save(base_state, out_dir / "pytorch_model.bin")

    if head_state:
        head_path = out_dir / head_relpath
        head_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(head_state, head_path)

    return {
        "base_param_count": len(base_state),
        "head_param_count": len(head_state),
        "base_total_bytes": sum(tensor_bytes(t) for t in base_state.values()),
        "head_total_bytes": sum(tensor_bytes(t) for t in head_state.values()),
    }


def convert_sharded(
    index_file: Path,
    src_dir: Path,
    out_dir: Path,
    head_relpath: Path,
    target_dtype: str,
    cast_policy: str,
    compact_storage: bool,
) -> Dict[str, int]:
    with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid weight_map in {index_file}")

    shard_names: List[str] = sorted(set(weight_map.values()))
    expected_by_shard: Dict[str, List[str]] = {}
    for key, shard in weight_map.items():
        expected_by_shard.setdefault(shard, []).append(key)

    new_weight_map: Dict[str, str] = {}
    all_head_state: Dict[str, torch.Tensor] = {}

    base_total_bytes = 0

    for shard_name in shard_names:
        shard_path = src_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard file: {shard_path}")

        shard_state = torch.load(shard_path, map_location="cpu")
        if not isinstance(shard_state, dict):
            raise TypeError(f"Unexpected checkpoint type in {shard_path}: {type(shard_state)}")

        expected_keys = expected_by_shard.get(shard_name, [])
        if not expected_keys:
            continue

        missing_keys = [k for k in expected_keys if k not in shard_state]
        if missing_keys:
            raise KeyError(
                f"Shard {shard_name} is missing keys declared in index, e.g. {missing_keys[:3]}"
            )

        # Only process keys explicitly referenced by the source index.
        # This avoids accidentally carrying auxiliary tensors into the output shards.
        indexed_shard_state = {k: shard_state[k] for k in expected_keys}

        base_shard, head_shard = split_state_dict(
            indexed_shard_state,
            target_dtype=target_dtype,
            cast_policy=cast_policy,
            compact_storage=compact_storage,
        )

        for k, v in head_shard.items():
            all_head_state[k] = v

        if not base_shard:
            continue

        out_shard_name = shard_name
        out_shard_path = out_dir / out_shard_name
        torch.save(base_shard, out_shard_path)

        for k in base_shard.keys():
            new_weight_map[k] = out_shard_name
        base_total_bytes += sum(tensor_bytes(t) for t in base_shard.values())

    if not new_weight_map:
        raise RuntimeError("No base-model weights left after conversion.")

    _assert_no_head_keys(list(new_weight_map.keys()), "weight_map")

    new_index_data = {
        "metadata": {
            "total_size": int(base_total_bytes),
        },
        "weight_map": new_weight_map,
    }
    with open(out_dir / "pytorch_model.bin.index.json", "w", encoding="utf-8") as f:
        json.dump(new_index_data, f, ensure_ascii=False, indent=2)

    if all_head_state:
        head_path = out_dir / head_relpath
        head_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_head_state, head_path)

    return {
        "base_param_count": len(new_weight_map),
        "head_param_count": len(all_head_state),
        "base_total_bytes": int(base_total_bytes),
        "head_total_bytes": sum(tensor_bytes(t) for t in all_head_state.values()),
    }


def prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dir already exists: {out_dir}. Use --overwrite to replace it."
            )
        if any(out_dir.iterdir()):
            shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert legacy ProcVLM checkpoints to vLLM-compatible Qwen layout."
    )
    parser.add_argument("--src", required=True, type=Path, help="Legacy checkpoint directory")
    parser.add_argument("--out", required=True, type=Path, help="Converted checkpoint directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )
    parser.add_argument(
        "--skip-copy-meta",
        action="store_true",
        help="Do not copy non-weight metadata/tokenizer files",
    )
    parser.add_argument(
        "--dtype",
        choices=["keep", "bf16", "fp16"],
        default="keep",
        help=(
            "Output weight dtype. 'keep' preserves source dtype. "
            "'bf16' is recommended for size reduction with minimal quality drop."
        ),
    )
    parser.add_argument(
        "--cast-policy",
        choices=["keep_sensitive", "all"],
        default="keep_sensitive",
        help=(
            "When --dtype is bf16/fp16: 'keep_sensitive' keeps norm/inv_freq in fp32 "
            "for better quality stability; 'all' casts all floating tensors."
        ),
    )
    parser.add_argument(
        "--no-compact-storage",
        action="store_true",
        help="Disable storage compaction (not recommended for legacy checkpoints).",
    )
    parser.add_argument(
        "--head-relpath",
        type=str,
        default=str(DEFAULT_HEAD_RELATIVE_PATH),
        help=(
            "Relative output path for ProcVLM head weights. "
            "Default keeps it away from model root to avoid vLLM scanning."
        ),
    )
    args = parser.parse_args()

    src_dir = args.src.resolve()
    out_dir = args.out.resolve()

    if not src_dir.exists() or not src_dir.is_dir():
        raise NotADirectoryError(f"Invalid src checkpoint directory: {src_dir}")

    prepare_output_dir(out_dir, overwrite=args.overwrite)

    layout, entry = detect_input_layout(src_dir)
    compact_storage = not args.no_compact_storage
    head_relpath = Path(args.head_relpath)
    if head_relpath.is_absolute():
        raise ValueError("--head-relpath must be relative to --out")

    if args.dtype == "keep" and args.cast_policy != "keep_sensitive":
        print("[warning] --cast-policy is ignored when --dtype=keep")

    if not args.skip_copy_meta:
        copy_non_weight_files(src_dir, out_dir)

    if layout == "single":
        stats = convert_single_file(
            entry,
            out_dir,
            head_relpath=head_relpath,
            target_dtype=args.dtype,
            cast_policy=args.cast_policy,
            compact_storage=compact_storage,
        )
    elif layout == "sharded":
        stats = convert_sharded(
            entry,
            src_dir,
            out_dir,
            head_relpath=head_relpath,
            target_dtype=args.dtype,
            cast_policy=args.cast_policy,
            compact_storage=compact_storage,
        )
    else:
        raise RuntimeError(f"Unexpected layout: {layout}")

    # Safety check: main index in output must not reference ProcVLM heads.
    out_index = out_dir / "pytorch_model.bin.index.json"
    if out_index.exists():
        with open(out_index, "r", encoding="utf-8") as f:
            out_weight_map = json.load(f).get("weight_map", {})
        if isinstance(out_weight_map, dict):
            _assert_no_head_keys(list(out_weight_map.keys()), "output index")

    report = {
        "src": str(src_dir),
        "out": str(out_dir),
        "layout": layout,
        "dtype": args.dtype,
        "cast_policy": args.cast_policy,
        "compact_storage": compact_storage,
        "head_relpath": str(head_relpath),
        **stats,
    }
    with open(out_dir / "conversion_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Conversion finished.")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
