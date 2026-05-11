import argparse
import re
import shlex
from pathlib import Path
from typing import List, Sequence, Tuple

from evqa.model import load_procvlm


_INLINE_IMAGE_PATTERN = re.compile(r"\[img=(.*?)\]")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive CLI chat for ProcVLM with optional image inputs.",
    )
    parser.add_argument("--model-path", required=True, help="Path to ProcVLM checkpoint")
    parser.add_argument("--device-map", default="auto", help="Model device map (default: auto)")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype (default: auto)")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max new tokens per turn")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Initial image path(s). Can be used multiple times.",
    )
    return parser.parse_args()


def _build_prompt(history: Sequence[Tuple[str, str]], user_text: str) -> str:
    if not history:
        return user_text

    parts: List[str] = []
    for u, a in history:
        parts.append(f"User: {u}")
        parts.append(f"Assistant: {a}")
    parts.append(f"User: {user_text}")
    parts.append("Assistant:")
    return "\n".join(parts)


def _normalize_and_validate_images(paths: Sequence[str]) -> Tuple[List[str], List[str]]:
    valid: List[str] = []
    invalid: List[str] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.exists() and p.is_file():
            valid.append(str(p))
        else:
            invalid.append(raw)
    return valid, invalid


def _extract_inline_images(text: str) -> Tuple[str, List[str], List[str]]:
    candidates = [m.strip() for m in _INLINE_IMAGE_PATTERN.findall(text)]
    clean_text = _INLINE_IMAGE_PATTERN.sub("", text)
    clean_text = " ".join(clean_text.split())

    valid, invalid = _normalize_and_validate_images(candidates)
    return clean_text, valid, invalid


def _merge_images(base_images: Sequence[str], turn_images: Sequence[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for p in list(base_images) + list(turn_images):
        if p not in seen:
            seen.add(p)
            merged.append(p)
    return merged


def _print_help() -> None:
    print("Commands:")
    print("  /help                 Show help")
    print("  /quit                 Exit")
    print("  /reset                Clear dialogue history")
    print("  /auto-reset [on/off]   on: single-turn mode, off: multi-turn mode")
    print("  /image <paths...>     Replace current image list")
    print("  /addimage <paths...>  Append image(s) to current image list")
    print("  /clearimage           Clear current image list")
    print("  /show                 Show current image list and turn count")
    print("Inline image marker in normal message:")
    print("  Please describe this image: [img=./demo.jpg]")


def main() -> None:
    args = _parse_args()

    print(f"Loading model from: {args.model_path}")
    model, processor = load_procvlm(
        model_path=args.model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    print("Model loaded. Type /help for commands.")
    print("Multi-turn is OFF by default. Use /auto-reset off to enable multi-turn.")

    history: List[Tuple[str, str]] = []
    auto_reset = True
    current_images, invalid_init = _normalize_and_validate_images(args.image)
    for bad in invalid_init:
        print(f"[WARN] invalid image path ignored: {bad}")

    while True:
        try:
            user_in = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_in:
            continue

        if user_in.startswith("/"):
            tokens = shlex.split(user_in)
            cmd = tokens[0].lower()
            cmd_args = tokens[1:]

            if cmd in {"/quit", "/exit"}:
                print("Bye.")
                break
            if cmd == "/help":
                _print_help()
                continue
            if cmd == "/reset":
                history.clear()
                print("[OK] history cleared")
                continue
            if cmd == "/clearimage":
                current_images = []
                print("[OK] image list cleared")
                continue
            if cmd == "/show":
                print(f"Turns: {len(history)}")
                print(f"Auto reset: {'on' if auto_reset else 'off'}")
                print(f"Multi-turn: {'off' if auto_reset else 'on'}")
                if current_images:
                    print("Images:")
                    for p in current_images:
                        print(f"  - {p}")
                else:
                    print("Images: (none)")
                continue
            if cmd == "/auto-reset":
                if len(cmd_args) != 1 or cmd_args[0].lower() not in {"on", "off"}:
                    print("[ERR] usage: /auto-reset [on/off]")
                    continue
                auto_reset = (cmd_args[0].lower() == "on")
                if auto_reset:
                    history.clear()
                print(f"[OK] auto reset: {'on' if auto_reset else 'off'}")
                print(f"[OK] multi-turn: {'off' if auto_reset else 'on'}")
                continue
            if cmd in {"/image", "/addimage"}:
                if not cmd_args:
                    print("[ERR] please provide at least one image path")
                    continue
                valid, invalid = _normalize_and_validate_images(cmd_args)
                for bad in invalid:
                    print(f"[WARN] invalid image path ignored: {bad}")
                if cmd == "/image":
                    current_images = valid
                else:
                    current_images.extend(valid)
                print(f"[OK] active image count: {len(current_images)}")
                continue

            print(f"[ERR] unknown command: {cmd}. Type /help")
            continue

        clean_user_text, inline_images, inline_invalid = _extract_inline_images(user_in)
        for bad in inline_invalid:
            print(f"[WARN] invalid inline image path ignored: {bad}")

        turn_images = _merge_images(current_images, inline_images)
        effective_user_text = clean_user_text if clean_user_text else user_in
        history_for_prompt = [] if auto_reset else history
        prompt = _build_prompt(history_for_prompt, effective_user_text)
        item = {
            "image": turn_images,
            "conversations": [{"from": "human", "value": prompt}],
        }

        try:
            answer = model.batch_infer(
                batch_items=[item],
                processor=processor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )[0]
        except Exception as e:
            print(f"[ERR] generation failed: {e}")
            continue

        print(f"Bot> {answer}")
        if auto_reset:
            history.clear()
        else:
            history.append((user_in, answer))


if __name__ == "__main__":
    main()
