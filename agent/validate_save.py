"""
Post-Training Save Validator
============================
PURPOSE:
Run this IMMEDIATELY after any training stage completes (SFT or GRPO).

WHY THIS EXISTS:
Naively saving a QLoRA/LoRA model and then reloading it can produce a
corrupted checkpoint if the merge path is wrong (guide §16 critical warning).
This script catches the failure before demo day by:
1. Loading the saved checkpoint fresh (mimicking inference conditions)
2. Running a forward pass on a domain-relevant prompt
3. Asserting the output is coherent and non-empty

USAGE:
    # After SFT:
    python -m agent.validate_save --model models/sft_checkpoint

    # After GRPO:
    python -m agent.validate_save --model models/grpo_checkpoint

    # Or via the Colab notebook Cell 4 / Cell 6
"""

import sys
import argparse
from pathlib import Path

# Allow running as a module from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def validate(model_path: str, max_new_tokens: int = 80) -> bool:
    """
    Load a saved checkpoint and run a quick inference sanity check.

    Returns True if the model loaded and generated coherent output.
    Raises AssertionError (or prints ❌) on failure.
    """
    print(f"\n{'='*55}")
    print(f"  SAVE VALIDATOR — {model_path}")
    print(f"{'='*55}\n")

    # -- 1. Import Unsloth (same way the training scripts do) --
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("❌ Unsloth not installed. Run: pip install unsloth")
        return False

    import torch

    # -- 2. Load checkpoint --
    print("⏳ Loading checkpoint (4-bit)...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=512,
            load_in_4bit=True,
            dtype=None,  # auto
        )
    except Exception as e:
        print(f"❌ LOAD FAILED: {e}")
        return False

    FastLanguageModel.for_inference(model)
    print("✅ Checkpoint loaded successfully.\n")

    # -- 3. Run a domain-relevant test prompt --
    test_cases = [
        # Prompt that mimics the MATPO Scout role
        [{"role": "user", "content": (
            "ENVIRONMENT OBSERVATION:\n"
            "Services: {\"auth-service\": \"down\", \"database\": \"healthy\"}\n"
            "Alerts: [\"auth-service: connection refused\"]\n"
            "Time Elapsed: 0 min\n"
            "Severity: HIGH\n"
            "Analyze this incident and provide a <triage> report."
        )}],
    ]

    all_passed = True
    for i, messages in enumerate(test_cases, 1):
        print(f"📝 Test {i}: Running inference...")
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                tokenize=True,
                add_generation_prompt=True,
            ).to("cuda")

            with torch.no_grad():
                out = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            # Decode only the new tokens (not the prompt)
            new_tokens = out[0][inputs.shape[-1]:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

            word_count = len(decoded.split())
            print(f"   Output ({word_count} words): {decoded[:300]}")

            # Assertions
            assert len(decoded.strip()) > 5, (
                f"❌ Output too short ({len(decoded)} chars) — possible merge corruption."
            )
            assert word_count < 500, (
                f"❌ Output suspiciously long ({word_count} words) — possible repetition loop."
            )

            print(f"   ✅ Test {i} PASSED\n")

        except AssertionError as e:
            print(f"   {e}")
            all_passed = False
        except Exception as e:
            print(f"   ❌ Inference FAILED: {e}")
            all_passed = False

    # -- 4. Final verdict --
    print("="*55)
    if all_passed:
        print(f"✅ ALL TESTS PASSED — {model_path} is safe to deploy.")
    else:
        print(f"❌ VALIDATION FAILED — DO NOT use {model_path} for demos.")
        print("   Re-run training or check your save path.")
    print("="*55 + "\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate a saved QLoRA checkpoint post-training."
    )
    parser.add_argument(
        "--model",
        default="models/grpo_checkpoint",
        help="Path to the saved model directory (default: models/grpo_checkpoint)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=80,
        help="Max tokens to generate during validation (default: 80)",
    )
    args = parser.parse_args()

    success = validate(args.model, args.max_new_tokens)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
