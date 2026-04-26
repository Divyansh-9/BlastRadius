"""
Cold-Start Supervised Fine-Tuning (SFT)
=======================================
Phase 1 of the DeepSeek R1 Training Recipe.

Rewritten to use standard Hugging Face components (transformers, peft, trl)
for maximum stability on H200 HF Jobs, removing fragile Unsloth dependencies.
"""

import sys
import argparse
import torch
from typing import Dict, Any
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig


def validate_environment(data_path: str, output_path: str):
    """Explicit runtime checks before starting the heavy lifting."""
    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CUDA is not available. GPU is required for training.")
    
    import os
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"FATAL: Dataset not found at {data_path}")
        
    try:
        os.makedirs(output_path, exist_ok=True)
        # Test writability
        test_file = os.path.join(output_path, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        raise PermissionError(f"FATAL: Output directory {output_path} is not writable. {e}")


def main():
    parser = argparse.ArgumentParser(description="Cold-Start SFT Training (Native HF)")
    parser.add_argument("--data", default="sft_data/expert_trajectories.jsonl", help="Path to jsonl trajectories")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="Base model")
    parser.add_argument("--output", default="models/sft_checkpoint", help="Output directory")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  STAGE 1: COLD-START SUPERVISED FINE-TUNING (NATIVE HF)")
    print(f"{'='*60}\n")

    # 1. Runtime Validations
    print("Validating environment...")
    validate_environment(args.data, args.output)
    
    is_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if is_bf16 else torch.float16
    print(f"CUDA BF16 Supported: {is_bf16}. Using compute dtype: {compute_dtype}")

    # 2. Load Model with Native BitsAndBytes (4-bit QLoRA)
    print("Loading model and tokenizer...")
    max_seq_length = 2048 
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
        )
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to load model {args.model}. Error: {e}")

    # Enable gradient checkpointing for VRAM savings
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 3. Attach PEFT (LoRA) Adapters
    print("Attaching LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Load and Format Dataset
    print(f"Loading dataset: {args.data}")
    try:
        dataset = load_dataset("json", data_files=args.data, split="train")
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to parse dataset {args.data}. Error: {e}")

    def formatting_prompts_func(example: Dict[str, Any]) -> Dict[str, list]:
        formatted_texts = []
        for sys_msg, usr_msg, response in zip(
            example["system_prompt"], 
            example["user_prompt"], 
            example["response"]
        ):
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg},
                {"role": "assistant", "content": response}
            ]
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            formatted_texts.append(text)
        return {"text": formatted_texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 5. Training Configuration
    training_args = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=300,
        learning_rate=2e-5,
        fp16=not is_bf16,
        bf16=is_bf16,
        logging_steps=10,
        output_dir=args.output,
        optim="adamw_torch_fused",
        dataset_text_field="text",
        max_length=max_seq_length,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        # Disable W&B for SFT — entity name mismatch causes CommError crash.
        # GRPO handles its own wandb.init() with the correct project/entity.
        report_to="none",
    )

    # 6. Execute Training
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    
    print("\nStarting SFT training...")
    try:
        # Graceful resume if checkpoint exists
        import os
        checkpoint_dir = os.path.join(args.output, "checkpoint-100") # check if any checkpoint
        resume = any(d.startswith("checkpoint-") for d in os.listdir(args.output)) if os.path.exists(args.output) else False
        trainer.train(resume_from_checkpoint=resume)
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("FATAL: Out of Memory during training. Reduce batch size or max_seq_length.")
    except Exception as e:
        raise RuntimeError(f"FATAL: Training loop failed: {e}")

    # 7. Save Artifacts
    print(f"\nSaving model to {args.output}")
    try:
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
    except Exception as e:
        raise RuntimeError(f"FATAL: Failed to save model artifacts: {e}")
    
    print("Done! The model is now ready for Stage 2: GRPO.")

if __name__ == "__main__":
    main()
