import os
import sys
import inspect

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer
from peft import LoraConfig

from dataset_builder import build_action_dataset
from reward_functions import (
    json_format_reward,
    remediation_keyword_reward,
    safety_reward
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "models/action-agent-grpo"


def make_grpo_config():
    allowed_args = inspect.signature(GRPOConfig).parameters

    desired_args = {
        "output_dir": OUTPUT_DIR,
        "learning_rate": 5e-6,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_generations": 2,
        "max_completion_length": 100,
        "max_steps": 100,
        "logging_steps": 1,
        "save_steps": 50,
        "report_to": [],
        "use_vllm": False,
    }

    filtered_args = {
        key: value
        for key, value in desired_args.items()
        if key in allowed_args
    }

    print("Using GRPOConfig args:")
    print(filtered_args)

    return GRPOConfig(**filtered_args)


def main():
    dataset = build_action_dataset()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = make_grpo_config()

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[
            json_format_reward,
            remediation_keyword_reward,
            safety_reward
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)

    print("GRPO training complete.")
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
