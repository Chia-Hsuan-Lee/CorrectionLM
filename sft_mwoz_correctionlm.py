# inspired by https://github.com/huggingface/trl/tree/main/examples 
import os
import torch
import wandb
import argparse
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer
from trl.import_utils import is_npu_available, is_xpu_available


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-Tune LM with LoRA using SFTTrainer")
    parser.add_argument("--WANDB_PROJECT", type=str, required=True, help="WANDB project name")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name")
    parser.add_argument("--train_data_file", type=str, required=True, help="Path to training data JSON file")
    parser.add_argument("--valid_data_file", type=str, required=True, help="Path to validation data JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation on the validation set")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy to use")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps between evaluations")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum number of training steps")
    parser.add_argument("--logging_steps", type=int, default=1000, help="Number of steps between logging outputs")
    parser.add_argument("--save_steps", type=int, default=1000, help="Number of steps between model checkpoints")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per device during evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", type=bool, default=False, help="Whether to use gradient checkpointing")
    parser.add_argument("--group_by_length", type=bool, default=False, help="Whether to group sequences by length during training")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Maximum gradient norm")
    parser.add_argument("--bf16", type=bool, default=True, help="Whether to use bf16 precision")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit", help="Optimizer to use")
    parser.add_argument("--remove_unused_columns", type=bool, default=True, help="Whether to remove unused columns from the dataset")
    parser.add_argument("--run_name", type=str, help="Run name for wandb logging")
    parser.add_argument("--report_to", type=str, default="wandb", help="Where to report training metrics")
    return parser.parse_args()


@dataclass
class ScriptArguments:
    packing: Optional[bool] = field(default=False, metadata={"help": "whether to use packing for SFTTrainer"})
    use_bnb: Optional[bool] = field(default=True, metadata={"help": "whether to use BitsAndBytes"})

    # LoraConfig
    lora_alpha: Optional[float] = field(default=8, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})

def create_datasets(tokenizer, train_data_file, valid_data_file):
    def template_dataset(sample):
        sample["text"] = f"{sample['SFT_prompt_no_table']}{sample['SFT_completion']} [end] [end] [end] {tokenizer.eos_token}"
        return sample

    train_data = load_dataset("json", data_files=train_data_file)["train"]
    valid_data = load_dataset("json", data_files=valid_data_file)["train"]

    # Shuffle and format the dataset
    train_data = train_data.shuffle(seed=42).map(template_dataset, remove_columns=list(train_data.features))
    valid_data = valid_data.shuffle(seed=42).map(template_dataset, remove_columns=list(valid_data.features))

    return train_data, valid_data


def main():
    args = parse_args()
    os.environ["WANDB_PROJECT"] = args.WANDB_PROJECT

    hf_parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args, remaining_args = hf_parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # Update training_args with args parsed via argparse
    for key, value in vars(args).items():
        if key not in vars(training_args):
            setattr(training_args, key, value)

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    if training_args.group_by_length and script_args.packing:
        raise ValueError("Cannot use both packing and group by length")

    if training_args.gradient_checkpointing:
        raise ValueError("gradient_checkpointing not supported")

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset, eval_dataset = create_datasets(tokenizer, args.train_data_file, args.valid_data_file)

    bnb_config = None
    if script_args.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
        trust_remote_code=True,
        use_auth_token=True,
    )
    base_model.config.use_cache = False

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field='text',
        packing=script_args.packing,
        max_seq_length=1990,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del base_model
    if is_xpu_available():
        torch.xpu.empty_cache()
    elif is_npu_available():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)


if __name__ == "__main__":
    main()
