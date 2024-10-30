import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model

def setup_parser():
    """Create a command-line argument parser"""
    
    parser = argparse.ArgumentParser(description="Pre-Finetune stage")
    
    parser.add_argument('--modelpath', 
                        type=str,
                        required=True,
                        help='Path for model requiring DPO-Positive preference optimization.')
    parser.add_argument('--beta', 
                        type=float,
                        default=0.4
                        )
    parser.add_argument('--epoch', 
                        type=int,
                        default=2
                        )
    parser.add_argument('--batchsize', 
                        type=int,
                        default=1
                        )
    parser.add_argument('--max_length', 
                        type=int,
                        default=2048
                        )
    parser.add_argument('--max_prompt_length', 
                        type=int,
                        default=512
                        )
    parser.add_argument('--learning_rate', 
                        type=float,
                        default=2e-7
                        )
    parser.add_argument('--output_dir', 
                        type=str,
                        default="../../DFPO_result"
                        )
    parser.add_argument('--loss_type', 
                        type=str,
                        default="dpop"
                        )
    parser.add_argument('--dpop_lambda', 
                        type=int,
                        default=2
                        )

    return parser.parse_args()
def process_func(example):
    example["prompt"] = f"<bos><start_of_turn>user\n{example['prompt']}<end_of_turn>\n<start_of_turn>model\n"
    example["chosen"] = f"{example['chosen']}<end_of_turn>\n"
    example["rejected"] = f"{example['rejected']}<end_of_turn>\n"

    return example

args = setup_parser()
accelerator = Accelerator()
device_map = {"": Accelerator().local_process_index}
num_cores = os.cpu_count()

MFPEA = load_dataset("Knifecat/DFPO-MFPEA")
MFPEA = MFPEA['train'].shuffle()
formatted_MFEPA = MFPEA.map(process_func,num_proc=num_cores//4)
tokenizer = AutoTokenizer.from_pretrained(args.modelpath,use_fast=True)
tokenizer.padding_side = 'right'  
model = AutoModelForCausalLM.from_pretrained(args.modelpath,
                                             low_cpu_mem_usage=True, 
                                             torch_dtype=torch.bfloat16, 
                                             device_map=device_map,
                                             attn_implementation='eager')
model.enable_input_require_grads()
args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        dpop_lambda=args.dpop_lambda,
        loss_type=args.loss_type,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epoch,
        optim="paged_adamw_4bit",
        per_device_train_batch_size=args.batchsize,
        max_length=args.max_length, 
        gradient_checkpointing=True,
        max_prompt_length=args.max_prompt_length,
        save_strategy="epoch",
        logging_steps=10
        )

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, 
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.1
)

dpop_trainer = DPOTrainer(
    model,
    ref_model=None,
    args=args,
    train_dataset=formatted_MFEPA,
    loss_type=args.loss_type,
    tokenizer=tokenizer,
    peft_config=config
)
dpop_trainer.train()