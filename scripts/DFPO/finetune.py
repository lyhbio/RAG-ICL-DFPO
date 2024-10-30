from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments, DataCollatorForSeq2Seq, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
import torch
import argparse
import os
from accelerate import Accelerator

def setup_parser():
    """Create a command-line argument parser"""
    
    parser = argparse.ArgumentParser(description="Pre-Finetune stage")
    
    parser.add_argument('--modelpath', 
                        type=str,
                        required=True,
                        help='Path for model requiring fine-tuning.')
    parser.add_argument('--datapreprocess', 
                        action="store_true",
                        help='Whether or not data preprocessing is performed to generate combined_data')
    
    parser.add_argument('--epoch', 
                        type=int,
                        default=5
                        )
    parser.add_argument('--batchsize', 
                        type=int,
                        default=1
                        )
    parser.add_argument('--gradient_accumulation_steps', 
                        type=str,
                        default=8
                        )
    parser.add_argument('--lr_scheduler_type', 
                        type=str,
                        default="constant_with_warmup"
                        )
    parser.add_argument('--learning_rate', 
                        type=float,
                        default=2e-4
                        )
    parser.add_argument('--output_dir', 
                        type=str,
                        default="../../result"
                        )
    return parser.parse_args()

def data_preprocess(tokenizer,num_cores,args):
    # Load datasets
    datasets = {
        'taiyi': load_dataset("Knifecat/DFPO-Preft-taiyi"),
        'adelie': load_dataset("Knifecat/DFPO-Preft-adelie")
    }

    def taiyi_filter_fields(example):
        example = example['conversation'][0]
        return {
            'instruction': example['human'],
            'input': '',
            'output': example['assistant']
        }
    
    def taiyi_process_func(example):
        
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(f"<bos><start_of_turn>user\n{example['instruction'] + example['input']}<end_of_turn>\n<start_of_turn>model\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = tokenizer(f"{example['output']}<end_of_turn>\n", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]  
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def ADELIIE_process_func(example):

        input_ids, attention_mask, labels = [], [], []
        
        system = ""  
        for message in example['messages']:
            if message["role"] == "system":
                system = message["content"].strip()
            elif message["role"] == "user":
                user = tokenizer(f'<bos><start_of_turn>user\n{system}\n{message["content"].strip()}<end_of_turn>\n<start_of_turn>model\n',add_special_tokens=False)
                input_ids+=user["input_ids"]
                labels+=[-100] * len(user["input_ids"])
                attention_mask+=user["attention_mask"]
            elif message["role"] == "assistant":
                assistant = tokenizer(f'{message["content"].strip()}<end_of_turn>\n', add_special_tokens=False)   
                input_ids+=assistant["input_ids"]
                labels+=assistant["input_ids"]
                attention_mask+=assistant["attention_mask"]

        input_ids = input_ids + [tokenizer.eos_token_id]
        attention_mask = attention_mask + [1]  
        labels = labels + [tokenizer.eos_token_id]  
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    filtered_taiyi = datasets['taiyi'].map(taiyi_filter_fields, 
                            remove_columns=datasets['taiyi']['train'].column_names, 
                            num_proc=num_cores//4)
    
    tokenized_taiyi = filtered_taiyi['train'].map(taiyi_process_func, 
                                                remove_columns=filtered_taiyi['train'].column_names, 
                                                num_proc=num_cores//4)
    tokenized_adelie = datasets['adelie']['train'].map(ADELIIE_process_func, 
                                remove_columns=datasets['adelie']['train'].column_names,
                                num_proc=num_cores//4)

    # Combine, filter, shuffle, and save combined dataset
    combined_data = concatenate_datasets([tokenized_taiyi,tokenized_adelie])
    combined_data = combined_data.filter(lambda x: len(x['input_ids']) < 3000)
    combined_data = combined_data.shuffle()
    combined_data.save_to_disk("./combined_data")
    return combined_data

accelerator = Accelerator()
device_map = {"": Accelerator().local_process_index}
args = setup_parser()
num_cores = os.cpu_count()
tokenizer = AutoTokenizer.from_pretrained(args.modelpath, use_fast=True)
# right padding
tokenizer.padding_side = 'right'
if args.datapreprocess:
    dataset = data_preprocess(tokenizer=tokenizer,num_cores=num_cores,args=args)
else:
    dataset = load_from_disk(f"{args.datapath}/combined_data")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, #
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)
model = AutoModelForCausalLM.from_pretrained(args.modelpath,  
                                             low_cpu_mem_usage=True, 
                                             torch_dtype=torch.bfloat16, 
                                             quantization_config=bnb_config,
                                             device_map=device_map, 
                                             attn_implementation='eager') # It is strongly recommended to use eager mode instead of flash_attention_2 when training gemma2.
model.enable_input_require_grads()

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, 
    r=32, 
    lora_alpha=16, 
    lora_dropout=0.1,
    bias='none',
)
model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batchsize,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    logging_steps=10,
    num_train_epochs=args.epoch,
    save_strategy='epoch',
    lr_scheduler_type=args.lr_scheduler_type,
    learning_rate=args.learning_rate,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    report_to='wandb',
    max_grad_norm=0.3,
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
