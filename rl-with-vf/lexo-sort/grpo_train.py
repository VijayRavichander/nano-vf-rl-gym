import os
from openai import OpenAI
from datasets import load_dataset
import torch
import verifiers as vf
from dotenv import load_dotenv
import wandb


"""
inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-0.5B-Instruct --enforce-eager

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file config/zero3.yaml lexo-sort/grpo_train.py
"""

load_dotenv()

wandb.init(project = "lexo-sort")

model_name = 'Qwen/Qwen2.5-0.5B-Instruct'

dataset = load_dataset('vijay-ravichander/V3-lexo-sort', split='train')

dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ['question', 'answer']]) #type: ignore

# TRAIN SIZE - 950
train_dataset = dataset.select(range(len(dataset) - 50)) #type: ignore

# EVAL SIZE - 50
eval_dataset = dataset.select(range(len(dataset) - 50, len(dataset))) #type: ignore

parser = vf.XMLParser(['think', 'answer'], answer_field="answer")

system_prompt = f"""Respond in the following format:
{parser.get_format_str()}

Sort the string lexographically without using code and give your final answer (the sorted letters) inside <answer></answer> tags"""


def reward_sort_func(completion, answer, **kwargs) -> float:
    """
    Check if the completion is sorted    
    """
    # print(f"Completion: {completion} \n Answer: {answer} \n")
    return 1.0 if parser.parse_answer(completion) == answer else 0.0


rubric = vf.Rubric(funcs=[
    reward_sort_func,
    parser.get_format_reward_func(),
], weights=[1.0, 0.2])


vf_env = vf.SingleTurnEnv(
    dataset=dataset, 
    eval_dataset=dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
    max_concurrent=100
)

args = vf.grpo_defaults(run_name = "Qwen2.5-0.5B-Lexo-Sort")
args.num_iterations = 2
args.per_device_train_batch_size = 4
args.num_generations = 8
args.gradient_accumulation_steps = 4
args.eval_strategy = 'steps'
args.eval_steps = 10
args.max_steps = 200
args.report_to = 'wandb'
args.push_to_hub = True
args.hub_strategy = "every_save"
args.save_strategy="steps"
args.save_steps=10

model_kwargs = dict(torch_dtype = torch.bfloat16, attn_implementation = "flash_attention_2", use_cache = False) #attention options: eager | flash_attention_2

model, tokenizer = vf.get_model_and_tokenizer(model_name, use_liger = True, model_kwargs = model_kwargs)

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args
)

trainer.train() 