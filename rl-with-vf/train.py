import os
from openai import OpenAI
from datasets import load_dataset
import torch
import verifiers as vf

"""
inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-0.5B-Instruct

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file zero3.yaml train.py
"""

model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
dataset = load_dataset('willcb/V3-wordle-test', cache_dir=None).map(lambda x: {'question': x['answer'], 'answer': sorted(x['answer'])})

parser = vf.XMLParser(['think', 'answer'], answer_field="answer")
system_prompt = f"""Respond in the following format:
{parser.get_format_str()}

Sort the string lexographically"""

def sort_reward_func(completion, answer, **kwargs) -> float:
    """
    Check if the completion is shorted    
    """
    return 1.0 if completion == answer else 0.0


rubric = vf.Rubric(funcs=[
    sort_reward_func,
    parser.get_format_reward_func(),
], weights=[1.0, 0.2])


vf_env = vf.SingleTurnEnv(
    dataset=dataset, 
    eval_dataset=dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
    max_concurrent=10
)


args = vf.grpo_defaults(run_name = "sort-text-Qwen-0.5B")
args.num_iterations = 2
args.per_device_train_batch_size = 10
args.num_generations = 10
args.gradient_accumulation_steps = 4
args.eval_strategy = 10
args.eval_steps = 10
args.max_steps = 100

model_kwargs = dict(torch_dtype = torch.float16, attn_implementation = "eager", use_cache = False)

model, tokenizer = vf.get_model_and_tokenizer(model_name, use_liger = False, model_kwargs = model_kwargs)

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args
)

trainer.train()