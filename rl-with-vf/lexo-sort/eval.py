import os 
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
import verifiers as vf

"""
Eval before and after training

CUDA_VISIBLE_DEVICES=0 vllm serve 'willcb/Qwen3-1.7B' --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching \
    --host 0.0.0.0 --port 8005
"""

load_dotenv()

client = OpenAI(base_url = "http://0.0.0.0:8005/v1", api_key = "API_KEY");

dataset = load_dataset('vijay-ravichander/V3-lexo-sort', split='train')

dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ['question', 'answer']]) #type: ignore

print(f"Before Filtering Dataset Size: {len(dataset)}")

## EVAL SPLIT
dataset = dataset.select(range(len(dataset) - 50, len(dataset))) #type:ignore
# dataset = dataset.select(range(0, 50)) #type:ignore


print(f"Dataset Size: {len(dataset)}")

## Setting the Parsers, Rubrics and Environment
parser = vf.XMLParser(['think', 'answer'], answer_field="answer")

system_prompt = f"""Respond in the following format:
{parser.get_format_str()}

Sort the string lexographically without using code and give your final answer (the sorted letters) inside <answer></answer> tags"""

def reward_sort_func(completion, answer, **kwargs) -> float:
    """
    Check if the completion is sorted    
    """
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

results = vf_env.evaluate(client, model="willcb/Qwen3-1.7B", num_samples = -1, max_concurrent = 128)

reward = [1 if v == 1.2 else 0 for v in results["reward"]]
acc = sum(reward) / len(reward)

print(f"Accuracy: {acc}")

