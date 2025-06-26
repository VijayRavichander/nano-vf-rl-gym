import os 
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
import verifiers as vf

load_dotenv()

client = OpenAI(base_url = os.getenv("DEEPINFRA_API_LINK"), api_key = os.getenv("DEEPINFRA_API_KEY"));

dataset = load_dataset('willcb/V3-wordle', split = "train",  cache_dir=None).map(lambda x: {'question': x['answer'], 'answer': "".join(sorted(x['answer']))})

dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ['question', 'answer']]) #type: ignore

## REMOVE THIS TO GENERATE DATA FROM THE ENTIRE DATASET
dataset = dataset.select(range(10)) #type:ignore

## Setting the Parsers, Rubrics and Environment
parser = vf.XMLParser(['think', 'answer'], answer_field="answer")

system_prompt = f"""Respond in the following format:
{parser.get_format_str()}

Sort the string lexographically without using code and give your final answer (the sorted letters) inside <answer></answer> tags"""

def sort_reward_func(completion, answer, **kwargs) -> float:
    """
    Check if the completion is sorted    
    """
    return 1.0 if parser.parse_answer(completion) == answer else 0.0


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
    max_concurrent=100
)

results = vf_env.evaluate(client, model="deepseek-ai/DeepSeek-V3-0324", num_samples = 10, max_concurrent = 128)

dataset_dsv3 = vf_env.make_dataset(results)

dataset_dsv3 = dataset_dsv3.sort("reward", reverse=True)

dataset_dsv3.push_to_hub("V3-lexo-sort")