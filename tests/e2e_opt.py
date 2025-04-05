import time
from vllm import LLM, SamplingParams
from vllm.spec.config import ModelConfig
from vllm.spec.spec_engine import SpecEngine 

import json 

def get_prompts(dataset: str, group_id: str):
    input_path = 'data/prompts/' + dataset + '/opt/result/group_'+ group_id + '.json'
    prompts = {}
    with open(input_path, 'r') as f:
        data = json.load(f)
        for idx, obj in enumerate(data):
            if idx > 500:
                break
            prompts[str(idx)] = obj['prompt']
    return prompts


ssm1 = 'data/opt_ssm/finance-opt-125m'
ssm2 = 'data/opt_ssm/chatbot-opt-125m'
ssm3 = 'data/opt_ssm/dialogue-opt-125m'

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=1, ignore_eos=True)
ssm1_model_config = ModelConfig(model=ssm1, gpu_memory_utilization=0.05, tensor_parallel_size=1, disable_log_stats=False, max_num_batched_tokens=32768*4, max_num_seqs=256, max_paddings=2048)
ssm2_model_config = ModelConfig(model=ssm2, gpu_memory_utilization=0.05, tensor_parallel_size=1, disable_log_stats=False, max_num_batched_tokens=32768*4, max_num_seqs=256, max_paddings=2048)
ssm3_model_config = ModelConfig(model=ssm3, gpu_memory_utilization=0.05, tensor_parallel_size=1, disable_log_stats=False, max_num_batched_tokens=32768*4, max_num_seqs=256, max_paddings=2048)
llm_model_config = ModelConfig(model="/dev/shm/opt-13b/", gpu_memory_utilization=0.7, tensor_parallel_size=2, disable_log_stats=False, max_num_batched_tokens=2048*4, max_num_seqs=64, max_paddings=256*16)
ssm_model_config = [ssm1_model_config, ssm2_model_config, ssm3_model_config]

available_gpu_num = 2
spec_engine = SpecEngine(ssm_model_config, llm_model_config, available_gpu_num, get_prompts('F_alpaca', '10'), 32, 32)

groups = ['10']
datasets = ['F_alpaca', 'Chatbot', 'Dialogue']
batches = [(32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]

for dataset in datasets:
    for group in groups:
        for (llm, ssm) in batches:
            prompts = get_prompts(dataset, group)
            print('group:', group, flush=True)
            print('dataset:', dataset, flush=True)
            print('llm_batch:', llm, flush=True)
            print('ssm_batch:', ssm, flush=True)
            spec_engine.generate(prompts, ssm_batch_size=ssm, llm_batch_size=llm)




