from typing import Optional
from typing import List, Dict, Set
import sys
import numpy as np
from vllm.spec.config import ModelConfig
from vllm import LLM
import random
import math
import time

class Schedule():
    def __init__(
        self,
        batch_size: Optional[int] = None,
        batch_id_list: Optional[List[str]] = None,
        step: Optional[int] = None,
        ssm_model_list: Optional[List[str]] = None,
    ) -> None:
        if batch_size is not None:
            self.batch_size = batch_size
        if batch_id_list is not None:
            self.batch_id_list = batch_id_list
        if step is not None:
            self.step = step
        if ssm_model_list is not None:
            self.ssm_model_list = ssm_model_list


class Scheduler():
    def __init__(
        self,
        running_group: Dict[str, List[int]],
        pending_group: Dict[str, List[int]],
        request_pool: Dict[str, List[int]],
        verified_request_set: Set[int],
        llm_total_space: int,
        available_gpu_num: int,
        ssm_batch_size: int,
        llm_batch_size: int,
        fix_step: int,
        ssm_step_punish_factor: int = 1,
        ssm_step_reward_factor: int = 1,
        ssm_step_determine_granularity: int = 20,
        ssm_step_average_granularity: int = 5
    ) -> None:
        self.running_group = running_group
        self.pending_group = pending_group
        self.request_pool = request_pool
        self.verified_request_set = verified_request_set
        self.llm_batch_size = llm_batch_size
        self.fix_step = fix_step
        self.llm_total_space = llm_total_space
        self.ssm_accept_step_monitor_list: List[float] = []
        self.ssm_step_time_monitor_list: List[float] = []
        self.ssm_step = 8
        self.available_gpu_num = available_gpu_num
        self.ssm_step_determine_counter = 0

        self.ssm_step_punish_factor = ssm_step_punish_factor
        self.ssm_step_reward_factor = ssm_step_reward_factor
        self.ssm_step_determine_granularity = ssm_step_determine_granularity
        self.ssm_step_average_granularity = ssm_step_average_granularity
        self.ssm_batch_size_thres = ssm_batch_size
        self.reset_temperature()
        self.ssm_last_step = None

    def add_monitor_data(self, accept_step: float, step_time: float):
        self.ssm_accept_step_monitor_list.append(accept_step)
        self.ssm_step_time_monitor_list.append(step_time)
        return

    def _get_sorted_ids(self, keys: List[str]) -> List[str]:
        int_keys = [int(key) for key in keys]
        int_keys.sort()
        sorted_keys = [str(key) for key in int_keys]
        return sorted_keys

    def reset_temperature(self):
        self.step_values: Dict[int, int] = {}
        self.temperature = 4
        self.temp_counter = 0

    def get_next_step(self):
        if self.ssm_last_step != None:
            delta_value = self.step_values[self.ssm_last_step] - self.step_values[self.ssm_step]
            if delta_value < 0 and math.exp(1e4 * delta_value / self.temperature) < random.uniform(0, 1):
                self.ssm_step = self.ssm_last_step

        self.temp_counter += 1
        if self.temp_counter % 2 == 0 and self.temperature > 0:
            self.temperature -= 1
        self.ssm_last_step = self.ssm_step
        self.ssm_step += random.randint(-self.temperature, self.temperature)


    def get_SSM_schedule(self, free_space_dict: Dict[int, int], ssm_model_dict: Dict[int, LLM]) -> Schedule:
        batch_size_threshold = self.ssm_batch_size_thres
        request_pool_thres = batch_size_threshold
        during = 0
        if self.ssm_step_determine_counter >= self.ssm_step_determine_granularity:
            avg_accept_step = sum(self.ssm_accept_step_monitor_list)/len(self.ssm_accept_step_monitor_list)
            avg_step_time = sum(self.ssm_step_time_monitor_list)/len(self.ssm_step_time_monitor_list)
            self.step_values[self.ssm_step] = avg_step_time / avg_accept_step
            
            self.get_next_step()
            
            self.ssm_accept_step_monitor_list = []
            self.ssm_step_time_monitor_list = []
            self.ssm_step_determine_counter = 0

        free_space = sys.maxsize
        for ssm_id in free_space_dict:
            if free_space_dict[ssm_id] < free_space:
                free_space = free_space_dict[ssm_id]
        model_weights = {ssm.model_id: ssm.weight for ssm in ssm_model_dict.values()}
        ssm_model_list = sorted(model_weights, key=model_weights.get, reverse=True)[:self.available_gpu_num]
        
        if self.fix_step is not None:
            self.ssm_step = self.fix_step
        batch_id_list = []
        if len(self.request_pool) >= request_pool_thres:
            return Schedule(0, batch_id_list, self.ssm_step, ssm_model_list), during
        request_ids = list(self.running_group.keys())
        if len(request_ids) > 0:
            request_ids = [int(request_id) for request_id in request_ids]
            request_ids.sort()
            request_ids = [str(request_id) for request_id in request_ids]
            for request_id in request_ids:
                if free_space >= len(self.running_group[request_id]) + self.ssm_step:
                    free_space -= len(self.running_group[request_id]) + self.ssm_step
                    batch_id_list.append(request_id)
                    if len(batch_id_list) >= batch_size_threshold:
                        return Schedule(0, batch_id_list=batch_id_list, step=self.ssm_step, ssm_model_list=ssm_model_list), during
        
        request_ids = list(self.pending_group.keys())
        if len(request_ids) > 0:
            request_ids = [int(request_id) for request_id in request_ids]
            request_ids.sort()
            request_ids = [str(request_id) for request_id in request_ids]
            for request_id in request_ids:
                if free_space >= len(self.pending_group[request_id]) + self.ssm_step:
                    free_space -= len(self.pending_group[request_id]) + self.ssm_step
                    batch_id_list.append(request_id)
                    if len(batch_id_list) >= batch_size_threshold:
                        return Schedule(0, batch_id_list=batch_id_list, step=self.ssm_step, ssm_model_list=ssm_model_list), during

        return Schedule(0, batch_id_list, step=self.ssm_step, ssm_model_list=ssm_model_list), during
   
    def get_LLM_schedule(self, free_space: int, model_config: ModelConfig) -> Schedule:
        threshold = 0.9
        seq_lens = []
        batch_id_list = []

        if len(self.request_pool) == 0:
            return Schedule(0, batch_id_list, None, None)

        request_ids = self._get_sorted_ids(self.request_pool)
        for request_id in request_ids:
            seq_len = len(self.request_pool[request_id]) if request_id not in self.verified_request_set else 1
            free_space += seq_len 
            new_seq_lens = seq_lens + [seq_len]
            if len(new_seq_lens) > self.llm_batch_size:
                continue
            num_batched_tokens = len(new_seq_lens) * max(new_seq_lens)
            if num_batched_tokens > model_config.max_num_batched_tokens:
                continue
            num_padding = num_batched_tokens - sum(new_seq_lens)
            if num_padding > model_config.max_paddings:
                continue
            seq_lens = new_seq_lens
            batch_id_list.append(request_id)

        return Schedule(len(seq_lens), batch_id_list, None, None)