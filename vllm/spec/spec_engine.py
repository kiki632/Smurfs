from vllm.spec.config import (ModelConfig)
from typing import List, Dict, Optional
from vllm import LLM
from vllm.spec.scheduler import (Scheduler, Schedule)
import time

class Request:
    def __init__(self):
        self.time = time.time()
        self.latency = None
        self.length = None

    def finish(self, length):
        self.latency = time.time() - self.time
        self.length = length

    def get_latency(self):
        return self.latency / self.length

class SpecEngine:
    
    def __init__(
        self,
        ssm_model_config: List[ModelConfig],
        llm_model_config: ModelConfig,
        available_gpu_num: int,
        prompts: Dict[str, str],
        ssm_batch_size: int,
        llm_batch_size: int,
    ) -> None:
        self.ssm_batch_size = ssm_batch_size
        self.llm_batch_size = llm_batch_size
        self.acc_step = 0
        self.acc_cnt = 0
        self.step_time = 0
        self.verify_time = 0
        self.left_requests = len(prompts)
        self.ssm_model_config = ssm_model_config
        self.llm_model_config = llm_model_config
        self.available_gpu_num = available_gpu_num
        self.request_info: Dict[str, Request] = {}
        self.ssm_model_dict: Dict[int, LLM] = {}
        self.pending_group: Dict[str, List[int]] = {}
        self.running_group: Dict[str, List[int]] = {}
        self.request_pool: Dict[str, List[int]] = {}
        self.verified_request_set = set()
        self.request_verify_step: Dict[str, int] = {}
        self.finished_token_length: Dict[str, int] = {}
        self.voted_model_dict: Dict[str, int] = {}
        
        self.outputs: Dict[str, str] = {}
        self.time = time.time()
        self.ssm_distribution = self._get_ssm_distribution()
        model_id = 0
        for ssmconfig in ssm_model_config:
            ssm_model_id = model_id
            model_id = model_id + 1
            gpu_list = self.ssm_distribution[ssm_model_id]
            self.ssm_model_dict[ssm_model_id] = LLM(ssmconfig, ssm_model_id, gpu_list)
        llm_gpu_list = [i for i in range(self.available_gpu_num)]
        self.llm = LLM(llm_model_config, model_id, llm_gpu_list)
        self.llm_allocator = self.llm.llm_engine.scheduler.block_manager.gpu_allocator
        self.scheduler = Scheduler(self.running_group, self.pending_group, self.request_pool,
                                   self.verified_request_set, 
                                   self.llm_allocator.num_blocks * self.llm_allocator.block_size, 
                                   self.available_gpu_num,
                                   ssm_batch_size=self.ssm_batch_size, 
                                   llm_batch_size=self.llm_batch_size,
                                   fix_step=None)
        for request_id in prompts:
            prompt_token_ids = self.llm.get_tokenizer().encode(prompts[request_id])
            self.pending_group[request_id] = prompt_token_ids
            self.request_info[request_id] = Request()
        
    def _get_ssm_distribution(self) -> Dict[int, List[int]]:
        gpu_dict: Dict[int, List[int]] = {}
        for i in range(0, len(self.ssm_model_config)):
            gpu_list = [i % self.available_gpu_num]
            gpu_dict[i] = gpu_list  
        return gpu_dict
    
    def generate(
        self,
        prompts: Dict[str, str],
        ssm_batch_size: int = 32,
        llm_batch_size: int = 16,
        fix_step: Optional[int] = None
    ) -> Dict[str, str]:
        self.ssm_batch_size = ssm_batch_size
        self.llm_batch_size = llm_batch_size
        self.acc_step = 0
        self.acc_cnt = 0
        self.step_time = 0
        self.verify_time = 0
        self.left_requests = len(prompts)
        self.request_info: Dict[str, Request] = {}
        self.pending_group: Dict[str, List[int]] = {}
        self.running_group: Dict[str, List[int]] = {}
        self.request_pool: Dict[str, List[int]] = {}
        self.verified_request_set = set()
        self.request_verify_step: Dict[str, int] = {}
        self.finished_token_length: Dict[str, int] = {}
        self.voted_model_dict: Dict[str, int] = {}
        self.outputs: Dict[str, str] = {}
        self.scheduler = Scheduler(self.running_group, self.pending_group, self.request_pool,
                                   self.verified_request_set, 
                                   self.llm_allocator.num_blocks * self.llm_allocator.block_size, 
                                   self.available_gpu_num,
                                   ssm_batch_size=self.ssm_batch_size, 
                                   llm_batch_size=self.llm_batch_size,
                                   fix_step=fix_step)
        for request_id in prompts:
            prompt_token_ids = self.llm.get_tokenizer().encode(prompts[request_id])
            self.pending_group[request_id] = prompt_token_ids
            self.request_info[request_id] = Request()

        self.run()
        
        for model_id, ssm in self.ssm_model_dict.items():
            print(model_id, 'weight:', ssm.weight)
            ssm.weight = 1

    def stop(self) -> None:
        self.llm.close()
        for ssm in self.ssm_model_dict.values():
            ssm.close()

    def run(self) -> Dict[str, str]:
        is_llm_finished = True
        is_ssm_finished = True
        llm_counter = 0
        ssm_counter = 0
        outputdict: Dict[str, Dict[int, List[int]]] = {}
        last_check = 0
        ssm_schedule = None
        start_time = time.time()
        run_llm_counter = 0
        self.vote = 0
        SA_time = 0
        while True:
            
            is_llm_finished, llm_counter = self.check_llm_state(llm_counter, is_llm_finished)
            if is_llm_finished:
                llm_schedule = self.scheduler.get_LLM_schedule(self.llm_allocator.get_num_free_blocks()*self.llm_allocator.block_size,
                                                                self.llm_model_config)
                llm_request, llm_request_verify_step, is_llm_enough = self.get_llm_request(llm_schedule.batch_id_list)
                if is_llm_enough:
                    self.run_llm(llm_request, llm_request_verify_step)
                    run_llm_counter = run_llm_counter+1
                    llm_counter = 0
                    is_llm_finished = False

            # check ssm state
            if ssm_schedule is not None:
                is_ssm_finished, ssm_counter = self.check_ssm_state(ssm_schedule, ssm_counter, outputdict, is_ssm_finished)
            if is_ssm_finished:
                free_space_dict: Dict[int, int] = {}
                for ssm_id in self.ssm_model_dict:
                    ssm_allocator = self.ssm_model_dict[ssm_id].llm_engine.scheduler.block_manager.gpu_allocator
                    free_space_dict[ssm_id] = ssm_allocator.get_num_free_blocks()*ssm_allocator.block_size
                
                ssm_schedule, during = self.scheduler.get_SSM_schedule(free_space_dict, self.ssm_model_dict)
                SA_time += during

                batch_id_list = ssm_schedule.batch_id_list
                request, is_ssm_enough = self.get_ssm_request(batch_id_list)
                if is_ssm_enough:
                    #print('step:', ssm_schedule.step)
                    self.run_ssm(ssm_schedule, request)
                    self.scheduler.ssm_step_determine_counter = self.scheduler.ssm_step_determine_counter + 1
                    ssm_counter = 0
                    is_ssm_finished = False
                    outputdict: Dict[str, Dict[int, List[int]]] = {}
            
            # check system state
            if len(self.pending_group)==0 and len(self.running_group)==0 and len(self.request_pool)==0 and self.is_ssms_finished() and self.is_llm_finished():
                finish_time = time.time()
                print("finish_time:{}".format(finish_time-start_time))
                print('SA time:', SA_time)
                break
        
        latency = 0
        for req in self.request_info.values():
            latency += req.get_latency()
        print('latency:', latency / len(self.request_info))
        print('step_rate:', self.acc_step / self.acc_cnt)
        print('last_step_time:', self.step_time / self.verify_time)
        print('verify_time:', self.verify_time)
        print('vote:', self.vote)
        return self.outputs
                  
    def get_ssm_request(self, batch_id_list:List[str]) -> (Dict[str,List[int]], bool):   
        request_list: Dict[str, List[int]] = {}
        if len(self.running_group)==0 and len(self.pending_group)==0:
            return request_list, False
        if len(batch_id_list)==0:
            return request_list, False
        for request_id in batch_id_list:
            if request_id in self.running_group:
                request_list[request_id] = self.running_group[request_id]
                del self.running_group[request_id]
            else:
                assert(request_id in self.pending_group)
                request_list[request_id] = self.pending_group[request_id]
                del self.pending_group[request_id]
        return request_list, True
                   
    def get_llm_request(self, batch_id_list:List[str]) -> (Dict[str, List[int]], Dict[str, int], bool):   
        request_list: Dict[str, List[int]] = {}
        request_verify_step_list: Dict[str, int] = {}
        if len(batch_id_list) == 0:
            return request_list, request_verify_step_list, False 
        for request_id in batch_id_list:
            request_list[request_id] = self.request_pool[request_id]
            del self.request_pool[request_id]
            request_verify_step_list[request_id] = self.request_verify_step[request_id]
        return request_list, request_verify_step_list, True
            
    def run_ssm(self, schedule: Schedule, request: Dict[str, List[int]]):
        step = schedule.step
        for id in schedule.ssm_model_list:
            ssm = self.ssm_model_dict[id]
            ssm.spec_begin_generate(request, step)
        return
            
    def run_llm(self, request: Dict[str, List[int]], request_verify_step: Dict[str,int]):
        self.llm.spec_begin_verify(request, request_verify_step)
        return 
    
    def check_ssm_state(self, schedule: Schedule, counter: int, outputdict: Dict[str, Dict[int, List[int]]], is_ssm_finished: bool) -> (bool, int):
        if is_ssm_finished:
            return True, counter
        for model_id in schedule.ssm_model_list:
            ssm = self.ssm_model_dict[model_id]
            if ssm.spec_is_generation_finished()=="FINISHED":
                counter = counter + 1
                outputs, last_step_time = ssm.spec_get_generation_results()

                for request_output in outputs:
                    if request_output.request_id not in outputdict.keys():
                        outputdict[request_output.request_id] = {}
                    outputdict[request_output.request_id][model_id] = request_output.prompt_token_ids + request_output.outputs[0].token_ids
                if counter == len(schedule.ssm_model_list):
                    real_output = self.majority_vote(outputdict)
                    for request_id in real_output.keys():
                        self.request_pool[request_id] = real_output[request_id]
                        self.request_verify_step[request_id] = schedule.step
                    return True, counter
        return False, counter
    
    def check_llm_state(self, counter: int, is_llm_finished: bool) -> (bool, int):
        if is_llm_finished:
            return True, counter
        if self.llm.spec_is_generation_finished()=="FINISHED":
            accepted_rate_dict: Dict[int, List[float]] = {}
            outputs, last_step_time_list = self.llm.spec_get_generation_results()
            last_step_time = last_step_time_list[-1]
            counter = 1
            accepted_length_list: List[int] = []
            accepts = 0
            steps = 0
            for output in outputs:
                accepted_length = output.get_accepted_token_length()
                if output.request_id not in self.verified_request_set:
                    self.verified_request_set.add(output.request_id)
                    accepted_length = accepted_length + 1
                voted_model = self.voted_model_dict[output.request_id]
                accepted_length_list.append(accepted_length)
                rate = accepted_length/self.request_verify_step[output.request_id]
                accepts += accepted_length
                steps += self.request_verify_step[output.request_id]
                self.acc_step += accepted_length
                self.acc_cnt += self.request_verify_step[output.request_id]
                if voted_model in accepted_rate_dict:
                    accepted_rate_list = accepted_rate_dict[voted_model]
                    accepted_rate_list.append(rate)
                else:
                    accepted_rate_list = [rate]
                    accepted_rate_dict[voted_model] = accepted_rate_list
                if output.request_id in self.finished_token_length:
                    self.finished_token_length[output.request_id] = self.finished_token_length[output.request_id]+ output.get_accepted_token_length()
                else:
                    self.finished_token_length[output.request_id] = output.get_accepted_token_length()
                if output.finished:
                    self.request_info[output.request_id].finish(len(output.outputs[0].token_ids))
                    self.outputs[output.request_id] = output.outputs[0].text
                    self.left_requests -= 1
                else:
                    self.running_group[output.request_id] = output.prompt_token_ids + output.outputs[0].token_ids

            self.step_time += last_step_time
            self.verify_time += 1
            self.update_ssm_weights(accepted_rate_dict)
            self.handle_monitor_data(accepted_length_list, last_step_time)
            return True, counter
        return False, counter
    
    def handle_monitor_data(self, accepted_token_length: List[int], last_step_time: float):
        average_accepted_length = sum(accepted_token_length)/len(accepted_token_length)
        # print(last_step_time)
        self.scheduler.add_monitor_data(average_accepted_length, last_step_time)
    
    def update_ssm_weights(self, accepted_rate_dict: Dict[int, List[float]]):
        reward_threshold = 0.8
        punish_threshold = 0.6
        #reward_threshold = 0.7
        #punish_threshold = 0.3
        reward_factor = 0.18
        punish_factor = 0.18
        for voted_model_id in accepted_rate_dict:
            accepted_rate_list = accepted_rate_dict[voted_model_id]
            average_rate = sum(accepted_rate_list)/len(accepted_rate_list)
            ssm = self.ssm_model_dict[voted_model_id]
            if average_rate < punish_threshold and ssm.weight > 0.2:
                ssm.weight -= punish_factor
            if average_rate > reward_threshold and ssm.weight < 1.8:
                ssm.weight += reward_factor
        return  

    def majority_vote(self, outputdict: Dict[str, Dict[int, List[int]]]) -> Dict[str, List[int]]:
        begin = time.time()
        token_dict: Dict[str, Dict[int, int]] = {}
        result_dict: Dict[str, List[int]] = {}
        model_ids = list(list(outputdict.values())[0].keys())
        model_weights = {model_id: self.ssm_model_dict[model_id].weight for model_id in model_ids}
        max_weight = 0
        sum_weight = 0
        max_model_id = -1

        for model_id, weight in model_weights.items():
            if weight > max_weight:
                max_weight = weight
                max_model_id = model_id
            sum_weight += weight

        if sum_weight - max_weight < max_weight:
            for request_id, request_dict in outputdict.items():
                self.voted_model_dict[request_id] = max_model_id
                result_dict[request_id] = request_dict[max_model_id]
            return result_dict

        for request_id, request_dict in outputdict.items():
            step = len(list(request_dict.values())[0])
            for idx in range(step):
                token_dict: Dict[int, float] = {}
                for model_id, token_ids in request_dict.items():
                    token = token_ids[idx]
                    if token not in token_dict.keys():
                        token_dict[token] = 0
                    else:
                        token_dict[token] += model_weights[model_id]

                next_token = max(token_dict, key=token_dict.get)
                remove_model = []

                for model_id, token_ids in request_dict.items():
                    token = token_ids[idx]
                    if token != next_token:
                        remove_model.append(model_id)
                    else:
                        self.voted_model_dict[request_id] = model_id

                for model_id in remove_model:
                    del request_dict[model_id]

                if request_id not in result_dict.keys():
                    result_dict[request_id] = []
                result_dict[request_id].append(next_token)
        self.vote += time.time() - begin
        return result_dict

    
    def is_ssms_finished(self) -> bool:
        finished = True
        for ssm_id in self.ssm_model_dict:
            ssm = self.ssm_model_dict[ssm_id]
            finished = finished and (ssm.spec_is_generation_finished()=="EMPTY")
        return finished
        
    def is_llm_finished(self) -> bool:
        if self.llm.spec_is_generation_finished() == "EMPTY":
            return True
        return False