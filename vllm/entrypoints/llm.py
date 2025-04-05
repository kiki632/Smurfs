from functools import partial
from typing import List, Optional, Union, Dict
import time

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine, EngineResult
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter
from vllm.spec.config import ModelConfig

class LLMResult:
    def __init__(self, future_result, remaining_steps, get_next_future_fn):
        self.remaining_steps = remaining_steps
        self.future_result = future_result
        self.get_next_future_fn = get_next_future_fn
        self.time_list = []

    def get(self):
        while not self.available():
            continue
        res, step_time = self.future_result.get()
        self.time_list.append(step_time)
        #print(self.time_list)
        return res, self.time_list

    def available(self):
        if self.remaining_steps > 0:
            if self.future_result.available():
                intermediate_output = self.future_result.get()
                self.time_list.append(intermediate_output[1])
                self.future_result = self.get_next_future_fn(intermediate_output)
                self.remaining_steps -= 1
            return False
        else:
            return self.future_result.available()

class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    NOTE: This class is intended to be used for offline inference. For online
    serving, use the `AsyncLLMEngine` class instead.
    NOTE: For the comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq". If None, we assume the model weights are not
            quantized and use `dtype` to determine the data type of the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
    """

    def __init__(
            self, 
            config: ModelConfig,
            model_id: int,
            gpu_list: List[int],
    ) -> None:
        engine_args = EngineArgs(
            model=config.model,
            tokenizer=config.tokenizer,
            tokenizer_mode=config.tokenizer_mode,
            trust_remote_code=config.trust_remote_code,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=config.dtype,
            quantization=config.quantization,
            revision=config.revision,
            tokenizer_revision=config.tokenizer_revision,
            seed=config.seed,
            gpu_memory_utilization=config.gpu_memory_utilization,
            swap_space=config.swap_space,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_seqs=config.max_num_seqs,
            max_paddings=config.max_paddings,
        )
        self.model_id = model_id
        self.port = model_id + 10000
        self.gpu_list = gpu_list
        self.llm_engine = LLMEngine.from_engine_args(engine_args, self.port, self.gpu_list)
        self.request_counter = Counter()

        self.status = "EMPTY"
        self.last_result = None
        self.weight = 1

    def update_weight(self, factor):
        self.weight = self.weight * factor
        return
    
    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer = tokenizer

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: A list of prompts to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
            prompt_token_ids: A list of token IDs for the prompts. If None, we
                use the tokenizer to convert the prompts to token IDs.
            use_tqdm: Whether to use tqdm to display the progress bar.

        Returns:
            A list of `RequestOutput` objects containing the generated
            completions in the same order as the input prompts.
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if (prompts is not None and prompt_token_ids is not None
                and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                             "must be the same.")
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        # Add requests to the engine.
        num_requests = len(prompts) if prompts is not None else len(
            prompt_token_ids)
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[
                i]
            self._add_request(prompt, sampling_params, token_ids)
        return self._run_engine(use_tqdm)

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id, prompt, sampling_params,
                                    prompt_token_ids)

    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts")
        # Run the engine.
        outputs: List[RequestOutput] = []
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs

    def _spec_generate_requests_async(self, request_ids, preempt_taboo_ids=None) -> List[EngineResult]:
        # Run the engine.
        step_outputs = self.llm_engine.spec_generate_requests_async(request_ids, preempt_taboo_ids=preempt_taboo_ids)
        return step_outputs

    def _spec_verify_requests_async(self, request_ids) -> List[EngineResult]:
        # Run the engine.
        step_outputs = self.llm_engine.spec_verify_requests_async(request_ids)
        return step_outputs

    def close(self):
        self.llm_engine.close()
    
    def spec_abort_request(self, spec_request_id):
        assert self.status == "EMPTY"
        self.llm_engine.abort_request(spec_request_id)

    def spec_begin_generate_prompt_input(
        self,
        steps,
        prompts: Optional[Dict[str, List[int]]] = None,
        prompt_token_ids: Optional[Dict[str, List[int]]] = None,
    ) -> List[RequestOutput]:
        assert self.status == "EMPTY"

        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                                "provided.")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if (prompts is not None and prompt_token_ids is not None
                and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                                "must be the same.")
        
        # SamplingParam for SSM spec is fixed
        sampling_params = SamplingParams(temperature=0, top_p=1, ignore_eos=True, max_tokens=steps)

        # Add requests to the engine.
        num_requests = len(prompts) if prompts is not None else len(prompt_token_ids)
        request_ids = []
        for i in range(num_requests):
            request_id = list(prompts.keys())[i] if prompts is not None else list(prompt_token_ids.keys())[i]
            prompt = prompts[request_id] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[request_id]
            self.llm_engine.add_request(request_id, prompt, sampling_params, token_ids)
            request_ids.append(request_id)

        outputs = self._spec_generate_requests_async(request_ids)
        self.last_result = LLMResult(outputs, steps - 1, lambda x: self._spec_generate_requests_async(request_ids))
        self.status = "INPROGRESS"

    def spec_begin_generate(
        self,
        prompt_token_ids: Dict[str, List[int]],
        steps: int
    ) -> List[RequestOutput]:
        assert self.status == "EMPTY"

        if prompt_token_ids is None:
            raise ValueError("Prompt_token_ids must be provided.")
        
        # SamplingParam for SSM spec is fixed
        sampling_params = SamplingParams(temperature=0, top_p=1, ignore_eos=True, max_tokens=steps)

        # Add requests to the engine.
        num_requests = len(prompt_token_ids)
        request_ids = []
        for i in range(num_requests):
            request_id = list(prompt_token_ids.keys())[i]
            prompt = None
            token_ids = prompt_token_ids[request_id]
            self.llm_engine.add_request(request_id, prompt, sampling_params, token_ids)
            request_ids.append(request_id)

        outputs = self._spec_generate_requests_async(request_ids)
        self.last_result = LLMResult(outputs, steps - 1, lambda x: self._spec_generate_requests_async(request_ids))
        self.status = "INPROGRESS"
    
    def spec_is_generation_finished(self):
        # assert self.status == "INPROGRESS"
        if self.status == "EMPTY":
            # EMPTY, so we cannot get result from model now
            return "EMPTY"
        
        if self.last_result.available():
            return "FINISHED"
        return "INPROGRESS"

    def spec_get_generation_results(self):
        assert self.status == "INPROGRESS"
        # print("\nTrying to get result\n")

        result = self.last_result.get()
        self.last_result = None
        self.status = "EMPTY"
        return result

    def _spec_do_verify(
        self,
        requests: Dict[str, List[int]],
        requests_verify_step: Dict[str, int]
    ):
        # print("request: ", requests_verify_step)
        request_ids = []
        for request_id, token_ids in requests.items():
            seq_group = self.llm_engine.verify_get_seq_group(request_id)
            assert seq_group is not None
            if requests_verify_step[request_id] == 0:
                spec_token_ids = []
            else:
                spec_token_ids = token_ids[-requests_verify_step[request_id]:]
            # set spec id
            seq_group.add_spec_tokens(spec_token_ids)
            request_ids.append(request_id)
        outputs = self.llm_engine.spec_generate_requests_async(request_ids)
        return outputs

    def spec_begin_verify(
        self,
        requests: Dict[str, List[int]],
        requests_verify_step: Dict[str, int]
    ) -> List[RequestOutput]:
        assert self.status == "EMPTY"

        # SamplingParam for LLM spec is given in model, fixed for development
        # todo: pass in from model
        sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=128)

        # 1. Create SeqGroup for those not started or have to be recomputed
        num_requests = len(requests)
        new_requests = []
        taboo_list = []
        for request_id, token_ids in requests.items():
            seq_group = self.llm_engine.verify_get_seq_group(request_id)
            if seq_group is None:
                token_ids = token_ids[:-requests_verify_step[request_id]]
                # print(token_ids)
                self.llm_engine.add_request(request_id, None, sampling_params, token_ids)
                new_requests.append(request_id)
            elif seq_group in self.llm_engine.scheduler.waiting:
                new_requests.append(request_id)
            else:
                taboo_list.append(request_id)

        # 2. start these new requests for only one step
        if len(new_requests) != 0:
            # 3.1 define what to do after all new requests are prefilled
            def run_spec_verify(requests, requests_verify_step, prefill_results):
                prefill_results, prefill_time = prefill_results
                # a. use prefill_result to replace request token and verify step
                for prefill_result in prefill_results:
                    request_id = prefill_result.request_id
                    prev_spec_step = requests_verify_step[request_id]
                    # change 1st spec token to ground truth
                    requests[request_id][-prev_spec_step] = prefill_result.outputs[0].token_ids[-1]
                    # reduce spec length by 1
                    requests_verify_step[request_id] = prev_spec_step - 1
                # b. start verify process
                return self._spec_do_verify(requests, requests_verify_step)
            outputs = self._spec_generate_requests_async(new_requests, taboo_list)
            # 3.2 bind previous func to prefill output, so after prefill an verification is automatically started
            result = LLMResult(outputs, 1, lambda x: run_spec_verify(requests, requests_verify_step, x))
        else:
            # 3.2 start verify_process
            outputs = self._spec_do_verify(requests, requests_verify_step)
            # just a verify process, nothing new
            result = LLMResult(outputs, 0, lambda x: None)

        self.last_result = result
        self.status = "INPROGRESS"

