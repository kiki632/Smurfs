from typing import Optional


class ModelConfig():
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_num_batched_tokens: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        max_paddings: Optional[int] = None,
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        **kwargs,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.max_paddings = max_paddings
        self.quantization = quantization
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.seed = seed
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space = swap_space


