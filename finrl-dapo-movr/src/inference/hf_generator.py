"""
src/inference/hf_generator.py — HuggingFace generate() Wrapper
==============================================================
Benchmark baseline for vLLM throughput comparison.
API is identical to VLLMGenerator so the benchmark script (05_vllm_benchmark.py)
can swap them freely.

Inputs: list of prompt strings, model name
Outputs: list[list[str]] — n completions per prompt
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List


class HFGenerator:
    """
    HuggingFace transformers generate() wrapper.

    Parameters
    ----------
    model_name : HuggingFace model identifier
    device     : 'cuda' or 'cpu'
    dtype      : torch dtype string ('bfloat16', 'float16', 'float32')
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                            "float32": torch.float32}[dtype]
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load tokenizer and model into GPU memory."""
        print(f"Loading {self.model_name} with HuggingFace... ", end="", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
        )
        self.model.eval()
        print("loaded.")

    def unload(self) -> None:
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        self.model = None
        self.tokenizer = None

    def generate(
        self,
        prompts: List[str],
        n: int = 1,
        max_tokens: int = 64,
        temperature: float = 0.8,
    ) -> List[List[str]]:
        """Generate `n` completions per prompt. Same API as VLLMGenerator."""
        if self.model is None:
            raise RuntimeError("Call load() before generate()")

        results = []
        for prompt in prompts:
            inputs = self.tokenizer(
                [prompt] * n,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            completions = [
                self.tokenizer.decode(ids[input_len:], skip_special_tokens=True)
                for ids in output_ids
            ]
            results.append(completions)
        return results

    def generate_batch(
        self,
        prompts: List[str],
        n: int = 1,
        max_tokens: int = 64,
        temperature: float = 0.8,
    ) -> List[List[str]]:
        """Batched generation — all prompts in one forward pass."""
        if self.model is None:
            raise RuntimeError("Call load() before generate_batch()")

        all_prompts = prompts * n
        inputs = self.tokenizer(
            all_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        all_completions = [
            self.tokenizer.decode(ids[input_len:], skip_special_tokens=True)
            for ids in output_ids
        ]
        results = []
        for i in range(len(prompts)):
            results.append(all_completions[i::len(prompts)])
        return results
