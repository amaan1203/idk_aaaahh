"""
src/inference/vllm_generator.py — vLLM Rollout Generator
=========================================================
Drop-in replacement for HuggingFace generate() for batch trading signal generation.
Exploits PagedAttention for high-throughput parallel rollout generation.

Corresponds to: Section 4 (Systems contribution) of the paper draft.

Inputs: list of prompt strings, model name
Outputs: list[list[str]] — n completions per prompt

Launches vLLM server in a background tmux session for persistence.
"""

import subprocess
import time
import requests
from typing import List
from openai import OpenAI


class VLLMGenerator:
    """
    Wraps vLLM's OpenAI-compatible API for batch rollout generation.

    Parameters
    ----------
    model_name : HuggingFace model identifier
    port       : port to run vLLM server on
    gpu_memory_utilization : fraction of GPU memory vLLM may use
    max_model_len : max context length
    dtype : torch dtype (bfloat16 recommended for A10G)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        port: int = 8000,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 2048,
        dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.base_url = f"http://localhost:{port}/v1"
        self._client = None

    def start_server(self, timeout: int = 180) -> None:
        """
        Launch vLLM server in a background tmux session named 'vllm'.
        Blocks until server is healthy or timeout is exceeded.
        """
        cmd = (
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {self.model_name} "
            f"--dtype {self.dtype} "
            f"--max-model-len {self.max_model_len} "
            f"--gpu-memory-utilization {self.gpu_memory_utilization} "
            f"--port {self.port}"
        )
        subprocess.run("tmux kill-session -t vllm 2>/dev/null", shell=True)
        subprocess.run(f"tmux new-session -d -s vllm '{cmd}'", shell=True, check=True)
        print(f"vLLM server starting on port {self.port}... ", end="", flush=True)

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"http://localhost:{self.port}/health", timeout=2)
                if r.status_code == 200:
                    print("ready.")
                    break
            except Exception:
                pass
            time.sleep(3)
        else:
            raise TimeoutError(f"vLLM server did not start within {timeout}s")

        self._client = OpenAI(base_url=self.base_url, api_key="token-unused")

    def stop_server(self) -> None:
        """Kill the vLLM tmux session."""
        subprocess.run("tmux kill-session -t vllm 2>/dev/null", shell=True)
        self._client = None
        print("vLLM server stopped.")

    def is_running(self) -> bool:
        try:
            r = requests.get(f"http://localhost:{self.port}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompts: List[str],
        n: int = 1,
        max_tokens: int = 64,
        temperature: float = 0.8,
    ) -> List[List[str]]:
        """Generate `n` completions for each prompt sequentially."""
        if self._client is None:
            raise RuntimeError("Call start_server() before generate()")

        results = []
        for prompt in prompts:
            response = self._client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                n=n,
                temperature=temperature,
            )
            completions = [choice.text for choice in response.choices]
            results.append(completions)
        return results

    def generate_batch(
        self,
        prompts: List[str],
        n: int = 1,
        max_tokens: int = 64,
        temperature: float = 0.8,
    ) -> List[List[str]]:
        """
        Send all prompts in a single batched request.
        Use this for benchmarking — exploits PagedAttention properly.
        """
        if self._client is None:
            raise RuntimeError("Call start_server() before generate_batch()")

        response = self._client.completions.create(
            model=self.model_name,
            prompt=prompts,
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
        )
        results = [[] for _ in prompts]
        for i, choice in enumerate(response.choices):
            results[i % len(prompts)].append(choice.text)
        return results
