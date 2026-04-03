import re
import time
from typing import Tuple, Dict
from llama_cpp import Llama
from llama_cpp.llama_types import (
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
)


class LlmWorkflow:
    """
    A workflow class for managing Large Language Model inference using llama.cpp.

    This class handles model initialization, prompt execution, performance
    metrics tracking, and specialized output parsing (e.g., separating
    reasoning/thinking blocks from the final answer).
    """

    def __init__(self):
        """
        Initializes the LlmWorkflow with a specific GGUF model and hardware-optimized settings.

        Configures the context window, GPU acceleration layers, and thread count
        for optimal performance on RTX-series hardware.
        """
        self.model_path = "./models/Qwen3.5-35B-A3B-Q5_K_M.gguf"
        self.n_ctx = 32768
        self.max_tokens = int(self.n_ctx / 2)
        self.temperature = 0.1

        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=15,
            n_ctx=self.n_ctx,
            n_threads=16,
            verbose=False,
            flash_attn=True,
        )

    @staticmethod
    def _metrics(output: Dict, start_time: float, end_time: float) -> None:
        """
        Calculates and prints performance metrics for the inference task

        Args:
            output: The raw dictionary response from the llama.cpp completion.
            start_time: Epoch timestamp when the request started.
            end_time: Epoch timestamp when the request finished.
        """
        usage = output['usage']
        prompt_tokens = usage['prompt_tokens']
        completion_tokens = usage['completion_tokens']
        total_time = end_time - start_time
        tps = completion_tokens / total_time

        print(f"\n--- Metrics ---")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Tokens: {completion_tokens}")
        print(f"Time: {total_time:.2f}s")
        print(f"Speed: {tps:.2f} tokens/sec")
        print(f"\n--- End Metrics ---")

    @staticmethod
    def _split_thinking(text: str) -> Tuple[str, str]:
        """
        Extracts internal reasoning blocks from the model's output.

        Models like Qwen and Gemma often use <think> tags for chain-of-thought.
        This method separates those thoughts from the actual final response.

        Args:
            text: The raw string content received from the LLM.

        Returns:
            A tuple containing (thinking_content, cleaned_response_text).
        """
        end_tag = "</think>"
        pos = text.find(end_tag)
        if pos != -1:
            thinking = text[:pos].strip()
            clean = text[pos + len(end_tag):].strip()

            print(f"\n--- Thinking ---")
            print(thinking)
            print(f"\n--- End Thinking ---")

            return thinking, clean
        else:
            return text, text

    def generate_response(self, prompt: str) -> str:
        """
        Processes a user prompt and returns a clean AI response.

        Handles message formatting, calls the LLM inference engine,
        triggers metrics logging, and post-processes the text to remove
        internal reasoning blocks.

        Args:
            prompt: The user's input string or code snippet for review.

        Returns:
            The final processed response string without thinking tags.

        Raises:
            ValueError: If the model returns an unexpected stream or iterator instead of a dict.
        """
        start_time = time.time()

        messages = [
            ChatCompletionRequestSystemMessage(
                role="system",
                content="You are a senior developer. Focus your thinking ONLY on complex logic. "
                        "For simple style issues, skip the reasoning and just provide the comment."
            ),
            ChatCompletionRequestUserMessage(role="user", content=prompt),
        ]

        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            repeat_penalty=1.1,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        if not isinstance(output, dict):
            raise ValueError("Expected dict response, but got a stream/iterator")

        self._metrics(output, start_time, time.time())

        full_content = output['choices'][0]['message']['content']
        _, clean_text = self._split_thinking(full_content)

        return clean_text
