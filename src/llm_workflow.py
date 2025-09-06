import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig


class LlmWorkflow:
    """
    LLM wrapper for text generation using HuggingFace Transformers.
    Provides quantized model loading and prompt-based response generation.
    """

    def __init__(self, default_model=True, model_name="", sys_prompt=""):
        """
        Initialize workflow with model and tokenizer.

        :param default_model: use default model designed for this workflow
        :param model_name: HuggingFace model name or local path
        :param sys_prompt: system prompt, required for some models
        """
        if default_model:
            self.sys_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            self.model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
        else:
            self.sys_prompt = sys_prompt
            self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.llm = self._load_model()

    def _load_model(self):
        """
        Load quantized LLM into memory with 4-bit precision.

        :return: Model instance
        """
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def generate_response(self, prompt: str, max_tokens: int, temperature: float = 0.7) -> str:
        """
        Generate response from the model for a given prompt.

        :param prompt: User input text
        :param max_tokens: Maximum number of tokens to generate
        :param temperature: Sampling temperature
        :return: Generated response as plain text
        """
        generation_config = GenerationConfig.from_pretrained(
            self.model_name,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": prompt},
        ]

        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([chat_prompt], return_tensors="pt").to(self.llm.device)
        output_ids = self.llm.generate(**inputs, generation_config=generation_config)[0]

        generated_ids = output_ids[len(inputs["input_ids"][0]):]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
