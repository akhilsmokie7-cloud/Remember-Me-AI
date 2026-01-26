import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

class ModelRegistry:
    """
    Manages local AI models.
    Focuses on lightweight, high-performance open weights models suitable for consumer hardware.
    """

    MODELS = {
        "tiny": {
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "name": "Qwen 2.5 (0.5B) - Tiny",
            "description": "Ultra-fast, low memory (Run anywhere)"
        },
        "small": {
            "id": "Qwen/Qwen2.5-1.5B-Instruct",
            "name": "Qwen 2.5 (1.5B) - Small",
            "description": "Balanced speed/intelligence (Recommended)"
        },
        "medium": {
            "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "name": "SmolLM2 (1.7B) - Medium",
            "description": "High reasoning capability"
        }
    }

    def __init__(self):
        self.current_model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = None

    def list_models(self):
        return self.MODELS

    def load_model(self, key: str):
        """
        Downloads and loads the specified model key.
        """
        if key not in self.MODELS:
            raise ValueError(f"Unknown model key: {key}")

        info = self.MODELS[key]
        print(f"Loading {info['name']} on {self.device}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(info["id"])
            self.current_model = AutoModelForCausalLM.from_pretrained(
                info["id"],
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            if self.device == "cpu":
                self.current_model.to("cpu")

            self.model_id = key
            print(f"✓ {info['name']} loaded successfully.")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

    def generate_response(self, user_input: str, context_str: str, system_prompt: str = None) -> str:
        """
        Generates a response using the loaded model, injecting context.
        """
        if not self.current_model:
            return "Error: No model loaded. Use /model to select one."

        # Construct Prompt with SELF-AWARENESS Injection
        default_system = (
            "You are a helpful AI assistant equipped with the Remember Me Cognitive Kernel. "
            "You have long-term memory via CSNP, and access to tools like Image Generation and Web Search. "
            "Do not deny these capabilities. If the user refers to past conversations, assume your memory context is accurate. "
            "Answer directly and helpfully."
        )
        sys_p = system_prompt if system_prompt else default_system

        # Combine context with user input
        full_context = ""
        if context_str:
            full_context = f"\n[RELEVANT LONG-TERM MEMORY]:\n{context_str}\n"

        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": f"{full_context}\nUSER: {user_input}"}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.current_model.generate(
                inputs.input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        # Decode
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def generate_stream(self, user_input: str, context_str: str):
        """
        Generator for streaming response.
        """
        if not self.current_model:
            yield "Error: No model loaded."
            return

        full_context = ""
        if context_str:
            full_context = f"\n[RELEVANT LONG-TERM MEMORY]:\n{context_str}\n"

        # SELF-AWARENESS Injection for Streaming too
        system_prompt = (
            "You are a helpful AI assistant equipped with the Remember Me Cognitive Kernel. "
            "You have long-term memory via CSNP. Use the provided memory context to answer questions about the past."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{full_context}\nUSER: {user_input}"}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7
        )

        thread = Thread(target=self.current_model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
