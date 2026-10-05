from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LLMResponse:
    text: str


class LLMClient:
    """
    Local Llama wrapper with the same interface as agentpoisoning.llm.LLMClient.
    Pass model=/home/ecs-user/Code/llama7b (or any HF-compatible local path).
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: float = 1.0,
        api_key_list: Optional[List[str]] = None,
        logprob_model: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        self.model_path = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.api_key_list = api_key_list or []
        self.logprob_model = logprob_model

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        if dtype:
            torch_dtype = getattr(torch, dtype, None)
        else:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        return input_ids, attention_mask

    def _format_prompt(self, prompt: str) -> str:
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generate(self, prompt: str, retries: int = 3) -> LLMResponse:
        prompt_text = self._format_prompt(prompt)
        input_ids, attention_mask = self._encode(prompt_text)
        max_new_tokens = max(1, int(self.max_tokens))
        do_sample = self.temperature > 0
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = input_ids.shape[1]
        gen_ids = outputs[0][prompt_len:]
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return LLMResponse(text=decoded.strip())

    def generate_with_logprobs(
        self,
        prompt: str,
        retries: int = 3,
        top_logprobs: int = 0,
    ) -> Tuple[str, List[Tuple[str, Optional[float]]]]:
        prompt_text = self._format_prompt(prompt)
        input_ids, attention_mask = self._encode(prompt_text)
        max_new_tokens = max(1, int(self.max_tokens))
        do_sample = self.temperature > 0
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences = outputs.sequences[0]
        prompt_len = input_ids.shape[1]
        gen_ids = sequences[prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        tokens: List[Tuple[str, Optional[float]]] = []
        scores = outputs.scores or []
        for step, token_id in enumerate(gen_ids):
            if step >= len(scores):
                tokens.append((self.tokenizer.convert_ids_to_tokens(int(token_id)), None))
                continue
            logits = scores[step][0]
            log_probs = torch.log_softmax(logits, dim=-1)
            lp = float(log_probs[int(token_id)])
            token_str = self.tokenizer.convert_ids_to_tokens(int(token_id))
            tokens.append((token_str, lp))

        return text, tokens

    def embedding(self, text: str, model: Optional[str] = None, retries: int = 3) -> Optional[List[float]]:
        input_ids, attention_mask = self._encode(text)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = outputs.hidden_states[-1][0]
        mask = attention_mask[0].unsqueeze(-1).float()
        denom = mask.sum().clamp(min=1.0)
        pooled = (hidden * mask).sum(dim=0) / denom
        return pooled.detach().cpu().tolist()

    def sequence_logprob(self, prompt: str, completion: str, retries: int = 3) -> Optional[float]:
        tokens = self.completion_token_logprobs(prompt, completion, retries=retries)
        if not tokens:
            return None
        total = 0.0
        for _, lp in tokens:
            if lp is not None:
                total += lp
        return total

    def completion_token_logprobs(
        self, prompt: str, completion: str, retries: int = 3
    ) -> Optional[List[Tuple[str, Optional[float]]]]:
        prompt_text = self._format_prompt(prompt)
        full_prompt = f"{prompt_text}{completion}"
        input_ids, attention_mask = self._encode(full_prompt)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        logits = outputs.logits[0]
        all_ids = input_ids[0]

        prompt_ids, _ = self._encode(prompt_text)
        cutoff = prompt_ids.shape[1]
        results: List[Tuple[str, Optional[float]]] = []
        for idx in range(cutoff, all_ids.shape[0]):
            if idx == 0:
                results.append((self.tokenizer.convert_ids_to_tokens(int(all_ids[idx])), None))
                continue
            log_probs = torch.log_softmax(logits[idx - 1], dim=-1)
            token_id = int(all_ids[idx])
            lp = float(log_probs[token_id])
            token_str = self.tokenizer.convert_ids_to_tokens(token_id)
            results.append((token_str, lp))
        return results
