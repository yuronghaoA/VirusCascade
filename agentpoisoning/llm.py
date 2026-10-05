from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import openai
except Exception:
    openai = None

try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None


_DEFAULT_TIMEOUT = 120
_FALLBACK_EMBEDDING_DIM = 256
_DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
_FIXED_LOGPROB_FALLBACK_MODEL = "gpt-4o-mini"


@dataclass
class LLMResponse:
    text: str


def _infer_backend(model: str, backend: Optional[str]) -> str:
    chosen = (backend or "").strip().lower()
    if chosen:
        return chosen
    name = (model or "").strip().lower()
    if name.startswith("qwen"):
        return "qwen"
    if name.startswith("gemini"):
        return "gemini"
    if name.startswith("claude"):
        return "claude"
    if name.startswith("deepseek") or name.startswith("siliconflow/deepseek"):
        return "deepseek"
    return "openai"


def _provider_env_key(backend: str) -> Optional[str]:
    candidates = {
        "openai": ("OPENAI_API_KEY",),
        "qwen": ("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "claude": ("ANTHROPIC_API_KEY",),
        "deepseek": ("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY"),
    }.get(backend, ())
    for name in candidates:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _default_api_base(backend: str) -> Optional[str]:
    if backend == "qwen":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if backend == "gemini":
        return "https://generativelanguage.googleapis.com/v1beta"
    if backend == "claude":
        return "https://api.anthropic.com/v1"
    if backend == "deepseek":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return os.environ.get("api_base") or os.environ.get("OPENAI_API_BASE")


def _sanitize_text(text: str) -> str:
    return (text or "").replace("\r\n", "\n").strip()


def _tokenize_for_fallback(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def _normalize_vector(values: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0:
        return values
    return [v / norm for v in values]


class LLMClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        max_completion_tokens: Optional[int] = None,
        top_p: float = 1.0,
        api_key_list: Optional[List[str]] = None,
        logprob_model: Optional[str] = None,
        backend: Optional[str] = None,
        api_base: Optional[str] = None,
        auxiliary_openai_model: Optional[str] = None,
        auxiliary_openai_logprob_model: Optional[str] = None,
        auxiliary_openai_embedding_model: Optional[str] = None,
        auxiliary_openai_api_key_list: Optional[List[str]] = None,
        auxiliary_openai_api_base: Optional[str] = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_completion_tokens = (
            None if max_completion_tokens is None else max(1, int(max_completion_tokens))
        )
        self.top_p = top_p
        self.api_key_list = [str(key).strip() for key in (api_key_list or []) if str(key).strip()]
        self.logprob_model = logprob_model
        self.backend = _infer_backend(model, backend)
        self.api_base = (api_base or _default_api_base(self.backend) or "").rstrip("/")
        self.auxiliary_openai_model = (auxiliary_openai_model or "gpt-4o-mini").strip()
        self.auxiliary_openai_logprob_model = (
            auxiliary_openai_logprob_model or logprob_model or "gpt-3.5-turbo-instruct"
        ).strip()
        self.auxiliary_openai_embedding_model = (
            auxiliary_openai_embedding_model or "text-embedding-3-small"
        ).strip()
        self.auxiliary_openai_api_key_list = [
            str(key).strip()
            for key in (auxiliary_openai_api_key_list or [])
            if str(key).strip()
        ]
        self.auxiliary_openai_api_base = (
            auxiliary_openai_api_base or os.environ.get("OPENAI_API_BASE") or _DEFAULT_OPENAI_API_BASE
        ).rstrip("/")
        self.timeout = max(1, int(timeout))
        self._current_key_idx = 0
        self._aux_openai_key_idx = 0

    def _uses_aux_openai_for_logprobs(self) -> bool:
        return self.backend in {"claude", "deepseek"}

    def _uses_aux_openai_for_embedding(self) -> bool:
        return self.backend in {"claude", "deepseek", "qwen"}

    def _next_key(self) -> Optional[str]:
        if not self.api_key_list:
            return _provider_env_key(self.backend)
        key = self.api_key_list[self._current_key_idx % len(self.api_key_list)]
        self._current_key_idx = (self._current_key_idx + 1) % len(self.api_key_list)
        return key

    def _build_openai_client(self, api_key: Optional[str] = None) -> Optional[Any]:
        if OpenAIClient is None:
            return None
        key = api_key or _provider_env_key("openai") or _provider_env_key(self.backend)
        kwargs: Dict[str, Any] = {}
        if key:
            kwargs["api_key"] = key
        base_url = self._primary_openai_api_base()
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAIClient(**kwargs)

    def _next_aux_openai_key(self) -> Optional[str]:
        if not self.auxiliary_openai_api_key_list:
            if self.backend == "openai" and self.api_key_list:
                return self._next_key()
            return _provider_env_key("openai")
        key = self.auxiliary_openai_api_key_list[
            self._aux_openai_key_idx % len(self.auxiliary_openai_api_key_list)
        ]
        self._aux_openai_key_idx = (
            self._aux_openai_key_idx + 1
        ) % len(self.auxiliary_openai_api_key_list)
        return key

    def _build_aux_openai_client(self, api_key: Optional[str] = None) -> Optional[Any]:
        if OpenAIClient is None:
            return None
        key = api_key or _provider_env_key("openai")
        kwargs: Dict[str, Any] = {}
        if key:
            kwargs["api_key"] = key
        base_url = self._auxiliary_openai_api_base()
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAIClient(**kwargs)

    def _primary_openai_api_base(self) -> str:
        base_url = self.api_base if self.backend == "openai" and self.api_base else (
            os.environ.get("api_base") or os.environ.get("OPENAI_API_BASE") or _DEFAULT_OPENAI_API_BASE
        )
        return (base_url or "").rstrip("/")

    def _auxiliary_openai_api_base(self) -> str:
        base_url = self.auxiliary_openai_api_base or os.environ.get("OPENAI_API_BASE") or _DEFAULT_OPENAI_API_BASE
        return (base_url or "").rstrip("/")

    @staticmethod
    def _field(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        getter = getattr(obj, "get", None)
        if callable(getter):
            try:
                return getter(name, default)
            except TypeError:
                try:
                    value = getter(name)
                    return default if value is None else value
                except Exception:
                    pass
        return getattr(obj, name, default)

    def _configure_legacy_openai(self, *, api_key: Optional[str], api_base: Optional[str]) -> bool:
        if openai is None:
            return False
        if api_key:
            openai.api_key = api_key
        if api_base:
            openai.api_base = api_base
        return True

    def _next_openai_key(self) -> Optional[str]:
        if self.backend == "openai" and self.api_key_list:
            return self._next_key()
        return _provider_env_key("openai")

    @staticmethod
    def _uses_max_completion_tokens(model: str) -> bool:
        normalized = (model or "").strip().lower()
        return normalized.startswith("gpt-5")

    def _openai_generation_token_kwargs(self, model: str) -> Dict[str, int]:
        if self._uses_max_completion_tokens(model):
            token_budget = self.max_completion_tokens if self.max_completion_tokens is not None else self.max_tokens
            return {"max_completion_tokens": max(1, int(token_budget))}
        return {"max_tokens": max(1, int(self.max_tokens))}

    def _generate_with_openai_logprobs(
        self,
        prompt: str,
        *,
        model: str,
        retries: int,
        top_logprobs: int,
        use_auxiliary: bool,
    ) -> Tuple[str, List[Tuple[str, Optional[float]]]]:
        for attempt in range(1, retries + 1):
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "logprobs": True,
                    **self._openai_generation_token_kwargs(model),
                }
                if top_logprobs > 0:
                    request_kwargs["top_logprobs"] = top_logprobs

                if OpenAIClient is not None:
                    if use_auxiliary:
                        client = self._build_aux_openai_client(api_key=self._next_aux_openai_key())
                    else:
                        client = self._build_openai_client(api_key=self._next_openai_key())
                    if client is None:
                        print(
                            f"OpenAI logprob client unavailable "
                            f"(auxiliary={use_auxiliary}, model={model}, attempt={attempt}/{retries})."
                        )
                        return "", []
                    response = client.chat.completions.create(**request_kwargs)
                    choice = response.choices[0] if response.choices else None
                    text = choice.message.content if choice and choice.message else ""
                    logprobs = getattr(choice, "logprobs", None) if choice else None
                    content = getattr(logprobs, "content", None) if logprobs else None
                else:
                    api_key = self._next_aux_openai_key() if use_auxiliary else self._next_openai_key()
                    api_base = self._auxiliary_openai_api_base() if use_auxiliary else self._primary_openai_api_base()
                    if not self._configure_legacy_openai(api_key=api_key, api_base=api_base):
                        print(
                            f"OpenAI logprob client unavailable "
                            f"(auxiliary={use_auxiliary}, model={model}, attempt={attempt}/{retries})."
                        )
                        return "", []
                    response = openai.ChatCompletion.create(**request_kwargs)
                    choices = self._field(response, "choices", []) or []
                    choice = choices[0] if choices else None
                    message = self._field(choice, "message", {}) or {}
                    text = self._field(message, "content", "")
                    logprobs = self._field(choice, "logprobs", None)
                    content = self._field(logprobs, "content", None) if logprobs else None

                tokens: List[Tuple[str, Optional[float]]] = []
                if content:
                    for entry in content:
                        token = self._field(entry, "token", "")
                        lp = self._field(entry, "logprob", None)
                        tokens.append((token, lp))
                return _sanitize_text(text or ""), tokens
            except Exception as exc:
                print(
                    f"OpenAI logprob request failed "
                    f"(auxiliary={use_auxiliary}, model={model}, attempt={attempt}/{retries}): {exc}"
                )
                time.sleep(2)
        return "", []

    def _openai_chat(self, prompt: str, retries: int = 3) -> LLMResponse:
        for attempt in range(1, retries + 1):
            api_key = self._next_openai_key()
            try:
                request_kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    **self._openai_generation_token_kwargs(self.model),
                }
                if OpenAIClient is not None:
                    client = self._build_openai_client(api_key=api_key)
                    if client is None:
                        return LLMResponse(text="")
                    response = client.chat.completions.create(**request_kwargs)
                    choice = response.choices[0] if response.choices else None
                    text = choice.message.content if choice and choice.message else ""
                else:
                    if not self._configure_legacy_openai(api_key=api_key, api_base=self._primary_openai_api_base()):
                        return LLMResponse(text="")
                    response = openai.ChatCompletion.create(**request_kwargs)
                    choices = self._field(response, "choices", []) or []
                    choice = choices[0] if choices else None
                    message = self._field(choice, "message", {}) or {}
                    text = self._field(message, "content", "")
                return LLMResponse(text=_sanitize_text(text or ""))
            except Exception as exc:
                print(
                    f"OpenAI generate request failed "
                    f"(model={self.model}, attempt={attempt}/{retries}): {exc}"
                )
                time.sleep(2)
        return LLMResponse(text="")

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retries: int = 3,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for _ in range(retries):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=payload,
                    params=params,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as exc:
                body = ""
                try:
                    body = exc.response.text if exc.response is not None else ""
                except Exception:
                    body = ""
                print(
                    f"HTTP request failed: method={method} url={url} "
                    f"status={getattr(exc.response, 'status_code', 'unknown')} body={body}"
                )
                last_error = exc
                time.sleep(2)
            except Exception as exc:
                last_error = exc
                time.sleep(2)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Request failed without a captured exception.")

    @staticmethod
    def _extract_openai_like_text(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if choices:
            choice = choices[0] or {}
            message = choice.get("message") or {}
            return _sanitize_text(message.get("content", ""))
        return ""

    @staticmethod
    def _extract_gemini_text(payload: Dict[str, Any]) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            return ""
        content = (candidates[0] or {}).get("content") or {}
        parts = content.get("parts") or []
        chunks: List[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if text:
                chunks.append(str(text))
        return _sanitize_text("".join(chunks))

    @staticmethod
    def _extract_claude_text(payload: Dict[str, Any]) -> str:
        content = payload.get("content") or []
        chunks: List[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                chunks.append(str(block["text"]))
        return _sanitize_text("".join(chunks))

    def _gemini_generate(self, prompt: str, retries: int = 3) -> LLMResponse:
        api_key = self._next_key()
        if not api_key or not self.api_base:
            return LLMResponse(text="")
        url = f"{self.api_base}/models/{self.model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "maxOutputTokens": self.max_tokens,
            },
        }
        response = self._request_json("POST", url, payload=payload, params={"key": api_key}, retries=retries)
        return LLMResponse(text=self._extract_gemini_text(response))

    def _claude_generate(self, prompt: str, retries: int = 3) -> LLMResponse:
        api_key = self._next_key()
        if not api_key or not self.api_base:
            return LLMResponse(text="")
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        response = self._request_json("POST", f"{self.api_base}/messages", headers=headers, payload=payload, retries=retries)
        return LLMResponse(text=self._extract_claude_text(response))

    def _deepseek_generate(self, prompt: str, retries: int = 3) -> LLMResponse:
        api_key = self._next_key()
        if not api_key or not self.api_base:
            return LLMResponse(text="")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        response = self._request_json(
            "POST",
            f"{self.api_base}/chat/completions",
            headers=headers,
            payload=payload,
            retries=retries,
        )
        return LLMResponse(text=self._extract_openai_like_text(response))

    def _qwen_generate(self, prompt: str, retries: int = 3) -> LLMResponse:
        api_key = self._next_key()
        if not api_key or not self.api_base:
            return LLMResponse(text="")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        response = self._request_json(
            "POST",
            f"{self.api_base}/chat/completions",
            headers=headers,
            payload=payload,
            retries=retries,
        )
        return LLMResponse(text=self._extract_openai_like_text(response))

    @staticmethod
    def _extract_openai_like_logprobs(
        payload: Dict[str, Any],
    ) -> Tuple[str, List[Tuple[str, Optional[float]]]]:
        text = LLMClient._extract_openai_like_text(payload)
        choices = payload.get("choices") or []
        if not choices:
            return text, []
        choice = choices[0] or {}
        logprobs = choice.get("logprobs") or {}
        content = logprobs.get("content") or []
        tokens: List[Tuple[str, Optional[float]]] = []
        for entry in content:
            if not isinstance(entry, dict):
                continue
            token = str(entry.get("token", ""))
            tokens.append((token, entry.get("logprob")))
        return text, tokens

    def _qwen_generate_with_logprobs(
        self,
        prompt: str,
        retries: int = 3,
        top_logprobs: int = 0,
    ) -> Tuple[str, List[Tuple[str, Optional[float]]]]:
        api_key = self._next_key()
        if not api_key or not self.api_base:
            return "", []
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "logprobs": True,
        }
        if top_logprobs > 0:
            payload["top_logprobs"] = top_logprobs
        try:
            response = self._request_json(
                "POST",
                f"{self.api_base}/chat/completions",
                headers=headers,
                payload=payload,
                retries=retries,
            )
        except Exception:
            return "", []
        return self._extract_openai_like_logprobs(response)

    def generate(self, prompt: str, retries: int = 3) -> LLMResponse:
        if self.backend == "openai":
            return self._openai_chat(prompt, retries=retries)
        if self.backend == "qwen":
            try:
                return self._qwen_generate(prompt, retries=retries)
            except Exception as exc:
                print(f"Qwen generate request failed (model={self.model}): {exc}")
                return LLMResponse(text="")
        if self.backend == "gemini":
            try:
                return self._gemini_generate(prompt, retries=retries)
            except Exception:
                return LLMResponse(text="")
        if self.backend == "claude":
            try:
                return self._claude_generate(prompt, retries=retries)
            except Exception as exc:
                print(f"Claude generate request failed (model={self.model}): {exc}")
                return LLMResponse(text="")
        if self.backend == "deepseek":
            try:
                return self._deepseek_generate(prompt, retries=retries)
            except Exception:
                return LLMResponse(text="")
        return LLMResponse(text="")

    def generate_with_logprobs(
        self,
        prompt: str,
        retries: int = 3,
        top_logprobs: int = 0,
    ) -> Tuple[str, List[Tuple[str, Optional[float]]]]:
        if self.backend == "openai":
            native_text, native_tokens = self._generate_with_openai_logprobs(
                prompt,
                model=self.model,
                retries=retries,
                top_logprobs=top_logprobs,
                use_auxiliary=False,
            )
            if native_tokens:
                return native_text, native_tokens
            if (self.model or "").strip() == _FIXED_LOGPROB_FALLBACK_MODEL:
                return native_text, native_tokens
            return self._generate_with_openai_logprobs(
                prompt,
                model=_FIXED_LOGPROB_FALLBACK_MODEL,
                retries=retries,
                top_logprobs=top_logprobs,
                use_auxiliary=True,
            )
        if self.backend == "qwen":
            native_text, native_tokens = self._qwen_generate_with_logprobs(
                prompt,
                retries=retries,
                top_logprobs=top_logprobs,
            )
            if native_tokens:
                return native_text, native_tokens
            return self._generate_with_openai_logprobs(
                prompt,
                model=_FIXED_LOGPROB_FALLBACK_MODEL,
                retries=retries,
                top_logprobs=top_logprobs,
                use_auxiliary=True,
            )
        if self._uses_aux_openai_for_logprobs() or self.backend == "gemini":
            return self._generate_with_openai_logprobs(
                prompt,
                model=_FIXED_LOGPROB_FALLBACK_MODEL,
                retries=retries,
                top_logprobs=top_logprobs,
                use_auxiliary=True,
            )
        return self.generate(prompt, retries=retries).text, []

    def _openai_embedding(
        self,
        text: str,
        model: str,
        retries: int = 3,
        use_auxiliary: bool = False,
    ) -> Optional[List[float]]:
        for _ in range(retries):
            try:
                if OpenAIClient is not None:
                    if use_auxiliary:
                        client = self._build_aux_openai_client(api_key=self._next_aux_openai_key())
                    else:
                        client = self._build_openai_client(api_key=self._next_openai_key())
                    if client is None:
                        return None
                    response = client.embeddings.create(model=model, input=text)
                    if not response.data:
                        return None
                    embedding = response.data[0].embedding
                else:
                    api_key = self._next_aux_openai_key() if use_auxiliary else self._next_openai_key()
                    api_base = self._auxiliary_openai_api_base() if use_auxiliary else self._primary_openai_api_base()
                    if not self._configure_legacy_openai(api_key=api_key, api_base=api_base):
                        return None
                    response = openai.Embedding.create(model=model, input=text)
                    data = self._field(response, "data", []) or []
                    if not data:
                        return None
                    embedding = self._field(data[0], "embedding", None)
                return embedding if isinstance(embedding, list) else None
            except Exception:
                time.sleep(2)
        return None

    def _gemini_embedding(self, text: str, model: str, retries: int = 3) -> Optional[List[float]]:
        api_key = self._next_key()
        if not api_key or not self.api_base:
            return None
        url = f"{self.api_base}/models/{model}:embedContent"
        payload = {
            "content": {
                "parts": [{"text": text}],
            }
        }
        try:
            response = self._request_json("POST", url, payload=payload, params={"key": api_key}, retries=retries)
        except Exception:
            return None
        values = (((response.get("embedding") or {}).get("values")) if isinstance(response, dict) else None)
        return values if isinstance(values, list) else None

    def _resolve_qwen_embedding_model(self, model: str) -> str:
        normalized = (model or "").strip()
        if normalized.startswith("text-embedding-v"):
            return normalized
        return "text-embedding-v4"

    def _qwen_embedding(self, text: str, model: str, retries: int = 3) -> Optional[List[float]]:
        api_key = self._next_key()
        if not api_key or not self.api_base:
            return None
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "input": text,
        }
        try:
            response = self._request_json(
                "POST",
                f"{self.api_base}/embeddings",
                headers=headers,
                payload=payload,
                retries=retries,
            )
        except Exception:
            return None
        data = response.get("data") if isinstance(response, dict) else None
        if not isinstance(data, list) or not data:
            return None
        embedding = self._field(data[0], "embedding", None)
        return embedding if isinstance(embedding, list) else None

    def _fallback_embedding(self, text: str) -> List[float]:
        vector = [0.0] * _FALLBACK_EMBEDDING_DIM
        for token in _tokenize_for_fallback(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % _FALLBACK_EMBEDDING_DIM
            sign = -1.0 if digest[2] % 2 else 1.0
            weight = 1.0 + (digest[3] / 255.0)
            vector[index] += sign * weight
        return _normalize_vector(vector)

    def embedding(self, text: str, model: Optional[str] = None, retries: int = 3) -> Optional[List[float]]:
        embedding_model = (model or self.model or "").strip()
        if self._uses_aux_openai_for_embedding():
            delegated_model = self.auxiliary_openai_embedding_model or embedding_model
            if not delegated_model.startswith("text-embedding"):
                delegated_model = "text-embedding-3-small"
            result = self._openai_embedding(text, delegated_model, retries=retries, use_auxiliary=True)
            return result if result is not None else self._fallback_embedding(text)
        if embedding_model.startswith("text-embedding"):
            result = self._openai_embedding(text, embedding_model, retries=retries)
            return result if result is not None else self._fallback_embedding(text)
        if embedding_model.startswith("gemini-embedding"):
            result = self._gemini_embedding(text, embedding_model, retries=retries)
            return result if result is not None else self._fallback_embedding(text)
        return self._fallback_embedding(text)

    def sequence_logprob(self, prompt: str, completion: str, retries: int = 3) -> Optional[float]:
        use_auxiliary = self._uses_aux_openai_for_logprobs()
        if self.backend != "openai" and not use_auxiliary:
            return None
        full_prompt = f"{prompt}{completion}"
        model_name = self.logprob_model if not use_auxiliary else self.auxiliary_openai_logprob_model
        for _ in range(retries):
            try:
                request_kwargs = {
                    "model": model_name,
                    "prompt": full_prompt,
                    "max_tokens": 0,
                    "temperature": 0,
                    "logprobs": 1,
                    "echo": True,
                }
                if OpenAIClient is not None:
                    if use_auxiliary:
                        client = self._build_aux_openai_client(api_key=self._next_aux_openai_key())
                    else:
                        client = self._build_openai_client(api_key=self._next_openai_key())
                    if client is None:
                        return None
                    response = client.completions.create(**request_kwargs)
                    choice = response.choices[0]
                    logprobs = getattr(choice, "logprobs", None)
                else:
                    api_key = self._next_aux_openai_key() if use_auxiliary else self._next_openai_key()
                    api_base = self._auxiliary_openai_api_base() if use_auxiliary else self._primary_openai_api_base()
                    if not self._configure_legacy_openai(api_key=api_key, api_base=api_base):
                        return None
                    response = openai.Completion.create(**request_kwargs)
                    choices = self._field(response, "choices", []) or []
                    if not choices:
                        return None
                    choice = choices[0]
                    logprobs = self._field(choice, "logprobs", None)
                offsets = self._logprob_field(logprobs, "text_offset")
                token_logprobs = self._logprob_field(logprobs, "token_logprobs")
                if not offsets or not token_logprobs:
                    return None
                cutoff = len(prompt)
                total = 0.0
                for offset, logprob in zip(offsets, token_logprobs):
                    if offset >= cutoff and logprob is not None:
                        total += logprob
                return total
            except Exception:
                time.sleep(2)
        return None

    def completion_token_logprobs(
        self, prompt: str, completion: str, retries: int = 3
    ) -> Optional[List[Tuple[str, Optional[float]]]]:
        use_auxiliary = self._uses_aux_openai_for_logprobs()
        if self.backend != "openai" and not use_auxiliary:
            return None
        full_prompt = f"{prompt}{completion}"
        model_name = self.logprob_model if not use_auxiliary else self.auxiliary_openai_logprob_model
        for _ in range(retries):
            try:
                request_kwargs = {
                    "model": model_name,
                    "prompt": full_prompt,
                    "max_tokens": 0,
                    "temperature": 0,
                    "logprobs": 1,
                    "echo": True,
                }
                if OpenAIClient is not None:
                    if use_auxiliary:
                        client = self._build_aux_openai_client(api_key=self._next_aux_openai_key())
                    else:
                        client = self._build_openai_client(api_key=self._next_openai_key())
                    if client is None:
                        return None
                    response = client.completions.create(**request_kwargs)
                    choice = response.choices[0]
                    logprobs = getattr(choice, "logprobs", None)
                else:
                    api_key = self._next_aux_openai_key() if use_auxiliary else self._next_openai_key()
                    api_base = self._auxiliary_openai_api_base() if use_auxiliary else self._primary_openai_api_base()
                    if not self._configure_legacy_openai(api_key=api_key, api_base=api_base):
                        return None
                    response = openai.Completion.create(**request_kwargs)
                    choices = self._field(response, "choices", []) or []
                    if not choices:
                        return None
                    choice = choices[0]
                    logprobs = self._field(choice, "logprobs", None)
                tokens = self._logprob_field(logprobs, "tokens")
                token_logprobs = self._logprob_field(logprobs, "token_logprobs")
                offsets = self._logprob_field(logprobs, "text_offset")
                if not tokens or not offsets:
                    return None
                cutoff = len(prompt)
                results: List[Tuple[str, Optional[float]]] = []
                for token, lp, offset in zip(tokens, token_logprobs, offsets):
                    if offset >= cutoff:
                        results.append((token, lp))
                return results
            except Exception:
                time.sleep(2)
        return None

    @staticmethod
    def _logprob_field(logprobs: Any, name: str) -> Any:
        if logprobs is None:
            return None
        if isinstance(logprobs, dict):
            return logprobs.get(name)
        return getattr(logprobs, name, None)
