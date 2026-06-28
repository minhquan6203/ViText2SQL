"""
Client gọi API LLM cho Text-to-SQL (DeepSeek, Gemini, OpenAI-compatible).

Dùng trong notebook hoặc script khi không cần chạy mô hình local.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


# Đăng ký mô hình API
API_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "deepseek-chat": {
        "provider": "deepseek",
        "model_id": "deepseek-chat",
        "description": "DeepSeek Chat (API)",
    },
    "deepseek-coder": {
        "provider": "deepseek",
        "model_id": "deepseek-coder",
        "description": "DeepSeek Coder (API, khuyến nghị cho SQL)",
    },
    "gemini-2.0-flash": {
        "provider": "gemini",
        "model_id": "gemini-2.0-flash",
        "description": "Google Gemini 2.0 Flash",
    },
    "gemini-1.5-pro": {
        "provider": "gemini",
        "model_id": "gemini-1.5-pro",
        "description": "Google Gemini 1.5 Pro",
    },
    "gemini-1.5-flash": {
        "provider": "gemini",
        "model_id": "gemini-1.5-flash",
        "description": "Google Gemini 1.5 Flash (nhanh, rẻ)",
    },
}


class BaseAPIClient(ABC):
    """Lớp cơ sở cho client API LLM."""

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Sinh văn bản từ prompt."""

    def generate_with_retry(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Gọi API có retry khi lỗi mạng hoặc rate limit."""
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self.generate(prompt, system_prompt=system_prompt)
            except Exception as error:
                last_error = error
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(
            f"API thất bại sau {self.max_retries} lần thử: {last_error}"
        ) from last_error


DEFAULT_SYSTEM_PROMPT = (
    "Bạn là chuyên gia Text-to-SQL. Nhiệm vụ: viết đúng một câu lệnh SQL "
    "dựa trên schema và câu hỏi tiếng Việt. "
    "Chỉ trả về câu SQL, không giải thích, không dùng markdown trừ khi được yêu cầu."
)


class DeepSeekClient(BaseAPIClient):
    """
    Client DeepSeek qua OpenAI-compatible API.

    Cần biến môi trường DEEPSEEK_API_KEY.
    Đăng ký: https://platform.deepseek.com/
    """

    def __init__(
        self,
        model_id: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        **kwargs: Any,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Thiếu DEEPSEEK_API_KEY. "
                "Đặt trong môi trường hoặc truyền api_key=..."
            )

        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError("Cần cài openai: pip install openai") from error

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class GeminiClient(BaseAPIClient):
    """
    Client Google Gemini API.

    Cần biến môi trường GEMINI_API_KEY (hoặc GOOGLE_API_KEY).
    Lấy key: https://aistudio.google.com/apikey
    """

    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Thiếu GEMINI_API_KEY hoặc GOOGLE_API_KEY. "
                "Lấy tại https://aistudio.google.com/apikey"
            )

        try:
            import google.generativeai as genai
        except ImportError as error:
            raise ImportError(
                "Cần cài google-generativeai: pip install google-generativeai"
            ) from error

        genai.configure(api_key=self.api_key)
        self._genai = genai
        self.model = genai.GenerativeModel(
            model_name=model_id,
            system_instruction=DEFAULT_SYSTEM_PROMPT,
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if system_prompt and system_prompt != DEFAULT_SYSTEM_PROMPT:
            model = self._genai.GenerativeModel(
                model_name=self.model_id,
                system_instruction=system_prompt,
            )
        else:
            model = self.model

        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text or ""


def create_api_client(
    model_key: str,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> BaseAPIClient:
    """
    Tạo client API từ khóa trong API_MODEL_REGISTRY.

    Args:
        model_key: Ví dụ 'deepseek-coder', 'gemini-2.0-flash'
        api_key: API key (tùy chọn, mặc định đọc từ biến môi trường)
    """
    if model_key not in API_MODEL_REGISTRY:
        available = ", ".join(API_MODEL_REGISTRY.keys())
        raise ValueError(f"Mô hình '{model_key}' không hỗ trợ. Có sẵn: {available}")

    config = API_MODEL_REGISTRY[model_key]
    provider = config["provider"]
    model_id = config["model_id"]

    if provider == "deepseek":
        return DeepSeekClient(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "gemini":
        return GeminiClient(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Provider không hỗ trợ: {provider}")
