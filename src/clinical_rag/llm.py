from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from anthropic import Anthropic

from .settings import Settings


@dataclass(frozen=True)
class LLMResult:
    model: str
    text: str


def chat_completion(
    *,
    messages: List[Dict[str, Any]],
    settings: Settings,
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = 700,
) -> LLMResult:
    """
    Calls the Anthropic Messages API and returns the assistant text.
    """
    client = Anthropic(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
        timeout=settings.anthropic_timeout_s,
    )
    
    # Convert OpenAI-style messages to Anthropic format
    system_message = None
    anthropic_messages = []
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "developer":
            # Anthropic uses "system" instead of "developer"
            system_message = content
        elif role == "system":
            system_message = content
        elif role == "user":
            anthropic_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            anthropic_messages.append({"role": "assistant", "content": content})
    
    # Make the API call
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_output_tokens or 700,
        temperature=temperature,
        system=system_message,
        messages=anthropic_messages,
    )
    
    # Extract text from response
    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text
    
    return LLMResult(model=settings.anthropic_model, text=text.strip())