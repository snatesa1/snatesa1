import openai
from typing import Dict, List, Optional
import streamlit as st
import anthropic

LLM_MODELS = {
    "GPT-4": {"provider": "openai", "model": "gpt-4"},
    "GPT-3.5-Turbo": {"provider": "openai", "model": "gpt-3.5-turbo"},
    "Claude 2": {"provider": "anthropic", "model": "claude-2"},
    "Claude Instant": {"provider": "anthropic", "model": "claude-instant-1"}
}

def initialize_llm_client(provider: str, api_key: str) -> Optional[object]:
    """Initialize the LLM client based on provider."""
    try:
        if provider == "openai":
            openai.api_key = api_key
            return openai
        elif provider == "anthropic":
            return anthropic.Anthropic(api_key=api_key)
        return None
    except Exception as e:
        st.error(f"Error initializing {provider} client: {e}")
        return None

def get_llm_response(
    client: object,
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    context: str = ""
) -> str:
    """Get response from LLM."""
    try:
        if provider == "openai":
            if context:
                messages = [{"role": "system", "content": context}] + messages
            # openai>=1.0.0 API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        elif provider == "anthropic":
            prompt = "\n\n".join([
                context if context else "",
                "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
            ])
            response = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens_to_sample=1000,
                temperature=0.7
            )
            return response.completion
        return "Model provider not supported."
    except Exception as e:
        return f"Error getting LLM response: {e}"
