from abc import ABC, abstractmethod
import anthropic
import openai
import os


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 1000
    ) -> str:
        """Get completion from LLM"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging/display"""
        pass


class ClaudeProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 1000
    ) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    @property
    def name(self) -> str:
        return f"Claude ({self.model})"


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 1000
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"
