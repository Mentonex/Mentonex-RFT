from abc import ABC
from typing import Any, Callable, List, Optional

from data_juicer.core.data.dj_dataset import Dataset

from trinity.common.config import FormatConfig
from trinity.utils.eval_utils import find_boxed_answer

# TODO, adjust and improve by @yilun


class BaseDataFormatter(ABC):
    """Base class for formatting RFT datasets into different RL framework formats.

    Supports:
    1. Chat/prompt template rendering
    2. Special token handling
    3. Reward field mapping
    4. Format validation
    """

    def __init__(self, config: FormatConfig):
        self.config = config

    def __call__(self, dataset: Dataset, num_proc: int = 1) -> Dataset:
        """Format a dataset using multiprocessing with the implemented transform method"""
        return dataset.map(self.transform, num_proc=num_proc)

    def transform(self, sample: dict) -> dict:
        """Transform a single sample into specific format"""
        raise NotImplementedError

    def _render_template(self, sample: dict) -> dict:
        """Render template with variables"""
        raise NotImplementedError

    def _validate_output(self, formatted_data: Any) -> bool:
        """Validate formatted output"""
        raise NotImplementedError


class ChatTemplateFormatter(BaseDataFormatter, ABC):
    def __init__(self, config: FormatConfig, tokenizer: Optional[Callable] = None):
        super().__init__(config)
        self.tokenizer = tokenizer

    def transform(self, sample: dict) -> dict:
        sample = self._render_template(sample)
        return sample


class ComposedFormatter(BaseDataFormatter):
    def __init__(self, formatters: List[BaseDataFormatter]):
        self.formatters = formatters
        super().__init__(self.formatters[0].config)

    def transform(self, sample: dict) -> dict:
        for formatter in self.formatters:
            sample = formatter.transform(sample)
        return sample


class BoxedMathAnswerFormatter(BaseDataFormatter):
    def transform(self, sample: dict) -> dict:
        sample[self.config.response_key] = find_boxed_answer(sample[self.config.solution_key])
        return sample


class RLHFFormatter(ChatTemplateFormatter):
    def _render_template(self, sample: dict) -> dict:
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            chat = sample[self.config.prompt_key]
            if isinstance(chat, str):
                chat = [
                    {
                        "role": "user",
                        "content": chat,
                    }
                ]
            sample[self.config.prompt_key] = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        elif self.config.chat_template:
            sample[self.config.prompt_key] = self.config.chat_template.format(
                sample[self.config.prompt_key]
            )
        return sample


class RewardFormatter(ChatTemplateFormatter):
    def _render_template(self, sample: dict) -> dict:
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            if self.config.prompt_key:
                prompt = self.tokenizer.apply_chat_template(
                    sample[self.config.prompt_key],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                sample[self.config.prompt_key] = prompt
            else:
                prompt = ""
            sample[self.config.chosen_key] = self.tokenizer.apply_chat_template(
                prompt + sample[self.config.chosen_key], tokenize=False
            )[len(prompt) :]
            sample[self.config.rejected_key] = self.tokenizer.apply_chat_template(
                prompt + sample[self.config.rejected_key], tokenize=False
            )[len(prompt) :]
        elif self.config.chat_template:
            if self.config.prompt_key:
                sample[self.config.prompt_key] = self.config.chat_template.format(
                    sample[self.config.prompt_key]
                )
        return sample


class SFTFormatter(ChatTemplateFormatter):
    def _render_template(self, sample: dict) -> dict:
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            prompt = sample[self.config.prompt_key]
            response = sample[self.config.response_key]

            if isinstance(prompt, str) and isinstance(response, str):
                prompt = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
                response = [
                    {
                        "role": "assistant",
                        "content": response,
                    }
                ]

            sample[self.config.prompt_key] = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            sample[self.config.response_key] = self.tokenizer.apply_chat_template(
                prompt + response,
                tokenize=False,
                add_generation_prompt=False,
            )[len(prompt) :]
        elif self.config.chat_template:
            sample[self.config.prompt_key] = self.config.chat_template.format(
                sample[self.config.prompt_key]
            )
            sample[self.config.response_key] = sample[self.config.response_key]
        return sample
