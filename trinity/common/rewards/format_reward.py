import re
from typing import Any, Dict, List

from .base import RewardShapper


class FormatRewardShapper(RewardShapper):
    """Shapper for format-based rewards"""

    def __init__(
        self, pattern: str, correct_format_reward: float = 1.0, incorrect_format_reward: float = 0.0
    ):
        self.pattern = re.compile(pattern, re.DOTALL | re.MULTILINE)
        self.correct_format_reward = correct_format_reward
        self.incorrect_format_reward = incorrect_format_reward

    def shape(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        response = sample["response"]
        reward = (
            self.correct_format_reward
            if self.pattern.match(response)
            else self.incorrect_format_reward
        )

        sample["format_reward"] = reward
        return sample

    def batch_shape(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.shape(sample) for sample in samples]
