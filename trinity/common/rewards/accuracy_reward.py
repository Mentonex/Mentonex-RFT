from typing import Any, Callable, Dict, List

from .base import RewardShapper


class AccuracyRewardShapper(RewardShapper):
    """Shapper for accuracy-based rewards"""

    def __init__(
        self,
        answer_parser: Callable[[str], str],
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        kwargs: Dict[str, Any] = {},
    ):
        self.answer_parser = answer_parser
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.response_key = kwargs.get("response", "response")
        self.truth_key = kwargs.get("ground_truth", "ground_truth")

    def shape(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        response = sample[self.response_key]
        truth = sample[self.truth_key]

        parsed_response = self.answer_parser(response)
        reward = self.correct_reward if parsed_response == truth else self.incorrect_reward

        sample["accuracy_reward"] = reward
        return sample

    def batch_shape(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.shape(sample) for sample in samples]
