from typing import Any, Dict, List, Tuple

from .base import RewardShapper


class CompositeRewardShapper(RewardShapper):
    """Combines multiple shappers with weights"""

    def __init__(self, shappers: List[Tuple[RewardShapper, float]]):
        self.shappers = shappers

    def shape(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        total_reward = 0.0
        shapped_sample = sample.copy()

        for shapper, weight in self.shappers:
            shapeged = shapper.shape(sample)
            for key, value in shapeged.items():
                if key.endswith("_reward"):
                    shapped_sample[key] = value
                    total_reward += value * weight

        shapped_sample["total_reward"] = total_reward
        return shapped_sample
