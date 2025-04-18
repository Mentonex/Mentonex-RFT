from abc import ABC, abstractmethod
from typing import Any, Dict, List


class RewardShapper(ABC):
    """Abstract base class for reward shapper

    Supports:
    1. Rule-based shaping
    2. Model-based shaping
    3. Tool-based shaping
    4. Agent-based shaping
    5. Human-in-the-loop shaping
    """

    @abstractmethod
    def shape(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Shape a sample with rewards"""
        pass

    @abstractmethod
    def batch_shape(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Shape a batch of samples"""
        pass
