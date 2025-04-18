from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Union


class ComparisonOp(Enum):
    GT = ">"
    LT = "<"
    GE = ">="
    LE = "<="
    EQ = "=="
    NE = "!="


class DataMetricComparator(ABC):
    """Base class for comparing data metrics

    Supports:
    1. Numeric comparisons
    2. Statistical significance tests
    3. Custom comparison logic
    """

    def __init__(self, comparison_type: str = "direct"):
        self.comparison_type = comparison_type
        self._comparators = {
            ComparisonOp.GT: lambda x, y: x > y,
            ComparisonOp.LT: lambda x, y: x < y,
            ComparisonOp.GE: lambda x, y: x >= y,
            ComparisonOp.LE: lambda x, y: x <= y,
            ComparisonOp.EQ: lambda x, y: x == y,
            ComparisonOp.NE: lambda x, y: x != y,
        }

    def compare(
        self,
        metric1: Any,
        metric2: Any,
        comparator: Union[str, ComparisonOp],
        tolerance: float = 0.0,
    ) -> bool:
        """Compare two metrics using specified comparator"""

        if not isinstance(comparator, ComparisonOp):
            comparator = ComparisonOp(comparator)

        if self.comparison_type == "direct":
            return self._direct_compare(metric1, metric2, comparator, tolerance)
        elif self.comparison_type == "statistical":
            return self._statistical_compare(metric1, metric2, comparator)
        else:
            return False  # to be implemented for backup logic

    def _direct_compare(
        self, metric1: Any, metric2: Any, comparator: ComparisonOp, tolerance: float
    ) -> bool:
        """Direct value comparison with tolerance"""
        if tolerance > 0:
            return abs(metric1 - metric2) <= tolerance
        else:
            return self._comparators[comparator](metric1, metric2)

    @abstractmethod
    def _statistical_compare(self, metric1: Any, metric2: Any, comparator: ComparisonOp) -> bool:
        """Statistical significance test"""
        # Implementation for statistical comparison
        pass


class CustomMetricComparator(DataMetricComparator):
    """Custom metric comparator with user-defined comparison logic"""

    def __init__(self, compare_func: Callable[[Any, Any], bool], comparison_type: str = "direct"):
        super().__init__(comparison_type)
        self.compare_func = compare_func

    def compare(
        self,
        metric1: Any,
        metric2: Any,
        comparator: Union[str, ComparisonOp],
        tolerance: float = 0.0,
    ) -> bool:
        return self.compare_func(metric1, metric2)
