import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from data_juicer.ops.base_op import OP as DJOperator

from trinity.data.core.dataset import RftDataset

logger = logging.getLogger(__name__)


class ProcessingResult:
    """Encapsulates the results of data processing"""

    def __init__(
        self,
        success: bool,
        dataset: Optional[RftDataset] = None,
        metrics: Optional[Dict[str, float]] = None,
        error: Optional[str] = None,
    ):
        self.success = success
        self.dataset = dataset
        self.metrics = metrics or {}
        self.error = error


class MetricType(Enum):
    QUALITY = "quality"
    DIFFICULTY = "difficulty"
    DIVERSITY = "diversity"
    QUANTITY = "quantity"


class DJhelper:
    @staticmethod
    def op_lookup_table(metric: str) -> DJOperator:
        """Lookup table for metric tracing operators"""
        # TODO: Implement this method
        return {
            "quality": DJOperator("op_set_1"),
            "difficulty": DJOperator("op_set_2"),
            "diversity": DJOperator("op_set_3"),
            "quantity": DJOperator("op_set_4"),
        }[metric]


class BaseDataProcessor(ABC):
    """Base class for all data processors.

    Features:
    1. Metrics tracking (quality/difficulty/diversity/quantity)
    2. Data lineage tracking
    3. DJ executor integration
    4. Processing pipeline management

    Args:
        dj_cfg: Data-Juicer configuration
        quality_metrics: List of quality metric names
        difficulty_metrics: List of difficulty metric names
        diversity_metrics: List of diversity metric names
        quantity_metrics: List of quantity metric names
    """

    def __init__(
        self,
        dj_cfg: Dict[str, Any],
        quality_metrics: Optional[List[str]] = None,
        difficulty_metrics: Optional[List[str]] = None,
        diversity_metrics: Optional[List[str]] = None,
        quantity_metrics: Optional[List[str]] = None,
    ):
        self.cfg = dj_cfg
        self.quality_metrics = quality_metrics or []
        self.difficulty_metrics = difficulty_metrics or []
        self.diversity_metrics = diversity_metrics or []
        self.quantity_metrics = quantity_metrics or []

        # Initialize executor
        self._init_executor()

        # Metric trackers
        self.metric_history: Dict[MetricType, List[float]] = {
            MetricType.QUALITY: [],
            MetricType.DIFFICULTY: [],
            MetricType.DIVERSITY: [],
            MetricType.QUANTITY: [],
        }

    def _init_executor(self):
        """Initialize DJ executor based on config"""
        try:
            if self.cfg.get("executor_type", "default") == "default":
                from data_juicer.core import DefaultExecutor

                self.dj_executor = DefaultExecutor(self.cfg)
            elif self.cfg["executor_type"] == "ray":
                from data_juicer.core.executor.ray_executor import RayExecutor

                self.dj_executor = RayExecutor(self.cfg)
            else:
                raise ValueError(f"Unknown executor type: {self.cfg['executor_type']}")
        except Exception as e:
            logger.error(f"Failed to initialize executor: {str(e)}")
            raise

    @abstractmethod
    def process(self, datasets: List[RftDataset], **kwargs) -> RftDataset:
        """Process dataset with configured operators"""
        raise NotImplementedError

    def trace_data_lineage(self, dataset):
        """Update dataset usage statistics"""
        try:
            dataset.data_his.update_usage_stats()
        except Exception as e:
            logger.warning(f"Failed to update lineage: {str(e)}")

    def trace_interested_metrics(self, dataset):
        """Trace all configured metrics"""
        try:
            self._trace_metrics(dataset, MetricType.QUALITY)
            self._trace_metrics(dataset, MetricType.DIFFICULTY)
            self._trace_metrics(dataset, MetricType.DIVERSITY)
            self._trace_metrics(dataset, MetricType.QUANTITY)
        except Exception as e:
            logger.error(f"Failed to trace metrics: {str(e)}")

    def _trace_metrics(self, dataset: RftDataset, metric_type: MetricType):
        """Trace specific type of metrics"""
        metrics = getattr(self, f"{metric_type.value}_metrics")
        results: Dict[str, float] = {}

        for metric in metrics:
            try:
                trace_ops = DJhelper.op_lookup_table(metric)
                results[metric] = trace_ops(dataset)
                dataset.stats[metric] = results[metric]
            except Exception as e:
                logger.warning(f"Failed to trace metric {metric}: {str(e)}")

            self.metric_history[metric_type].append(results[metric])
