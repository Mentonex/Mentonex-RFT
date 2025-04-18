from typing import Any, Dict, List, Optional

from trinity.data.core.dataset import RftDataset
from trinity.data.processors.base import BaseDataProcessor
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class DataSynthesizer(BaseDataProcessor):
    """Data synthesizer that generates new samples or augments existing ones.

    Synthesis types:
    1. self_gen: Generate new samples from scratch
    2. mix: Mix multiple datasets
    3. augment: Augment existing samples
    """

    def __init__(
        self,
        dj_cfg: Dict[str, Any],
        synth_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(dj_cfg, **kwargs)
        self.synth_config = synth_config or {}
        # self.synth_history = []

    def process(self, datasets: List[RftDataset], **kwargs: Any) -> RftDataset:
        """Synthesize data based on specified type"""

        result_dataset = None
        synth_type: str = kwargs.get("self_gen", "self_gen")
        try:
            if synth_type == "self_gen":
                self._check_self_gen_params(self.dj_executor)
                result_dataset = self._self_generate()

            elif synth_type == "mix":
                self._check_mix_params(self.dj_executor)
                if not datasets or len(datasets) < 2:
                    raise ValueError("Mix synthesis requires multiple datasets")
                result_dataset = self._mix_datasets(datasets)

            elif synth_type == "augment":
                self._check_augment_params(self.dj_executor)
                if not datasets:
                    raise ValueError("Augmentation requires input dataset")
                result_dataset = self._augment_datasets(datasets)
            else:
                raise ValueError(f"Unknown synthesis type: {synth_type}")

            # Track metrics for synthesized data
            if result_dataset:
                self.trace_interested_metrics(result_dataset)
                self.trace_data_lineage(result_dataset)

            return result_dataset

        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            raise

    def _self_generate(self) -> RftDataset:
        """Generate new samples from scratch"""
        generator = self.synth_config.get("generator")
        if not generator:
            raise ValueError("Generator not configured for self generation")
        return generator.generate()

    def _mix_datasets(self, datasets: List[RftDataset]) -> RftDataset:
        """Mix multiple datasets"""
        weights = self.synth_config.get("mix_weights")
        if weights and len(weights) != len(datasets):
            raise ValueError("Mix weights don't match dataset count")

        return self.dj_executor.sample(datasets, weights)

    def _augment_datasets(self, datasets: List[RftDataset]) -> RftDataset:
        """Augment existing datasets"""
        augmented_datasets = []
        for dataset in datasets:
            augmented = self.dj_executor.process(dataset, self.synth_config.get("augment_ops", []))
            augmented_datasets.append(augmented)
        return self._merge_datasets(augmented_datasets)

    def _check_self_gen_params(self, executor):
        """Validate self generation parameters"""
        required_params = ["generator", "gen_size", "gen_config"]
        for param in required_params:
            if param not in self.synth_config:
                raise ValueError(f"Missing required parameter: {param}")

    def _check_augment_params(self, executor):
        """Validate augmentation parameters"""
        if "augment_ops" not in self.synth_config:
            raise ValueError("No augmentation operators configured")

    def _check_mix_params(self, executor):
        """Validate mix parameters"""
        if "mix_strategy" not in self.synth_config:
            raise ValueError("No mix strategy configured")

    def _merge_datasets(self, datasets: List[RftDataset]) -> RftDataset:
        """Merge multiple datasets into one"""
        # Implementation depends on RftDataset merge capability
        return datasets[0].merge(others=datasets[1:])
