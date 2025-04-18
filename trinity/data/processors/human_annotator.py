import copy
from typing import List, Optional

from data_juicer.core.data import NestedDataset
from jsonargparse import Namespace, namespace_to_dict

from trinity.data.controllers.default_ops import DEFAULT_HUMAN_ANNOTATOR
from trinity.data.core.dataset import RftDataset
from trinity.data.processors.base import BaseDataProcessor
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class DataHumanAnnotator(BaseDataProcessor):
    def __init__(
        self,
        dj_cfg: Optional[Namespace],
        **kwargs,
    ):
        self.dj_cfg = self.keep_human_annotator_op_cfg(dj_cfg)
        super().__init__(self.dj_cfg, **kwargs)

    def keep_human_annotator_op_cfg(self, dj_cfg):
        """Only consider human annotator ops in data-juicer configs."""
        dj_cfg = copy.deepcopy(dj_cfg)
        for i in range(len(dj_cfg.process)):
            if isinstance(dj_cfg.process[i], Namespace):
                dj_cfg.process[i] = namespace_to_dict(dj_cfg.process[i])
        new_process = []
        for op in dj_cfg.process:
            op_name = list(op.keys())[0]
            if op_name in DEFAULT_HUMAN_ANNOTATOR:
                new_process.append(op)
        dj_cfg.process = new_process
        return dj_cfg

    def process(self, datasets: List[RftDataset], **kwargs) -> RftDataset:
        """Annotate dataset with human annotation ops."""
        rft_dataset = datasets[0]
        dataset = NestedDataset.from_dict(rft_dataset.data.to_dict())

        logger.info("Data-Juicer Executor is running...")
        dataset = self.dj_executor.run(dataset)

        rft_dataset.data = dataset
        return rft_dataset
