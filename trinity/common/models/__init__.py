from typing import List

from trinity.common.config import Config
from trinity.common.models.model import InferenceModel


def create_rollout_models(
    config: Config,
) -> List[InferenceModel]:
    """Create `engine_num` rollout models.

    Each model has `tensor_parallel_size` workers.
    """
    import ray
    import vllm
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from trinity.common.models.vllm_async_model import vLLMAysncRolloutModel
    from trinity.common.models.vllm_model import vLLMRolloutModel
    from trinity.utils.log import get_logger

    logger = get_logger(__name__)

    assert vllm.__version__ >= "0.7.3", "Trinity-RFT only supports vllm >= 0.7.3"

    engine_num = config.explorer.engine_num
    tensor_parallel_size = config.explorer.tensor_parallel_size

    vllm_engines = []

    if config.explorer.engine_type == "vllm":
        engine_cls = vLLMRolloutModel
    elif config.explorer.engine_type == "vllm_async":
        engine_cls = vLLMAysncRolloutModel
    else:
        raise ValueError(f"Unknown engine type: {config.explorer.engine_type}")

    bundles = [{"GPU": tensor_parallel_size, "CPU": 1} for _ in range(engine_num)]
    pg = placement_group(bundles)
    ray.get(pg.ready())

    for i in range(engine_num):
        logger.info(f"Creating vLLM engine {i}")
        scheduling_strategy = None

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=i,
        )

        vllm_engines.append(
            engine_cls.options(  # type: ignore [attr-defined]
                num_cpus=0,
                num_gpus=tensor_parallel_size,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                config=config,
            )
        )

    return vllm_engines
