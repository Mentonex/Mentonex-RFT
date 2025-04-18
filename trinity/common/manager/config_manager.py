import streamlit as st
import yaml
from verl.trainer.ppo.ray_trainer import AdvantageEstimator

from trinity.common.rewards import REWARD_FUNCTIONS
from trinity.common.workflows.workflow import WORKFLOWS


def main():
    st.title("Config File Generator")

    st.header("Overvall Config")
    project = st.text_input("Project", "Trinity-RFT")
    group = st.text_input("Group", "test")
    name = st.text_input("Experiment Name", "qwen2.5-1.5B")
    node_num = st.number_input("Node Num", value=1, min_value=1)
    gpu_per_node = st.number_input("GPU Per Node", value=8, min_value=1, max_value=8)

    st.header("Data Config")
    total_epoch = st.number_input("Total Epoch", value=20, min_value=1)
    batch_size_per_gpu = st.number_input("Batch Size Per GPU", value=1, min_value=1)
    dataset_path = st.text_input("Dataset Path", "")
    default_workflow_type = st.selectbox("Default Workflow Type", WORKFLOWS.modules.keys(), index=1)
    dataset_config = st.text_input("Dataset Config (dict required)", r'{"split": "train"}')
    default_reward_fn_type = st.selectbox(
        "Default Reward Fn Type", REWARD_FUNCTIONS.modules.keys(), index=3
    )
    prompt_key = st.text_input("Prompt Key", "question")
    response_key = st.text_input("Response Key", "answer")

    st.header("Model Config")
    model_path = st.text_input("Model Path", "")
    max_prompt_tokens = st.number_input("Max Prompt Tokens", value=256, min_value=1)
    max_response_tokens = st.number_input("Max Response Tokens", value=1024, min_value=1)
    checkpoint_path = st.text_input("Checkpoint Path", "")
    load_checkpoint = st.checkbox("Load Checkpoint", value=True)

    st.header("Explorer and Buffer Config")
    engine_type = st.selectbox("Engine Type", ["vllm_async", "vllm"], index=0)
    if "engine_num" not in st.session_state:
        st.session_state.engine_num = 2
    old_engine_num = min(st.session_state.engine_num, gpu_per_node * node_num)
    engine_num = st.number_input(
        "Engine Num (cannot exceed `gpu_per_node` * `node_num`)",
        value=old_engine_num,
        min_value=1,
        max_value=gpu_per_node * node_num,
    )
    st.session_state.engine_num = engine_num
    tensor_parallel_size = st.number_input(
        "Tensor Parallel Size", value=1, min_value=1, max_value=8
    )
    repeat_times = st.number_input("Repeat Times", value=1, min_value=1)
    with st.expander("Advanced Config"):
        runner_num = st.number_input("Runner Num", value=32, min_value=1)
        enable_prefix_caching = st.checkbox("Enable Prefix Caching", value=False)
        enforce_eager = st.checkbox("Enforce Eager", value=True)
        dtype = st.selectbox("Dtype", ["float16", "bfloat16", "float32"], index=1)
        temperature = st.number_input("Temperature", value=1.0, min_value=0.0, max_value=2.0)
        top_p = st.number_input("Top P", value=1.0, min_value=0.0, max_value=1.0)
        top_k = st.number_input("Top K", value=1, min_value=1, max_value=512)
        seed = st.number_input("Seed", value=42)
        logprobs = st.number_input("Logprobs", value=0, min_value=0, max_value=20)
        use_ray = st.checkbox("Use Ray", value=False)
        backend = st.selectbox("Backend", ["nccl"], index=0)
        max_pending_requests = st.number_input("Max Pending Requests", value=32, min_value=1)
        max_waiting_steps = st.number_input("Max Waiting Steps", value=4, min_value=1)
    gpu_num = gpu_per_node * node_num - engine_num

    st.header("Buffer and Synchronizer Config")
    storage_type = st.selectbox("Storage Type", ["sql", "redis", "queue"], index=2)
    if storage_type in ["sql", "redis"]:
        db_url = st.text_input("DB URL (used for sql and redis)", "")
    else:
        db_url = ""
    if storage_type == "sql":
        max_retry_times = st.number_input("Max Retry Times", value=3, min_value=1)
        max_retry_interval = st.number_input("Max Retry Interval", value=1, min_value=1)
    else:
        max_retry_times = max_retry_interval = 0

    sync_method = st.selectbox("Sync Method", ["online", "offline"], index=0)
    sync_iteration_interval = st.number_input("Sync Iteration Interval", value=10, min_value=1)

    st.header("Trainer Config")
    trainer_type = st.selectbox("Trainer Type", ["verl"], index=0)
    sft_warmup_iteration = st.number_input("SFT Warmup Iteration", value=0, min_value=0)
    eval_interval = st.number_input("Eval Interval", value=1000, min_value=1)
    if trainer_type == "verl":
        trainer_config_path = st.text_input("Trainer Config Path", "")

        st.subheader("RL Algorithm Config")
        gamma = st.number_input("Gamma", value=1.0)
        lam = st.number_input("lam", value=1.0)
        adv_estimator = st.selectbox(
            "Advantage Estimator", [member.value for member in AdvantageEstimator], index=0
        )
        kl_penalty = st.selectbox("KL Penalty", ["kl", "abs", "mse", "low_var_kl"], index=0)
        kl_ctrl_type = st.selectbox("KL Ctrl Type", ["fixed", "adaptive"], index=0)
        kl_ctrl_coef = st.number_input("KL Ctrl Coef", value=0.001)

        st.subheader("Training Strategy Config")
        if "save_freq" not in st.session_state:
            st.session_state.save_freq = 100
        if sync_method == "online":
            save_freq = st.number_input("Save Freq", value=st.session_state.save_freq, min_value=1)
            st.session_state.save_freq = save_freq
        else:
            st.session_state.save_freq = sync_iteration_interval
            save_freq = st.number_input(
                "Save Freq (locked for offline update)",
                value=st.session_state.save_freq,
                min_value=1,
                disabled=True,
            )
        training_strategy = st.selectbox(
            "Training Strategy (megatron is not tested)", ["fsdp", "megatron"], index=0
        )
        if training_strategy == "fsdp":
            param_offload = st.checkbox("Param Offload", value=False)
            optimizer_offload = st.checkbox("Optimizer Offload", value=False)
            fsdp_config = {
                "wrap_policy": {"min_num_params": 0},
                "param_offload": param_offload,
                "optimizer_offload": optimizer_offload,
                "fsdp_size": -1,
            }
        else:
            fsdp_config = {}

        st.subheader("Actor Model Config")
        actor_ppo_micro_batch_size_per_gpu = st.number_input(
            "PPO Micro Batch Size Per GPU for actor", value=4, min_value=1
        )
        actor_grad_clip = st.number_input(
            "Grad Clip for actor", value=1.0, min_value=0.0, max_value=1.0
        )
        actor_clip_ratio = st.number_input(
            "Clip Ratio for actor", value=0.2, min_value=0.0, max_value=1.0
        )
        actor_entropy_coeff = st.number_input(
            "Entropy Coeff for actor", value=0.001, min_value=0.0, max_value=1.0
        )
        actor_use_kl_loss = st.checkbox("Use KL Loss for actor (True for GRPO)", value=False)
        actor_kl_loss_coef = st.number_input(
            "KL Loss Coef for actor", value=0.001, min_value=0.0, max_value=1.0
        )
        actor_kl_loss_type = st.selectbox(
            "KL Loss Type for actor", ["kl", "abs", "mse", "low_var_kl"], index=0
        )
        actor_ulysses_sequence_parallel_size = st.number_input(
            "Ulysses Sequence Parallel Size for actor", value=1, min_value=1, max_value=8
        )
        actor_checkpoint = st.multiselect(
            "Checkpoint for actor",
            ["model", "hf_model", "optimizer", "extra"],
            default=["model", "hf_model", "optimizer", "extra"],
        )
        actor_lr = st.number_input(
            "Learning Rate for actor", value=1e-6, min_value=1e-7, max_value=1e-3, format="%.1e"
        )
        actor_alg_type = st.selectbox("Algorithm Type", ["ppo", "opmd", "pairwise_opmd"], index=0)
        # if actor_alg_type != 'ppo':
        actor_alg_type_is_opmd = actor_alg_type != "ppo"
        actor_tau = st.number_input(
            "Tau for actor",
            value=0.0,
            min_value=0.0,
            max_value=1.0,
            format="%.1e",
            disabled=not actor_alg_type_is_opmd,
        )
        actor_opmd_baseline = st.selectbox(
            "OPMD Baseline for actor",
            ["mean", "logavgexp"],
            index=0,
            disabled=not actor_alg_type_is_opmd,
        )
        actor_use_uid = st.checkbox(
            "Use UID for actor", value=False, disabled=not actor_alg_type_is_opmd
        )
        # else:
        #     actor_tau = 0.0
        #     actor_opmd_baseline = 'mean'
        #     actor_use_uid = False

        st.subheader("Reference Model Config")
        ref_log_prob_micro_batch_size_per_gpu = st.number_input(
            "Log Prob Micro Batch Size Per GPU for reference", value=16, min_value=1
        )

        st.subheader("Critic Model Config")
        critic_lr = st.number_input(
            "Learning Rate for critic", value=1e-6, min_value=1e-7, max_value=1e-3, format="%.1e"
        )
        critic_model_path = st.text_input("Critic Model Path", "")
        critic_ppo_micro_batch_size_per_gpu = st.number_input(
            "PPO Micro Batch Size Per GPU for critic", value=8, min_value=1
        )
        critic_ulysses_sequence_parallel_size = st.number_input(
            "Ulysses Sequence Parallel Size for critic", value=1, min_value=1, max_value=8
        )
        critic_grad_clip = st.number_input(
            "Grad Clip for critic", value=1.0, min_value=0.0, max_value=1.0
        )
        critic_cliprange_value = st.number_input(
            "Cliprange Value for critic", value=0.5, min_value=0.0, max_value=1.0
        )

        rollout_node_num = engine_num * tensor_parallel_size // gpu_per_node
        trainer_nnodes = node_num - rollout_node_num
        if node_num == 1:
            trainer_n_gpus_per_node = gpu_per_node - engine_num * tensor_parallel_size
        else:
            trainer_n_gpus_per_node = gpu_per_node
        trainer_config = {
            "data": {
                "tokenizer": None,
                "train_files": "placeholder",
                "val_files": "placeholder",
                "prompt_key": "placeholder",
                "max_prompt_length": max_prompt_tokens,
                "max_response_length": max_response_tokens,
                "train_batch_size": batch_size_per_gpu * gpu_num * repeat_times,
                "val_batch_size": None,
                "return_raw_input_ids": False,
                "return_raw_chat": False,
                "shuffle": True,
                "filter_overlong_prompts": False,
                "truncation": "error",
                "image_key": "images",
            },
            "actor_rollout_ref": {
                "hybrid_engine": True,
                "model": {
                    "path": model_path,
                    "external_lib": None,
                    "override_config": {},
                    "enable_gradient_checkpointing": True,
                    "use_remove_padding": True,
                },
                "actor": {
                    "strategy": training_strategy,
                    "ppo_mini_batch_size": batch_size_per_gpu * gpu_num,
                    "ppo_micro_batch_size_per_gpu": actor_ppo_micro_batch_size_per_gpu,
                    "use_dynamic_bsz": True,
                    "ppo_max_token_len_per_gpu": repeat_times
                    * (max_prompt_tokens + max_response_tokens),
                    "grad_clip": actor_grad_clip,
                    "clip_ratio": actor_clip_ratio,
                    "entropy_coeff": actor_entropy_coeff,
                    "use_kl_loss": actor_use_kl_loss,
                    "kl_loss_coef": actor_kl_loss_coef,
                    "kl_loss_type": actor_kl_loss_type,
                    "ppo_epochs": 1,  # TODO
                    "shuffle": False,
                    "ulysses_sequence_parallel_size": actor_ulysses_sequence_parallel_size,
                    "checkpoint": {"contents": actor_checkpoint},
                    "optim": {
                        "lr": actor_lr,
                        "lr_warmup_steps_ratio": 0.0,  # TODO
                        "warmup_style": "constant",  # TODO
                        "total_training_steps": -1,
                    },
                    "fsdp_config": fsdp_config,
                    "alg_type": actor_alg_type,
                    "tau": actor_tau,
                    "opmd_baseline": actor_opmd_baseline,
                    "use_uid": actor_use_uid,
                },
                "ref": {
                    "fsdp_config": fsdp_config,
                    "log_prob_micro_batch_size_per_gpu": ref_log_prob_micro_batch_size_per_gpu,
                    "log_prob_use_dynamic_bsz": "${actor_rollout_ref.actor.use_dynamic_bsz}",
                    "log_prob_max_token_len_per_gpu": "${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}",
                    "ulysses_sequence_parallel_size": "${actor_rollout_ref.actor.ulysses_sequence_parallel_size}",
                },
                "rollout": {
                    "name": "vllm",
                    "temperature": temperature,
                    "top_k": -1,
                    "top_p": 1,
                    "use_fire_sampling": False,
                    "prompt_length": "${data.max_prompt_length}",
                    "response_length": "${data.max_response_length}",
                    "dtype": "bfloat16",
                    "gpu_memory_utilization": 0.4,
                    "ignore_eos": False,
                    "enforce_eager": True,
                    "free_cache_engine": True,
                    "load_format": "dummy_dtensor",
                    "tensor_model_parallel_size": 2,
                    "max_num_batched_tokens": 8192,
                    "max_model_len": None,
                    "max_num_seqs": 1024,
                    "log_prob_micro_batch_size_per_gpu": 4,
                    "log_prob_use_dynamic_bsz": "${actor_rollout_ref.actor.use_dynamic_bsz}",
                    "log_prob_max_token_len_per_gpu": "${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}",
                    "disable_log_stats": True,
                    "enable_chunked_prefill": True,
                    "do_sample": True,
                    "n": repeat_times,
                },
            },
            "critic": {
                "strategy": training_strategy,
                "optim": {
                    "lr": critic_lr,
                    "lr_warmup_steps_ratio": 0.0,  # TODO
                    "warmup_style": "constant",  # TODO
                    "total_training_steps": -1,  # TODO
                },
                "model": {
                    "path": critic_model_path,
                    "tokenizer_path": "${actor_rollout_ref.model.path}",
                    "override_config": {},
                    "external_lib": "${actor_rollout_ref.model.external_lib}",
                    "enable_gradient_checkpointing": True,
                    "use_remove_padding": False,
                    "fsdp_config": fsdp_config,
                },
                "ppo_mini_batch_size": "${actor_rollout_ref.actor.ppo_mini_batch_size}",
                "ppo_micro_batch_size_per_gpu": critic_ppo_micro_batch_size_per_gpu,
                "forward_micro_batch_size": "${critic.ppo_micro_batch_size}",
                "forward_micro_batch_size_per_gpu": "${critic.ppo_micro_batch_size_per_gpu}",
                "use_dynamic_bsz": "${actor_rollout_ref.actor.use_dynamic_bsz}",
                "ppo_max_token_len_per_gpu": repeat_times
                * (max_prompt_tokens + max_response_tokens)
                * 2,
                "forward_max_token_len_per_gpu": "${critic.ppo_max_token_len_per_gpu}",
                "ulysses_sequence_parallel_size": critic_ulysses_sequence_parallel_size,
                "ppo_epochs": "${actor_rollout_ref.actor.ppo_epochs}",
                "shuffle": "${actor_rollout_ref.actor.shuffle}",
                "grad_clip": critic_grad_clip,
                "cliprange_value": critic_cliprange_value,
            },
            "reward_model": {
                "enable": False,
                "strategy": "fsdp",
                "model": {
                    "input_tokenizer": "${actor_rollout_ref.model.path}",
                    "path": "~/models/FsfairX-LLaMA3-RM-v0.1",
                    "external_lib": "${actor_rollout_ref.model.external_lib}",
                    "use_remove_padding": False,
                    "fsdp_config": {"min_num_params": 0, "param_offload": False, "fsdp_size": -1},
                },
                "ulysses_sequence_parallel_size": 1,
                "use_dynamic_bsz": "${critic.use_dynamic_bsz}",
                "forward_max_token_len_per_gpu": "${critic.forward_max_token_len_per_gpu}",
                "reward_manager": "naive",
            },
            "custom_reward_function": {"path": None, "name": "compute_score"},
            "algorithm": {
                "gamma": gamma,
                "lam": lam,
                "adv_estimator": adv_estimator,
                "kl_penalty": kl_penalty,
                "kl_ctrl": {"type": kl_ctrl_type, "kl_coef": kl_ctrl_coef},
            },
            "trainer": {
                "balance_batch": True,  # TODO
                "total_epochs": 15,  # TODO
                "project_name": project,
                "experiment_name": name,
                "logger": ["wandb"],  # TODO
                "val_generations_to_log_to_wandb": 0,
                "nnodes": trainer_nnodes,
                "n_gpus_per_node": trainer_n_gpus_per_node,
                "save_freq": save_freq,
                "resume_mode": "auto",  # TODO
                "resume_from_path": False,  # TODO
                "test_freq": 100,
                "critic_warmup": 0,
                "default_hdfs_dir": None,
                "remove_previous_ckpt_in_save": False,  # TODO
                "del_local_ckpt_after_load": False,  # TODO
                "default_local_dir": checkpoint_path,
                "val_before_train": False,
                "training_rollout_mode": "parallel",
                "enable_exp_buffer": True,
                "steps_per_epoch": 1280,
                "sync_freq": sync_iteration_interval,
            },
        }
    else:
        raise ValueError(f"Invalid trainer type: {trainer_type}")

    flag = False
    if not dataset_path.strip():
        flag = True
        st.warning("Please input dataset path")
    if not model_path.strip():
        flag = True
        st.warning("Please input model path")
    if not checkpoint_path.strip():
        flag = True
        st.warning("Please input checkpoint path")
    if not trainer_config_path.strip():
        flag = True
        st.warning("Please input trainer config path")
    if not critic_model_path:
        critic_model_path = model_path
    if st.button("Generate Config", disabled=flag):
        config = {
            "data": {
                "total_epochs": total_epoch,
                "batch_size": batch_size_per_gpu * gpu_num,
                "dataset_path": dataset_path,
                "default_workflow_type": default_workflow_type,
                "dataset_config": eval(dataset_config),
                "default_reward_fn_type": default_reward_fn_type,
                "format_config": {
                    "prompt_key": prompt_key,
                    "response_key": response_key,
                },
            },
            "model": {
                "model_path": model_path,
                "max_prompt_tokens": max_prompt_tokens,
                "max_response_tokens": max_response_tokens,
                "checkpoint_path": checkpoint_path,
                "load_checkpoint": load_checkpoint,
            },
            "cluster": {
                "node_num": node_num,
                "gpu_per_node": gpu_per_node,
            },
            "buffer": {
                "storage_type": storage_type,
                "db_url": db_url,
                "read_batch_size": batch_size_per_gpu * gpu_num * repeat_times,
                "max_retry_times": max_retry_times,
                "max_retry_interval": max_retry_interval,
            },
            "explorer": {
                "engine_type": engine_type,
                "engine_num": engine_num,
                "runner_num": runner_num,
                "tensor_parallel_size": tensor_parallel_size,
                "enable_prefix_caching": enable_prefix_caching,
                "enforce_eager": enforce_eager,
                "dtype": dtype,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed,
                "logprobs": logprobs,
                "repeat_times": repeat_times,
                "use_ray": use_ray,
                "backend": backend,
                "max_pending_requests": max_pending_requests,
                "max_waiting_steps": max_waiting_steps,
            },
            "synchronizer": {
                "sync_method": sync_method,
                "sync_iteration_interval": sync_iteration_interval,
            },
            "trainer": {
                "trainer_type": trainer_type,
                "trainer_config_path": trainer_config_path,
                "sft_warmup_iteration": sft_warmup_iteration,
                "eval_interval": eval_interval,
            },
            "monitor": {
                "cache_root_dir": "",
                "project": project,
                "group": group,
                "name": name,
            },
        }
        st.header("Generated Config File")
        st.subheader("Overall Config File")
        yaml_config = yaml.dump(config, allow_unicode=True, sort_keys=False)
        st.code(yaml_config, language="yaml")
        st.subheader("Trainer Config File")
        trainer_config = yaml.dump(trainer_config, allow_unicode=True, sort_keys=False)
        st.code(trainer_config, language="yaml")


if __name__ == "__main__":
    main()
