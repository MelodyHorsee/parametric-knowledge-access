import asyncio
import logging
from datetime import datetime
from typing import Literal

import chz
from tinker_cookbook import cli_utils, model_info
import trivia_env
from tinker_cookbook.rl.train import AsyncConfig, Config, KLReferenceConfig, main
from tinker_cookbook.rl.types import RLDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Simple command-line configuration for RL training."""

    # Model configuration
    model_name: str = "openai/gpt-oss-20b"
    lora_rank: int = 32
    renderer_name: str | None = "gpt_oss_low_reasoning"
    load_checkpoint_path: str | None = None

    # Optional prefill for the answer generation (e.g. "The answer is"),
    # appended right after the thinking block. Pass
    # renderer_name=gpt_oss_no_reasoning alongside this to force an empty
    # thinking block first (see trivia_env.build_gpt_oss_no_reasoning_renderer).
    prefill: str | None = None

    # Environment configuration
    env: str = "trivia"
    seed: int = 0  # Random seed for data shuffling

    # Training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 32
    learning_rate: float = 2e-5
    max_tokens: int = 2048
    kl_penalty_coef: float = 0.01

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    # Logging configuration
    log_path: str | None = None  # If None, uses /tmp/tinker-examples/trivia_rl/{run_name}
    wandb_project: str | None = "triviaqa-rl-nemotron3"
    wandb_name: str | None = None  # If None, uses auto-generated run_name
    compute_post_kl: bool = False

    # Evals
    eval_every: int = 20

    # Checkpointing
    save_every: int = 20
    # TTL for periodic (non-final) checkpoints; the final checkpoint is
    # always kept indefinitely by tinker_cookbook regardless of this value.
    ttl_seconds: int | None = 7 * 24 * 60 * 60  # 7 days

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps_off_policy: int | None = None
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    # Cap run length; override explicitly if a longer run is really needed.
    max_steps: int | None = 1200


def get_dataset_builder(
    env: str,
    batch_size: int,
    model_name: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
    prefill: str | None = None,
) -> RLDatasetBuilder:
    return trivia_env.get_triviaqa_dataset_builder(
        batch_size=batch_size,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        group_size=group_size,
        seed=seed,
        prefill=prefill,
    )


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get tokenizer for stop sequences
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    model_name = cli_config.model_name.replace("/", "-")
    run_name = f"{cli_config.env}-{model_name}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-{cli_config.group_size}group-{cli_config.groups_per_batch}batch-{cli_config.loss_fn}-seed{cli_config.seed}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    # create log path if it doesn't exist
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"logs/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name
    # Create full config
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(
            env=cli_config.env,
            batch_size=cli_config.groups_per_batch,
            model_name=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            seed=cli_config.seed,
            prefill=cli_config.prefill,
        ),
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_reference_config=KLReferenceConfig(
            base_model=cli_config.model_name,
            load_checkpoint_path=None,
        ) if cli_config.kl_penalty_coef > 0 else None,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        ttl_seconds=cli_config.ttl_seconds,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        loss_fn=cli_config.loss_fn,
        max_steps=cli_config.max_steps,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
