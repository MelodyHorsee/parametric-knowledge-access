"""
SFT (Supervised Fine-Tuning) training script for TriviaQA.

This script performs supervised fine-tuning instead of RL training,
forcing the model to generate the answer directly after the question
with minimal thinking/reasoning.

Uses the same pattern as the minimal SFT example.
"""
import logging
import time

import chz
import datasets
import tinker
from tinker_cookbook import checkpoint_utils, cli_utils, model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.format_colorized import format_colorized  

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    """Configuration for SFT training."""

    # Model configuration
    model_name: str = "openai/gpt-oss-20b"
    lora_rank: int = 32
    renderer_name: str | None = None  # Will use gpt_oss_low_reasoning for gpt-oss-20b if None

    # Training hyperparameters
    batch_size: int = 512
    learning_rate: float = 2e-5
    max_length: int = 32768
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE
    num_epochs: int = 5

    # Dataset configuration
    seed: int = 0
    max_steps: int | None = None  # Limit number of training steps per epoch

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = "triviaqa-sft"
    wandb_name: str | None = None

    # Checkpointing
    save_every: int = 100  # 0 = disabled

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def build_messages(question: str, answer: str) -> list[dict]:
    """Build conversation messages for training.
    
    For gpt-oss-20b with gpt_oss_low_reasoning renderer, the assistant message
    uses ThinkingPart and TextPart to format:
    - Thinking trace in analysis channel: "Need answer: {answer}."
    - Answer in the final channel: <answer>...</answer>
    
    The renderer formats this as:
    <|channel|>analysis<|message|>Need answer: {answer}.<|end|><|start|>assistant<|channel|>final<|message|>The answer is <answer>{answer}</answer>.
    """
    # Format answer in <answer></answer> tags if not already
    if not answer.startswith("<answer>") and not answer.endswith("</answer>"):
        formatted_answer = f"The answer is <answer>{answer}</answer>."
    else:
        formatted_answer = answer
    
    # Create thinking trace
    thinking_trace = f"Need answer: {answer}."
    
    # Build assistant message with ThinkingPart and TextPart
    assistant_message = {
        "role": "assistant",
        "thinking": thinking_trace,
        "content": formatted_answer,
    }
    
    return [
        {
            "role": "system",
            "content": "You will be given a question. Give your final answer in <answer></answer> tags.",
        },
        {"role": "user", "content": question},
        assistant_message,
    ]


def main(config: Config):
    # Setup logging
    model_name = config.model_name.replace("/", "-")
    if config.log_path is None:
        run_name = (
            f"trivia-sft-{model_name}-{config.lora_rank}rank-"
            f"{config.learning_rate}lr-{config.batch_size}batch-"
            f"seed{config.seed}"
        )
        log_path = f"tmp/triviaqa_sft/{run_name}"
    else:
        log_path = config.log_path
        run_name = config.wandb_name or "triviaqa-sft"

    cli_utils.check_log_dir(log_path, behavior_if_exists=config.behavior_if_log_dir_exists)

    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=config.wandb_project,
        wandb_name=run_name,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    # For gpt-oss-20b, ensure we use gpt_oss_low_reasoning renderer for proper channel formatting
    if config.renderer_name is None:
        if "gpt-oss-20b" in config.model_name.lower():
            renderer_name = "gpt_oss_low_reasoning"
            logger.info(f"Using gpt_oss_low_reasoning renderer for gpt-oss-20b (required for proper channel formatting)")
        else:
            renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    else:
        renderer_name = config.renderer_name
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")
    
    test_messages = build_messages("Test question?", "Test answer")
    test_datum = conversation_to_datum(
        test_messages,
        renderer,
        config.max_length,
        config.train_on_what,
    )

    # Print statement to ensure SFT dataset is built correctly
    print(format_colorized(test_datum.model_input.to_ints(),   
                      test_datum.loss_fn_inputs["weights"].tolist(),   
                      renderer.tokenizer))
    
    # Verify the renderer name is correct for gpt-oss-20b
    if "gpt-oss-20b" in config.model_name.lower() and "gpt_oss_low_reasoning" not in renderer_name.lower():
        logger.warning(
            f"Renderer '{renderer_name}' may not be optimal for gpt-oss-20b. "
            f"Consider using 'gpt_oss_low_reasoning' for proper channel formatting."
        )

    # Load TriviaQA dataset
    logger.info("Loading TriviaQA dataset...")
    dataset = datasets.load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    dataset = dataset.shuffle(seed=config.seed)

    # Limit dataset size if max_steps is set
    if config.max_steps:
        max_examples = config.max_steps * config.batch_size
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    n_train_batches = len(dataset) // config.batch_size
    total_steps = n_train_batches * config.num_epochs
    logger.info(f"Train batches per epoch: {n_train_batches}")
    logger.info(f"Total epochs: {config.num_epochs}")
    logger.info(f"Total steps: {total_steps}")

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    # Check for resuming
    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        resume_loop_state = resume_info.get("loop_state", {})
        start_epoch = resume_info.get("epoch", 0)  
        start_batch = resume_info.get("batch", 0)
        global_step = start_epoch * n_train_batches + start_batch
        logger.info(f"Resuming from epoch {start_epoch}, batch {start_batch} (global step {global_step})")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_epoch = 0
        start_batch = 0
        global_step = 0

    # Training loop (multiple epochs)
    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")

        # Shuffle dataset at the start of each epoch
        epoch_dataset = dataset.shuffle(seed=config.seed + epoch)

        # Determine batch range for this epoch
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        epoch_end_batch = n_train_batches

        for batch_idx in range(epoch_start_batch, epoch_end_batch):
            start_time = time.time()
            global_step = epoch * n_train_batches + batch_idx
            metrics = {}

            # Save checkpoint
            if config.save_every > 0 and global_step % config.save_every == 0 and global_step > 0:
                checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"{global_step:06d}",
                    log_path=log_path,
                    kind="state",
                    loop_state={"epoch": epoch, "batch": batch_idx},
                )

            # Linear learning rate schedule across all epochs
            lr_mult = max(0.0, 1.0 - global_step / total_steps)
            current_lr = config.learning_rate * lr_mult
            adam_params = tinker.AdamParams(learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8)

            # Get training batch and convert to datums online
            batch_start = batch_idx * config.batch_size
            batch_end = min((batch_idx + 1) * config.batch_size, len(epoch_dataset))
            batch_rows = epoch_dataset.select(range(batch_start, batch_end))

            batch = []
            for row in batch_rows:
                try:
                    question = row["question"]
                    # Get the first answer as the target
                    answers = row["answer"]["normalized_aliases"]
                    target_answer = answers[0] if answers else ""

                    if not target_answer:
                        continue

                    # Build conversation messages
                    messages = build_messages(question, target_answer)

                    # Convert to datum
                    datum = conversation_to_datum(
                        messages,
                        renderer,
                        config.max_length,
                        config.train_on_what,
                    )
                    batch.append(datum)
                except Exception as e:
                    logger.warning(f"Failed to create datum: {e}")
                    continue

            if not batch:
                logger.warning(f"Empty batch at epoch {epoch + 1}, step {batch_idx + 1}, skipping")
                continue

            # Training step
            fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
            optim_step_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_step_future.result()

            if optim_result.metrics:
                metrics.update(optim_result.metrics)

            # Compute train metrics
            train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
            train_weights = [d.loss_fn_inputs["weights"] for d in batch]
            train_nll = compute_mean_nll(train_logprobs, train_weights)

            # Log metrics
            metrics.update(
                epoch=epoch + 1,
                num_sequences=len(batch),
                num_tokens=sum(d.model_input.length for d in batch),
                learning_rate=current_lr,
                train_mean_nll=train_nll,
                progress=global_step / total_steps,
                time_total=time.time() - start_time,
            )
            ml_logger.log_metrics(metrics=metrics, step=global_step)

            if (global_step + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{config.num_epochs}, "
                    f"Step {batch_idx + 1}/{n_train_batches} "
                    f"(Global {global_step + 1}/{total_steps}), "
                    f"NLL: {train_nll:.4f}, LR: {current_lr:.2e}"
                )

        # Reset start_batch for next epoch
        start_batch = 0

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=log_path,
        kind="both",
        loop_state={"epoch": config.num_epochs, "batch": n_train_batches},
    )

    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
