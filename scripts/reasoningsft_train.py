"""
SFT training script for GPT-OSS-20B on TriviaQA.

Supports two modes:
1. Standard SFT: Train on TriviaQA with synthetic minimal thinking traces.
   The model learns to produce "Need answer: {answer}." as its reasoning.
2. Reasoning-SFT: Train on correct RL trajectories (trajectory_path).
   The model learns to reproduce the full reasoning traces from recalled examples.

See the paper for details on each training mode.
"""
import json
import logging
import random
import re
import time

import chz
import datasets
import tinker
from tinker.types import SamplingParams
from tinker_cookbook import checkpoint_utils, cli_utils, model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.format_colorized import format_colorized
from trivia_grading import extract_answer, exact_match_score, recall_score

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    """Configuration for SFT training."""

    # Model configuration
    model_name: str = "openai/gpt-oss-20b"
    lora_rank: int = 32
    renderer_name: str | None = None

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-5
    warmup_fraction: float = 0.05
    max_length: int = 32768
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE
    num_epochs: int = 8

    # Dataset configuration
    seed: int = 6
    max_steps: int | None = None

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = "triviaqa-sft"
    wandb_name: str | None = None

    # Checkpointing
    save_every: int = 100

    # Service configuration
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # NLL filtering
    nll_threshold: float | None = None

    # Reasoning-SFT: path to JSON with correct trajectories
    trajectory_path: str | None = None

    # Validation
    val_path: str | None = None
    eval_every: int = 100
    gen_eval_every: int = 500


def build_messages(question: str, answer: str) -> list[dict]:
    """Build conversation messages for standard SFT training.

    Constructs a synthetic minimal thinking trace and formatted answer:
    - Analysis channel: "Need answer: {answer}."
    - Final channel: "The answer is <answer>{answer}</answer>."
    """
    if not answer.startswith("<answer>") and not answer.endswith("</answer>"):
        formatted_answer = f"The answer is <answer>{answer}</answer>."
    else:
        formatted_answer = answer

    return [
        {
            "role": "system",
            "content": "You will be given a question. Give your final answer in <answer></answer> tags.",
        },
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "thinking": f"Need answer: {answer}.",
            "content": formatted_answer,
        },
    ]


def build_messages_correct(question: str, raw_prediction: str) -> list[dict]:
    """Build conversation messages for Reasoning-SFT training.

    Extracts thinking and answer from raw_prediction in the format:
    <|channel|>analysis<|message|>{thinking}<|end|><|start|>assistant<|channel|>final<|message|>{answer}
    """
    thinking_match = re.search(
        r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>',
        raw_prediction,
        re.DOTALL,
    )
    thinking_trace = thinking_match.group(1) if thinking_match else ""

    answer_match = re.search(
        r'<\|channel\|>final<\|message\|>(.*)',
        raw_prediction,
        re.DOTALL,
    )
    answer = answer_match.group(1) if answer_match else raw_prediction

    return [
        {
            "role": "system",
            "content": "You will be given a question. Give your final answer in <answer></answer> tags.",
        },
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "thinking": thinking_trace,
            "content": answer,
        },
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
        log_path = f"triviaqa_sft/{run_name}"
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
    if config.renderer_name is None:
        if "gpt-oss-20b" in config.model_name.lower():
            renderer_name = "gpt_oss_low_reasoning"
        else:
            renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    else:
        renderer_name = config.renderer_name
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load data and build test datum
    if config.trajectory_path:
        logger.info(f"Loading trajectories from {config.trajectory_path}...")
        with open(config.trajectory_path) as f:
            trajectory_data = json.load(f)
        all_trajectories = trajectory_data["results"]
        trajectories = [t for t in all_trajectories if t.get("recall", False)]
        logger.info(f"Loaded {len(trajectories)} recalled trajectories (out of {len(all_trajectories)} total)")

        test_messages = build_messages_correct(
            trajectories[0]["question"],
            trajectories[0]["raw_prediction"],
        )
    else:
        trajectories = None
        test_messages = build_messages("Test question?", "Test answer")

    # Verify training format with a test datum
    test_datum = conversation_to_datum(
        test_messages, renderer, config.max_length, config.train_on_what,
    )
    print(format_colorized(
        test_datum.model_input.to_ints(),
        test_datum.loss_fn_inputs["weights"].tolist(),
        renderer.tokenizer,
    ))

    # Load dataset
    if trajectories is not None:
        random.seed(config.seed)
        random.shuffle(trajectories)
        dataset = trajectories
    else:
        logger.info("Loading TriviaQA dataset...")
        dataset = datasets.load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
        dataset = dataset.shuffle(seed=config.seed)

    if config.max_steps:
        max_examples = config.max_steps * config.batch_size
        if trajectories is not None:
            dataset = dataset[:max_examples]
        else:
            dataset = dataset.select(range(min(max_examples, len(dataset))))

    # Load validation set
    val_datums = []
    val_questions = []
    if config.val_path:
        logger.info(f"Loading validation set from {config.val_path}...")
        with open(config.val_path) as f:
            val_data = json.load(f)
        for ex in val_data:
            try:
                answer = ex["ground_truth"][0] if ex["ground_truth"] else ""
                if not answer:
                    continue
                messages = build_messages(ex["question"], answer)
                datum = conversation_to_datum(messages, renderer, config.max_length, config.train_on_what)
                val_datums.append(datum)
                val_questions.append({"question": ex["question"], "ground_truth": ex["ground_truth"]})
            except Exception as e:
                logger.warning(f"Failed to create val datum: {e}")
        logger.info(f"Loaded {len(val_datums)} validation examples")

    n_train_batches = len(dataset) // config.batch_size
    total_steps = n_train_batches * config.num_epochs
    logger.info(f"Train batches per epoch: {n_train_batches}, total steps: {total_steps}")

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)
    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
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

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")

        if trajectories is not None:
            epoch_dataset = list(dataset)
            random.seed(config.seed + epoch)
            random.shuffle(epoch_dataset)
        else:
            epoch_dataset = dataset.shuffle(seed=config.seed + epoch)

        epoch_start_batch = start_batch if epoch == start_epoch else 0

        for batch_idx in range(epoch_start_batch, n_train_batches):
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

            # Learning rate schedule (linear decay)
            lr_mult = max(0.0, 1.0 - global_step / total_steps)
            current_lr = config.learning_rate * lr_mult
            adam_params = tinker.AdamParams(learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8)

            # Build batch
            batch_start_idx = batch_idx * config.batch_size
            batch_end_idx = min((batch_idx + 1) * config.batch_size, len(epoch_dataset))

            if trajectories is not None:
                batch_rows = epoch_dataset[batch_start_idx:batch_end_idx]
            else:
                batch_rows = epoch_dataset.select(range(batch_start_idx, batch_end_idx))

            batch = []
            for row in batch_rows:
                try:
                    if trajectories is not None:
                        messages = build_messages_correct(row["question"], row["raw_prediction"])
                    else:
                        answers = row["answer"]["normalized_aliases"]
                        target_answer = answers[0] if answers else ""
                        if not target_answer:
                            continue
                        messages = build_messages(row["question"], target_answer)

                    datum = conversation_to_datum(
                        messages, renderer, config.max_length, config.train_on_what,
                    )
                    batch.append(datum)
                except Exception as e:
                    logger.warning(f"Failed to create datum: {e}")
                    continue

            if not batch:
                logger.warning(f"Empty batch at epoch {epoch + 1}, step {batch_idx + 1}, skipping")
                continue

            # Optional NLL filtering
            if config.nll_threshold is not None:
                eval_result = training_client.forward(batch, loss_fn="cross_entropy").result()
                filtered_batch = []
                for i, datum in enumerate(batch):
                    logprobs = eval_result.loss_fn_outputs[i]["logprobs"]
                    weights = datum.loss_fn_inputs["weights"]
                    nll = compute_mean_nll([logprobs], [weights])
                    if nll < config.nll_threshold:
                        filtered_batch.append(datum)
                if not filtered_batch:
                    logger.debug(f"No examples passed NLL filter at step {global_step}, skipping")
                    continue
                train_batch = filtered_batch
            else:
                train_batch = batch

            # Forward-backward and optimizer step
            num_batch_tokens = sum(d.model_input.length for d in train_batch)
            fwd_bwd_future = training_client.forward_backward(train_batch, loss_fn="cross_entropy")
            optim_step_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_step_future.result()

            if optim_result.metrics:
                metrics.update(optim_result.metrics)

            # Compute and log train metrics
            train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
            train_weights = [d.loss_fn_inputs["weights"] for d in train_batch]
            train_nll = compute_mean_nll(train_logprobs, train_weights)

            metrics.update(
                epoch=epoch + 1,
                num_sequences=len(train_batch),
                num_sequences_before_filter=len(batch),
                num_tokens=num_batch_tokens,
                learning_rate=current_lr,
                train_mean_nll=train_nll,
                progress=global_step / total_steps,
                time_total=time.time() - start_time,
            )
            ml_logger.log_metrics(metrics=metrics, step=global_step)

            # Validation NLL evaluation
            if val_datums and config.eval_every > 0 and global_step % config.eval_every == 0 and global_step > 0:
                logger.info(f"Step {global_step}: running validation ({len(val_datums)} examples)...")
                val_start_time = time.time()
                val_futures = []
                val_chunks = []
                for vi in range(0, len(val_datums), config.batch_size):
                    chunk = val_datums[vi:vi + config.batch_size]
                    val_chunks.append(chunk)
                    val_futures.append(training_client.forward(chunk, loss_fn="cross_entropy"))
                all_val_logprobs = []
                all_val_weights = []
                for chunk, future in zip(val_chunks, val_futures):
                    val_result = future.result()
                    all_val_logprobs.extend([x["logprobs"] for x in val_result.loss_fn_outputs])
                    all_val_weights.extend([d.loss_fn_inputs["weights"] for d in chunk])
                val_nll = compute_mean_nll(all_val_logprobs, all_val_weights)
                ml_logger.log_metrics(metrics={"val_mean_nll": val_nll, "val_time": time.time() - val_start_time}, step=global_step)
                logger.info(f"Step {global_step}: val NLL = {val_nll:.4f} ({time.time() - val_start_time:.1f}s)")

            # Generation-based eval (EM/Recall)
            if val_questions and config.gen_eval_every > 0 and global_step % config.gen_eval_every == 0 and global_step > 0:
                logger.info(f"Step {global_step}: running generation eval ({len(val_questions)} examples)...")
                gen_start_time = time.time()
                sampling_client = training_client.save_weights_and_get_sampling_client(f"gen_eval_{global_step}")
                stop_sequences = renderer.get_stop_sequences()
                sampling_params = SamplingParams(max_tokens=1028, stop=stop_sequences)

                gen_futures = []
                for vq in val_questions:
                    prompt = renderer.build_generation_prompt([
                        {"role": "system", "content": "You will be given a question. Give your final answer in <answer></answer> tags."},
                        {"role": "user", "content": vq["question"]},
                    ])
                    gen_futures.append(sampling_client.sample(prompt, sampling_params=sampling_params, num_samples=1))

                em_count = 0
                recall_count = 0
                neither_count = 0
                total_count = 0
                for vq, future in zip(val_questions, gen_futures):
                    try:
                        output = future.result()
                        sampled_message, _ = renderer.parse_response(output.sequences[0].tokens)
                        extracted = extract_answer(sampled_message["content"])
                        em = exact_match_score(extracted, vq["ground_truth"])
                        rec = recall_score(extracted, vq["ground_truth"])
                        if em:
                            em_count += 1
                        if rec:
                            recall_count += 1
                        if not em and not rec:
                            neither_count += 1
                        total_count += 1
                    except Exception as e:
                        logger.warning(f"Gen eval failed for question: {e}")
                        total_count += 1

                gen_metrics = {
                    "val_EM_rate": em_count / total_count if total_count > 0 else 0,
                    "val_Recall_only_rate": (recall_count - em_count) / total_count if total_count > 0 else 0,
                    "val_EM_or_Recall_rate": recall_count / total_count if total_count > 0 else 0,
                    "val_Neither_rate": neither_count / total_count if total_count > 0 else 0,
                    "val_gen_time": time.time() - gen_start_time,
                }
                ml_logger.log_metrics(metrics=gen_metrics, step=global_step)
                logger.info(
                    f"Step {global_step}: EM={gen_metrics['val_EM_rate']:.4f}, "
                    f"Recall={gen_metrics['val_EM_or_Recall_rate']:.4f}, "
                    f"Neither={gen_metrics['val_Neither_rate']:.4f} "
                    f"({time.time() - gen_start_time:.1f}s)"
                )

            if (global_step + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{config.num_epochs}, "
                    f"Step {batch_idx + 1}/{n_train_batches} "
                    f"(Global {global_step + 1}/{total_steps}), "
                    f"NLL: {train_nll:.4f}, LR: {current_lr:.2e}"
                )

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
