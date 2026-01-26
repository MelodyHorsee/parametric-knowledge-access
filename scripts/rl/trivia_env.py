import math
import re
from functools import partial
from typing import Literal, Sequence, cast

import chz
from datasets import Dataset, load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.recipes.trivia_rl.trivia_grading import (
    extract_answer,
    safe_grade,
)
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.rl.types import Action, StepResult
import tinker
from tinker_cookbook.utils import logtree


class TriviaEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        answers: list[str],
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        timeout: float = 1.0,
    ):
        super().__init__(renderer, convo_prefix)
        self.problem = problem
        self.answers = answers
        self.timeout = timeout

    @classmethod
    def question_suffix(cls) -> str:
        return " Give your final answer in <answer></answer> tags."

    def get_question(self) -> str:
        return self.problem + self.question_suffix()

    def check_format(self, sample_str: str) -> bool:
        try:
            _ = extract_answer(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> dict[str, bool]:
        try:
            answer = extract_answer(sample_str)
        except ValueError:
            return {"EM": False, "Recall": False}
        return safe_grade(answer, self.answers, self.timeout)

    def get_reference_answer(self) -> str:
        return str(self.answers)

    # Override
    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = message["content"]
        correct_format = float(parse_success) and float(self.check_format(content))
        answer_result = self.check_answer(content)
        if answer_result['EM']:
            correct_answer = 1.0
        elif answer_result['Recall']:
            correct_answer = 0.5
        else:
            correct_answer = 0.0
        total_reward = self.format_coef * (correct_format - 1) + correct_answer

        # Log the attempt
        logtree.log_text(f"Problem: {self.get_question()}")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(f"Reference Answer: {self.get_reference_answer()}")
        logtree.log_text(
            f"Format Valid: {'✓' if correct_format else '✗'}, Correct: {'EM' if correct_answer==1 else 'Recall' if correct_answer==0.5 else '✗'}, Reward: {total_reward:.2f}"
        )

        # Compute breakdown metrics
        is_em = float(answer_result['EM'])
        is_recall_only = float(answer_result['Recall'] and not answer_result['EM'])
        is_em_or_recall = float(answer_result['EM'] or answer_result['Recall'])
        is_neither = float(not answer_result['EM'] and not answer_result['Recall'])

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": correct_answer,
                "EM_rate": is_em,
                "Recall_only_rate": is_recall_only,
                "EM_or_Recall_rate": is_em_or_recall,
                "Neither_rate": is_neither,
            },
        )

class TriviaQADataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "validation"] = "train",
        seed: int = 0,
        train_val_split: float = 0.8,
    ):
        if split not in ("train", "validation"):
            raise ValueError("split must be 'train' or 'validation'")

        # Load TriviaQA's "train" split and shuffle
        # (we don't use TriviaQA's "validation" split - that's the held-out test set)
        full_train_ds = cast(Dataset, load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train"))
        full_train_ds = full_train_ds.shuffle(seed=seed)

        # Split into our train (80%) and validation (20%)
        train_size = int(len(full_train_ds) * train_val_split)
        if split == "train":
            self.ds = full_train_ds.select(range(train_size))
        else:  # validation
            self.ds = full_train_ds.select(range(train_size, len(full_train_ds)))

        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix
    
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        try:
            problem = x["question"]
            answers = x["answer"]["normalized_aliases"]
        except Exception as e:
            logger.warning(f"Failed to parse TriviaQA row: {e}")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                TriviaEnv, problem, answers, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
        )

@chz.chz
class TriviaQADatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0

    async def __call__(self) -> tuple[TriviaQADataset, TriviaQADataset]:
        convo_prefix = None
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        datasets = [
            TriviaQADataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split=split,
                seed=self.seed,
            )
            for split in ("train", "validation")
        ]
        print(f"Train questions: {len(datasets[0].ds)}, batches: {len(datasets[0])}")
        print(f"Val questions: {len(datasets[1].ds)}, batches: {len(datasets[1])}")
        return (datasets[0], datasets[1])

def get_triviaqa_dataset_builder(
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
) -> RLDatasetBuilder:
    """
    Args:
        batch_size: Number of groups per batch
        model_name_for_tokenizer: Model name for tokenizer
        renderer_name: Name of the renderer to use
        group_size: Number of environments per group
        seed: Random seed for data shuffling (default: 0)
    Returns:
        The appropriate dataset builder instance
    """

    builder_class = TriviaQADatasetBuilder

    return builder_class(
        batch_size=batch_size,
        model_name_for_tokenizer=model_name_for_tokenizer,
        renderer_name=renderer_name,
        group_size=group_size,
        seed=seed,
    )
