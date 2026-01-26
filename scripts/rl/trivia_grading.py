"""
TriviaQA grading utilities for RL training.
"""
import html
import logging
import re
import signal
import string
import types
from typing import Any, Callable, Dict, Tuple, TypeVar

logger = logging.getLogger(__name__)


T = TypeVar("T")

# ======================================================================
# Math Normalize Functions
# ======================================================================


def normalize_answer(s):
    # HTML cleanup
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\b&\b", " and ", s)

    def normalize_unicode_dashes(text):
        return re.sub(r"[‐-‒–—―−]", " ", text)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(
        remove_articles(
            handle_punc(
                lower(
                    normalize_unicode_dashes(
                        replace_underscore(s)
                    )
                )
            )
        )
    ).strip()


# ======================================================================
# Extract Boxed Functions
# ======================================================================


def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# ======================================================================
# EM and Recall
# ======================================================================


def exact_match_score(prediction, ground_truths):
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(gt) for gt in ground_truths)


def recall_score(prediction, ground_truths):
    pred = normalize_answer(prediction)
    return any(normalize_answer(gt) in pred for gt in ground_truths)


# ======================================================================
# Grader Functions
# ======================================================================


def grade_answer(given_answer: str, ground_truths: list[str]) -> dict[str, bool]:
    """
    EM receives 1.0 reward
    Recall receives 0.5 reward
    False answers receive 0 reward
    """
    if given_answer is None:
        return {"EM": False, "Recall": False}

    EM = exact_match_score(given_answer, ground_truths)
    Recall = recall_score(given_answer, ground_truths)

    return {
        "EM": EM,
        "Recall": Recall
    }


def safe_grade(given_answer: str, ground_truths: list[str], timeout: float = 1.0) -> dict[str, bool]:
    """Grade with timeout protection."""
    import math
    out = run_with_timeout_signal(
        grade_answer, args=(given_answer, ground_truths), timeout_seconds=int(math.ceil(timeout))
    )
    if out is None:
        logger.warning(f"Timeout grading {given_answer} against {ground_truths}")
        return {"EM": False, "Recall": False}
    return out

# ======================================================================
# Timeout Functions
# ======================================================================


# Define a custom exception for timeouts
class TimeoutException(Exception):
    pass


# The handler function that raises the exception
def _timeout_handler(signum: int, frame: types.FrameType | None) -> None:
    raise TimeoutException("Function call timed out")


def run_with_timeout_signal(
    func: Callable[..., T],
    args: Tuple[Any, ...] = (),
    kwargs: Dict[str, Any] = {},
    timeout_seconds: int = 5,
) -> T | None:
    """
    Runs a function with a timeout using signal.alarm (Unix only).

    Args:
        func: The function to execute.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        timeout_seconds: Maximum time allowed in seconds.

    Returns:
        The result of the function call, or None if it times out.
    """
    # Set the signal handler for SIGALRM
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    # Schedule the alarm
    signal.alarm(timeout_seconds)

    try:
        result = func(*args, **kwargs)
    except TimeoutException:
        logger.warning(f"Function timed out after {timeout_seconds} seconds.")
        result = None
    except Exception as e:
        # Handle other exceptions from the function if needed
        logger.warning(f"Function raised an exception: {e}")
        result = None  # Or re-raise
    finally:
        # Disable the alarm
        signal.alarm(0)
        # Restore the original signal handler
        signal.signal(signal.SIGALRM, old_handler)

    return result
