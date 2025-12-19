# observability.py
import logging
import time
from typing import Any, Optional, Dict, Mapping
from collections import deque
import threading


def get_logger(name: str = "app") -> logging.Logger:
    """
    Get a module-specific logger with a consistent format.
    Safe to call from many files – handlers are attached only once.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Avoid double-logging if root logger is also configured elsewhere
        logger.propagate = False

    return logger


class Timer:
    """
    Simple context manager to time a block and log its duration.

    Usage:
        with Timer(logger, "docling_convert"):
            ...
    """

    def __init__(self, logger: logging.Logger, label: str):
        self.logger = logger
        self.label = label
        self.start: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self):
        self.start = time.time()
        self.logger.info("%s started", self.label)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.time() - self.start
        if exc_type:
            self.logger.error("%s failed after %.2fs", self.label, self.elapsed)
        else:
            self.logger.info("%s completed in %.2fs", self.label, self.elapsed)


def _safe_get(obj: Any, attr: str) -> Any:
    """Helper: handle both dict-like and attribute-like usage objects."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)


def log_llm_usage(
    logger: logging.Logger,
    usage: Any,
    latency_seconds: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log LLM usage metrics (tokens, latency, cache hits).
    `usage` may be an object (resp.usage) or a dict.
    """

    prompt_tokens = _safe_get(usage, "prompt_tokens")
    completion_tokens = _safe_get(usage, "completion_tokens")
    total_tokens = _safe_get(usage, "total_tokens")

    # Extract cached_tokens from prompt_tokens_details (Azure OpenAI prompt caching)
    cached_tokens = 0
    prompt_tokens_details = _safe_get(usage, "prompt_tokens_details")
    if isinstance(prompt_tokens_details, dict):
        cached_tokens = prompt_tokens_details.get("cached_tokens", 0) or 0

    # Calculate cache hit percentage
    cache_pct = (cached_tokens / prompt_tokens * 100) if prompt_tokens else 0.0

    msg = (
        "[LLM] Usage: prompt_tokens=%s, cached_tokens=%s (%.1f%%), "
        "completion_tokens=%s, total_tokens=%s, latency=%.2fs"
    )
    logger.info(
        msg, prompt_tokens, cached_tokens, cache_pct,
        completion_tokens, total_tokens, latency_seconds
    )

    if extra:
        # Log any extra context as a separate line (e.g., model, file, etc.)
        logger.info("[LLM] Extra context: %s", extra)


def log_rate_limit(logger: logging.Logger, headers: Mapping[str, str]) -> None:
    """
    Log Azure OpenAI rate-limit headers, if present.
    This works for both request- and token-based limits.
    """
    if not headers:
        logger.info("[RateLimit] No rate-limit headers found on response.")
        return

    rl_tokens_limit = headers.get("x-ratelimit-limit-tokens")
    rl_tokens_remaining = headers.get("x-ratelimit-remaining-tokens")
    rl_tokens_reset = headers.get("x-ratelimit-reset-tokens")

    rl_req_limit = headers.get("x-ratelimit-limit-requests")
    rl_req_remaining = headers.get("x-ratelimit-remaining-requests")
    rl_req_reset = headers.get("x-ratelimit-reset-requests")

    logger.info(
        "[RateLimit] tokens_limit=%s, tokens_remaining=%s, tokens_reset=%s | "
        "requests_limit=%s, requests_remaining=%s, requests_reset=%s",
        rl_tokens_limit,
        rl_tokens_remaining,
        rl_tokens_reset,
        rl_req_limit,
        rl_req_remaining,
        rl_req_reset,
    )