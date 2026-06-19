"""Fixed prompts + scenario definitions for reproducible benchmarking."""

# Primary benchmark prompt (identical across ALL runs for fair comparison, §5.4).
PRIMARY_PROMPT = (
    "Explain in three short sentences how virtual memory paging works in modern operating systems."
)

# A second, longer prompt used to stress the Prefill stage (more input tokens).
LONG_PROMPT = (
    "You are a patient teacher. " * 20
    + "Now explain in three short sentences how virtual memory paging works."
)

SCENARIOS = {
    "short": {"prompt": PRIMARY_PROMPT, "max_new_tokens": 48},
    "long":  {"prompt": LONG_PROMPT,    "max_new_tokens": 32},
}

# A simple rubric to score output quality across quantization levels (§5.4).
QUALITY_RUBRIC = ["on_topic", "coherent", "factually_correct", "complete"]
