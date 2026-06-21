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

# Key terms expected in a correct answer about virtual memory paging.
_PAGING_TERMS = ["pag", "virtual memory", "address space", "frame", "swap", "disk", "mmap",
                 "physical memory", "backing store", "page fault"]


def score_output(output: str) -> dict:
    """Heuristic quality score against QUALITY_RUBRIC for the paging prompt.

    Returns a dict mapping each rubric dimension to True/False.
    Designed for comparing quantization levels; not a substitute for human eval.
    """
    low = output.lower()
    term_hits = sum(1 for t in _PAGING_TERMS if t in low)
    on_topic = term_hits >= 2
    coherent = len(output.split()) >= 10
    factually_correct = term_hits >= 3
    # Complete = output ends with sentence-terminator (not truncated mid-word/sentence)
    stripped = output.strip()
    complete = bool(stripped) and stripped[-1] in ".!?"
    return {
        "on_topic": on_topic,
        "coherent": coherent,
        "factually_correct": factually_correct,
        "complete": complete,
        "term_hits": term_hits,
    }
