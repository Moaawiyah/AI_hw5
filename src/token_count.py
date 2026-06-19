"""Token counting for API economics (§5.5). Uses cl100k_base as a stable proxy."""
import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate token count for a string."""
    if not text:
        return 0
    return len(_ENC.encode(text))


def request_tokens(prompt: str, output: str) -> dict:
    """Return input/output token counts for one request."""
    return {
        "input_tokens": count_tokens(prompt),
        "output_tokens": count_tokens(output),
        "total_tokens": count_tokens(prompt) + count_tokens(output),
    }


def exact_tokens_with_tokenizer(tokenizer, prompt: str, output: str) -> dict:
    """Prefer the model's own tokenizer when available for an exact count."""
    try:
        tin = len(tokenizer(prompt)["input_ids"])
        tout = len(tokenizer(output)["input_ids"])
        return {"input_tokens": tin, "output_tokens": tout, "total_tokens": tin + tout}
    except Exception:
        return request_tokens(prompt, output)
