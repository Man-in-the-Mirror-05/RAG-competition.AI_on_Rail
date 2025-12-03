from typing import Optional

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore


_TOKEN_ENCODER: Optional["tiktoken.Encoding"] = None # type: ignore


def _get_encoder() -> Optional["tiktoken.Encoding"]: # type: ignore
    global _TOKEN_ENCODER
    if _TOKEN_ENCODER is not None:
        return _TOKEN_ENCODER
    if tiktoken is None:
        return None
    _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    return _TOKEN_ENCODER

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)
