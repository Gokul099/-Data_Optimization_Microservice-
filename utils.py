import re, time
from functools import wraps
from fastapi import HTTPException

def rate_limiter(max_calls, period):
    """
    Simple in-memory sliding-window rate limiter.
    NOTE: per-process only. For multi-instance use a shared store (Redis).
    """
    calls = []
    def deco(func):
        @wraps(func)
        async def wrapper(*a, **kw):
            nonlocal calls
            now = time.time()
            # keep calls in the last `period` seconds
            calls = [t for t in calls if now - t < period]
            if len(calls) >= max_calls:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            calls.append(now)
            return await func(*a, **kw)
        return wrapper
    return deco

def mask_names(txt: str) -> str:
    """
    Mask obvious person-like tokens with [MASKED].
    - Matches sequences of Titlecase tokens (e.g., "John", "Mary-Anne").
    - Avoids masking acronyms and numbers.
    This is a heuristic — for production use PII detectors / entity recognition.
    """
    # Pattern: word starting with uppercase then lowercase letters (allow hyphenated)
    return re.sub(r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b", "[MASKED]", txt)
