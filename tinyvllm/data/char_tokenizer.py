"""Character-level tokenizer for educational text encoding."""

import string

PAD = 0
UNK = 1

# Printable chars for class names: letters, digits, space, hyphen, slash
_CHARS = string.ascii_lowercase + string.digits + " -/_"
_CHAR_TO_ID = {c: i + 2 for i, c in enumerate(_CHARS)}
_ID_TO_CHAR = {i + 2: c for i, c in enumerate(_CHARS)}


def vocab_size() -> int:
    return len(_CHARS) + 2


def encode(text: str, max_len: int = 32) -> list[int]:
    """Lowercase string → fixed-length token ids (padded with PAD)."""
    text = text.lower().strip()
    ids = [UNK if c not in _CHAR_TO_ID else _CHAR_TO_ID[c] for c in text]
    ids = ids[:max_len]
    pad_count = max_len - len(ids)
    ids.extend([PAD] * pad_count)
    return ids


def decode(ids: list[int]) -> str:
    chars = []
    for i in ids:
        if i == PAD:
            break
        if i == UNK:
            chars.append("?")
        else:
            chars.append(_ID_TO_CHAR.get(i, "?"))
    return "".join(chars).strip()
