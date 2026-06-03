from dataclasses import dataclass

# Tried in order; fall back to the next separator when the chunk is still too large.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    text: str
    start_char: int
    chunk_index: int


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split *text* into overlapping character-level chunks.

    Tries to break at paragraph / sentence / word boundaries before falling
    back to a hard character cut.
    """
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        window = text[start:end]

        # Find a natural break point in the second half of the window.
        if end < len(text):
            for sep in _SEPARATORS[:-1]:  # skip the empty-string fallback
                boundary = window.rfind(sep, chunk_size // 2)
                if boundary != -1:
                    window = window[: boundary + len(sep)]
                    break

        chunk_text_clean = window.strip()
        if chunk_text_clean:
            chunks.append(Chunk(text=chunk_text_clean, start_char=start, chunk_index=idx))
            idx += 1

        if end >= len(text):
            break  # consumed all text; don't slide into a partial tail

        advance = max(1, len(window) - chunk_overlap)
        start += advance

    return chunks
