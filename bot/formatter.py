DISCORD_LIMIT = 1900  # stay safely under the 2000-char limit


def chunk_response(text: str) -> list[str]:
    """Split a long response into Discord-safe messages at newline boundaries."""
    if len(text) <= DISCORD_LIMIT:
        return [text]

    parts = []
    current = ""
    for line in text.splitlines(keepends=True):
        # Hard-split a single oversized line so Discord never rejects it
        if len(line) > DISCORD_LIMIT:
            if current:
                parts.append(current.rstrip())
                current = ""
            for i in range(0, len(line), DISCORD_LIMIT):
                parts.append(line[i:i + DISCORD_LIMIT].rstrip())
            continue

        if len(current) + len(line) > DISCORD_LIMIT:
            if current:
                parts.append(current.rstrip())
            current = line
        else:
            current += line

    if current.strip():
        parts.append(current.rstrip())

    return parts or [text[:DISCORD_LIMIT]]
