"""Tweet normalisation, tokenisation and a training-only vocabulary."""
import html
import re
from collections import Counter

PAD, UNK, URL, USER = "<pad>", "<unk>", "<url>", "<user>"

_MOJIBAKE_HINT = re.compile("[âÃÂð]")
_URL = re.compile(r"(?:https?://|www\.)\S*", re.IGNORECASE)
_USER = re.compile(r"@\w+")
_RETWEET = re.compile(r"^\s*RT\b:?", re.IGNORECASE)
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
# Placeholders, words (any script, optional apostrophe suffix), and ! / ? which carry tone.
_TOKEN = re.compile(r"<url>|<user>|[^\W_]+(?:'[^\W_]+)?|[!?]")


def fix_mojibake(text):
    """Repair UTF-8 text that was wrongly decoded as Windows-1252 (e.g. 'Weâ€™re' -> 'We’re').

    Only applied when the round trip succeeds, so correctly encoded text is never changed.
    """
    if not _MOJIBAKE_HINT.search(text):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text


def normalise(text):
    text = html.unescape(fix_mojibake(text)).translate(_QUOTES)
    text = _RETWEET.sub(" ", text)
    text = _URL.sub(f" {URL} ", text)
    text = _USER.sub(f" {USER} ", text)
    return text.replace("#", " ").lower()


def tokenize(text):
    return _TOKEN.findall(normalise(text))


class Vocab:
    """Token <-> id mapping. Index 0 is padding, 1 is unknown."""

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        assert self.itos[:2] == [PAD, UNK]

    @classmethod
    def build(cls, token_lists, min_freq=2, max_size=None):
        counts = Counter(token for tokens in token_lists for token in tokens)
        kept = sorted((t for t, c in counts.items() if c >= min_freq and t not in (PAD, UNK)),
                      key=lambda t: (-counts[t], t))
        return cls([PAD, UNK] + kept[:max_size])

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_len):
        """Ids truncated/padded to max_len, and the true length (at least 1, so empty tweets become <unk>)."""
        ids = [self.stoi.get(token, 1) for token in tokens[:max_len]] or [1]
        return ids + [0] * (max_len - len(ids)), len(ids)
