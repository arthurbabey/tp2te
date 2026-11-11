"""Roster-driven scrubbing of names, emails and home-directory paths.

Students write their names in any order and case, with or without accents
("ABLET Anaelle", "Arthur Wuthrich" for a roster "Wüthrich"), and sometimes as
an email ("bastien.vienet@heig-vd.ch"). Matching is therefore done per name
token, case- and accent-insensitive, on a length-preserving folded copy of the
text so that spans map back onto the original.

Notebook and cell metadata are not scrubbed here because they are never
carried: ``notebook.parse_notebook`` keeps only sources and text outputs.
"""

import re
import unicodedata
from collections.abc import Iterable

NAME_PLACEHOLDER = "[NAME]"
EMAIL_PLACEHOLDER = "[EMAIL]"
USER_PLACEHOLDER = "[USER]"

MIN_TOKEN_LEN = 3   # shorter tokens ("de", "da", "Li") are too ambiguous to scrub

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
POSIX_HOME_RE = re.compile(r"(/(?:Users|home)/)(?!\[USER\])([^/\s'\"\\]+)")
WIN_HOME_RE = re.compile(r"([A-Za-z]:\\+(?:Users|Documents and Settings)\\+)(?!\[USER\])([^\\\s'\"/]+)",
                         re.IGNORECASE)

# Letter-or-digit boundaries; "_" counts as a separator (Ablet_Ragoubi.ipynb).
_B_BEFORE = r"(?<![^\W_])"
_B_AFTER = r"(?![^\W_])"


def _fold_char(c: str) -> str:
    base = unicodedata.normalize("NFD", c)[0].lower()
    return base if len(base) == 1 else c


def fold(text: str) -> str:
    """Lowercase and strip accents, keeping exactly one char per input char."""
    return "".join(_fold_char(c) for c in text)


def name_tokens(name: str) -> list[str]:
    """Folded tokens of a name, e.g. "Sam Nadir Nait-Abbou" -> sam, nadir, nait, abbou."""
    toks = re.split(r"[^\w]+|_", fold(unicodedata.normalize("NFC", name)))
    return [t for t in toks if len(t) >= MIN_TOKEN_LEN and not t.isdigit()]


def _token_regex(tokens: Iterable[str]) -> re.Pattern | None:
    toks = sorted(set(tokens), key=len, reverse=True)
    if not toks:
        return None
    return re.compile(_B_BEFORE + "(?:" + "|".join(map(re.escape, toks)) + ")" + _B_AFTER)


class Scrubber:
    def __init__(self, names: Iterable[str]):
        self.tokens = {t for n in names for t in name_tokens(n)}
        self._re = _token_regex(self.tokens)

    def scrub(self, text: str) -> str:
        if not text:
            return text
        text = unicodedata.normalize("NFC", text)
        text = EMAIL_RE.sub(EMAIL_PLACEHOLDER, text)
        text = POSIX_HOME_RE.sub(r"\1" + USER_PLACEHOLDER, text)
        text = WIN_HOME_RE.sub(lambda m: m.group(1) + USER_PLACEHOLDER, text)
        if self._re is None:
            return text
        folded = fold(text)
        out, last = [], 0
        for m in self._re.finditer(folded):
            out.append(text[last:m.start()])
            out.append(NAME_PLACEHOLDER)
            last = m.end()
        out.append(text[last:])
        return "".join(out)


def find_leaks(text: str, names: Iterable[str]) -> list[str]:
    """Pre-flight privacy check on text about to go into a prompt.

    Returns one description per leak; an empty list means the text is clean.
    """
    leaks: list[str] = []
    folded = fold(unicodedata.normalize("NFC", text))
    for name in names:
        for tok in name_tokens(name):
            if re.search(_B_BEFORE + re.escape(tok) + _B_AFTER, folded):
                leaks.append(f"name token {tok!r} of roster entry {name!r}")
    for m in EMAIL_RE.finditer(text):
        leaks.append(f"email {m.group(0)!r}")
    for rx in (POSIX_HOME_RE, WIN_HOME_RE):
        for m in rx.finditer(text):
            leaks.append(f"home path {m.group(0)!r}")
    return leaks
