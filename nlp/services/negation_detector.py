"""
Vietnamese negation detection for medical text.
Detects negation cues and determines which entities are negated.
"""

import re
from dataclasses import dataclass
from typing import Literal, TypedDict

from nlp.services.constants import (
    POST_NEG,
    PRE_NEG,
    PSEUDO_NEG,
    PUNCT_TERMINATORS,
    SCOPE_TERMINATORS,
    UNCERTAIN_PRE,
)

Direction = Literal["PRE", "POST"]
CueType = Literal["NEG", "PSEUDO", "UNCERTAIN"]


class NegationCueDict(TypedDict):
    """Dictionary representation of a negation cue."""

    text: str
    span: tuple[int, int]
    direction: Direction
    cue_type: CueType
    window: int


class NegationResult(TypedDict):
    """Result from detect_negation method."""

    text: str
    negation_cues: list[NegationCueDict]
    negation_scopes: list[tuple[int, int, CueType]]


@dataclass
class NegationCue:
    """Represents a detected negation cue in text."""

    text: str
    start: int
    end: int
    direction: Direction
    cue_type: CueType
    window: int


@dataclass
class Token:
    """Represents a token with character positions."""

    text: str
    start: int
    end: int


class VietnameseNegationDetector:
    """Detects negation patterns in Vietnamese medical text using NegEx-style rules."""

    def __init__(self) -> None:
        # Compile regex patterns
        self._pre_patterns = self._compile_phrase_pattern(list(PRE_NEG.keys()))
        self._post_patterns = self._compile_phrase_pattern(list(POST_NEG.keys()))
        self._pseudo_patterns = self._compile_phrase_pattern(PSEUDO_NEG)
        self._uncertain_patterns = self._compile_phrase_pattern(
            list(UNCERTAIN_PRE.keys())
        )

        # Token regex (alphanumeric + Vietnamese diacritics)
        self._token_re = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)

    def _compile_phrase_pattern(self, phrases: list[str]) -> re.Pattern[str]:
        """Compile phrases into a regex pattern, longest match first."""
        phrases_sorted = sorted(set(phrases), key=len, reverse=True)
        alt = "|".join(re.escape(p) for p in phrases_sorted)
        return re.compile(rf"\b(?:{alt})\b", flags=re.IGNORECASE)

    def _tokenize(self, text: str) -> list[Token]:
        """Tokenize text into Token objects with positions."""
        return [
            Token(m.group(0), m.start(), m.end()) for m in self._token_re.finditer(text)
        ]

    def _is_terminator_token(self, tok: Token, raw_text: str) -> bool:
        """Check if token is a scope terminator."""
        t = tok.text.lower()
        if t in SCOPE_TERMINATORS:
            return True
        if tok.end < len(raw_text) and raw_text[tok.end] in PUNCT_TERMINATORS:
            return True
        return False

    def find_negation_cues(self, text: str) -> list[NegationCue]:
        """Find all negation cues in text."""
        cues: list[NegationCue] = []
        low = text.lower()

        def add_cues(
            pattern: re.Pattern[str],
            direction: Direction,
            cue_type: CueType,
            win_lookup: dict[str, int] | None = None,
        ) -> None:
            for m in pattern.finditer(low):
                phrase = low[m.start() : m.end()]
                window = (win_lookup or {}).get(phrase, None)
                if window is None and win_lookup:
                    for k, v in win_lookup.items():
                        if phrase == k.lower():
                            window = v
                            break
                if window is None:
                    window = 5
                cues.append(
                    NegationCue(
                        text=text[m.start() : m.end()],
                        start=m.start(),
                        end=m.end(),
                        direction=direction,
                        cue_type=cue_type,
                        window=window,
                    )
                )

        add_cues(self._pseudo_patterns, "PRE", "PSEUDO")
        add_cues(self._pre_patterns, "PRE", "NEG", PRE_NEG)
        add_cues(self._post_patterns, "POST", "NEG", POST_NEG)
        add_cues(self._uncertain_patterns, "PRE", "UNCERTAIN", UNCERTAIN_PRE)

        # Remove overlaps: prefer PSEUDO > NEG/UNCERTAIN, and longer matches
        cues.sort(key=lambda c: (c.start, -(c.end - c.start)))
        pruned: list[NegationCue] = []
        for c in cues:
            if any(
                not (c.end <= x.start or c.start >= x.end)
                and (x.cue_type == "PSEUDO" or (x.end - x.start) >= (c.end - c.start))
                for x in pruned
            ):
                continue
            pruned.append(c)
        return pruned

    def get_negation_scope(
        self, text: str, cue: NegationCue
    ) -> tuple[int, int, CueType]:
        """Calculate the scope of a negation cue."""
        if cue.cue_type == "PSEUDO":
            return (cue.start, cue.end, "PSEUDO")

        tokens = self._tokenize(text)
        cue_tok_idx: int | None = None

        for i, t in enumerate(tokens):
            if not (cue.end <= t.start or cue.start >= t.end):
                cue_tok_idx = i
                break

        if cue_tok_idx is None:
            cue_tok_idx = max(
                0,
                min(
                    len(tokens) - 1,
                    next(
                        (i for i, t in enumerate(tokens) if t.start >= cue.end),
                        len(tokens) - 1,
                    ),
                ),
            )

        if cue.direction == "PRE":
            start_char = cue.start
            end_idx = min(len(tokens) - 1, cue_tok_idx + cue.window)
            end_char = tokens[end_idx].end if tokens else len(text)
            for j in range(
                cue_tok_idx + 1, min(len(tokens), cue_tok_idx + cue.window + 1)
            ):
                if self._is_terminator_token(tokens[j], text):
                    end_char = tokens[j].start
                    break
            return (start_char, end_char, cue.cue_type)

        else:  # POST
            end_char = cue.end
            start_idx = max(0, cue_tok_idx - cue.window)
            start_char = tokens[start_idx].start if tokens else 0
            for j in range(cue_tok_idx - 1, max(-1, cue_tok_idx - cue.window - 1), -1):
                if self._is_terminator_token(tokens[j], text):
                    start_char = tokens[j].end
                    break
            return (start_char, end_char, cue.cue_type)

    def detect_negation(self, text: str) -> NegationResult:
        """
        Detect negation in text.

        Returns:
            Dictionary with:
            - text: Original text
            - negation_cues: List of detected cues
            - negation_scopes: List of (start, end, cue_type) tuples
        """
        cues = self.find_negation_cues(text)
        scopes: list[tuple[int, int, CueType]] = []

        for cue in cues:
            scope = self.get_negation_scope(text, cue)
            if scope[2] != "PSEUDO":
                scopes.append(scope)

        return {
            "text": text,
            "negation_cues": [
                {
                    "text": c.text,
                    "span": (c.start, c.end),
                    "direction": c.direction,
                    "cue_type": c.cue_type,
                    "window": c.window,
                }
                for c in cues
            ],
            "negation_scopes": scopes,
        }

    def is_entity_negated(
        self,
        entity_span: tuple[int, int],
        negation_scopes: list[tuple[int, int, CueType]],
    ) -> str:
        """
        Check if an entity is negated.

        Returns: 'negated' | 'uncertain' | 'affirmed'
        """
        e_start, e_end = entity_span
        hits = [t for (s, e, t) in negation_scopes if not (e_end <= s or e_start >= e)]

        if not hits:
            return "affirmed"

        if "NEG" in ("NEG" if h == "NEG" else "UNCERTAIN" for h in hits):
            return "negated"

        if any(h == "UNCERTAIN" for h in hits):
            return "uncertain"

        return "negated"
