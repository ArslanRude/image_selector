"""Remember the user's feedback on past verdicts and turn it into lessons for future decisions.

Each verdict the user rates earns a reward: +1 when the pick was right, -1 when the user had to
correct it. The feedback most relevant to the current request, mistakes first, is passed to Jev so it
learns the user's preferences.
"""

import json
import logging
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from image_selector.vision import ImageDescription

logger = logging.getLogger(__name__)

DEFAULT_LESSON_LIMIT = 8

# Words too common to say two requests are about the same thing.
_STOPWORDS = frozenset(
    "a an and are as at be by for from in into is it of on or the this that to with without image photo picture".split()
)


@dataclass(frozen=True)
class Feedback:
    user_prompt: str
    descriptions: tuple[ImageDescription, ...]
    predicted_index: int
    correct_index: int
    created_at: str
    instructions: str = ""
    reason: str = ""
    """The user's explanation of their choice, if they gave one."""

    @property
    def was_right(self) -> bool:
        return self.predicted_index == self.correct_index

    @property
    def reward(self) -> int:
        return 1 if self.was_right else -1

    def as_lesson(self) -> dict:
        """The form Jev sees: what the user preferred over the alternatives for a given request."""
        lesson = {
            "prompt": self.user_prompt,
            "user_preferred": self.descriptions[self.correct_index].as_dict(),
            "user_rejected": [description.as_dict() for index, description in enumerate(self.descriptions) if index != self.correct_index],
            "earlier_pick_was_right": self.was_right,
        }
        if self.instructions:
            lesson["instructions"] = self.instructions
        if self.reason:
            lesson["user_reason"] = self.reason
        return lesson


@dataclass(frozen=True)
class Stats:
    right: int = 0
    wrong: int = 0

    @property
    def total(self) -> int:
        return self.right + self.wrong

    @property
    def reward(self) -> int:
        return self.right - self.wrong

    @property
    def accuracy(self) -> float | None:
        return self.right / self.total if self.total else None


class FeedbackMemory:
    """Append-only JSON Lines store of rated verdicts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(
        self,
        user_prompt: str,
        descriptions: Sequence[ImageDescription],
        predicted_index: int,
        correct_index: int,
        *,
        instructions: str = "",
        reason: str = "",
    ) -> Feedback:
        if not 0 <= correct_index < len(descriptions):
            raise ValueError(f"correct_index {correct_index} is out of range for {len(descriptions)} images.")
        feedback = Feedback(
            user_prompt=user_prompt,
            descriptions=tuple(descriptions),
            predicted_index=predicted_index,
            correct_index=correct_index,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            instructions=instructions.strip(),
            reason=reason.strip(),
        )
        line = json.dumps(
            {
                "user_prompt": feedback.user_prompt,
                "instructions": feedback.instructions,
                "descriptions": [description.as_dict() for description in feedback.descriptions],
                "predicted_index": feedback.predicted_index,
                "correct_index": feedback.correct_index,
                "reason": feedback.reason,
                "reward": feedback.reward,
                "created_at": feedback.created_at,
            }
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
        return feedback

    def load(self) -> list[Feedback]:
        """All feedback, oldest first. Unreadable lines are logged and skipped so one bad line can't lose the rest."""
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        entries = []
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                entries.append(
                    Feedback(
                        user_prompt=raw["user_prompt"],
                        descriptions=tuple(ImageDescription.from_dict(description) for description in raw["descriptions"]),
                        predicted_index=raw["predicted_index"],
                        correct_index=raw["correct_index"],
                        created_at=raw["created_at"],
                        instructions=raw.get("instructions", ""),
                        reason=raw.get("reason", ""),
                    )
                )
            except (ValueError, KeyError, TypeError) as error:
                logger.warning("Skipping unreadable line %d in %s: %s", number, self.path, error)
        return entries

    def stats(self) -> Stats:
        entries = self.load()
        right = sum(entry.was_right for entry in entries)
        return Stats(right=right, wrong=len(entries) - right)

    def lessons(self, user_prompt: str = "", limit: int = DEFAULT_LESSON_LIMIT) -> list[dict]:
        """Past feedback as lessons: most relevant to `user_prompt` first, then mistakes, then newest."""
        query = _keywords(user_prompt)

        def rank(item: tuple[int, Feedback]) -> tuple[float, bool, int]:
            age, entry = item
            words = _keywords(f"{entry.user_prompt} {entry.instructions}")
            relevance = len(query & words) / len(query | words) if query and words else 0.0
            return relevance, not entry.was_right, age

        ranked = sorted(enumerate(self.load()), key=rank, reverse=True)
        return [entry.as_lesson() for _, entry in ranked[:limit]]


def _keywords(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in _STOPWORDS and len(word) > 1}
