"""Pick the image that best matches a prompt from its graded descriptions, using Jev's `Choice` primitive."""

from collections.abc import Sequence
from dataclasses import dataclass

from typesafe_sdk import Choice, TypeSafeClient

from image_selector.config import Settings
from image_selector.vision import ImageDescription

QUESTION = "winner"

INSTRUCTIONS = (
    "Each option is one candidate image, graded by a vision model against a shared checklist derived from "
    "the user's request. Choose the image that best satisfies the request. Weigh evidence in this order: "
    "(1) the checklist, where earlier requirements matter more and a 'no' on an important requirement "
    "outweighs minor defects; (2) reference_match, when present, since references show what the user "
    "wants; (3) defects and overall quality. Follow user_instructions strictly when present. If "
    "past_feedback is present, it records which images this user preferred and rejected for earlier "
    "requests, why when they said, and whether your earlier pick was right; learn the user's taste from "
    "it, especially from the picks they corrected, and apply it where the requests are similar."
)


class DecisionError(RuntimeError):
    """Jev returned an answer that does not correspond to a submitted image."""


@dataclass(frozen=True)
class Verdict:
    winner_index: int
    """Zero-based index of the winning image."""
    confidence: float
    """Probability that the winner is the best match, from 0 to 1, averaged across orderings."""
    probabilities: tuple[float, ...]
    """Probability that each image is the best match, in submission order."""
    consistent: bool = True
    """False when Jev picked a different image after the options were reordered, a sign of a close call."""


def make_client(settings: Settings) -> TypeSafeClient:
    return TypeSafeClient(api_key=settings.typesafe_api_key, model=settings.decision_model)


def label(index: int) -> str:
    return f"image_{index + 1}"


def pick_winner(
    user_prompt: str,
    descriptions: Sequence[ImageDescription],
    *,
    client: TypeSafeClient,
    instructions: str = "",
    lessons: Sequence[dict] = (),
    debias: bool = True,
) -> Verdict:
    """Ask Jev which described image best matches `user_prompt`.

    With `debias`, Jev decides twice, once with the options reversed, and the probabilities are averaged
    so the verdict doesn't depend on which image happened to be listed first.
    """
    if len(descriptions) < 2:
        raise ValueError("At least two images are required to pick a winner.")

    labels = [label(index) for index in range(len(descriptions))]
    criteria = {name: description.as_dict() for name, description in zip(labels, descriptions)}
    state = {"user_prompt": user_prompt}
    if instructions.strip():
        state["user_instructions"] = instructions.strip()
    if lessons:
        state["past_feedback"] = list(lessons)

    orderings = [labels, labels[::-1]] if debias else [labels]
    runs = [_ask(client, state, {name: criteria[name] for name in order}, labels) for order in orderings]

    probabilities = tuple(sum(run_probabilities[name] for _, run_probabilities in runs) / len(runs) for name in labels)
    picks = {choice for choice, _ in runs}
    if len(picks) == 1:
        # Agreement: keep Jev's own pick even if averaging left a tie.
        winner_index = labels.index(picks.pop())
        consistent = True
    else:
        winner_index = max(range(len(labels)), key=lambda index: probabilities[index])
        consistent = False
    return Verdict(
        winner_index=winner_index,
        confidence=probabilities[winner_index],
        probabilities=probabilities,
        consistent=consistent,
    )


def _ask(client: TypeSafeClient, state: dict, criteria: dict, labels: list[str]) -> tuple[str, dict[str, float]]:
    """One Jev decision: the chosen label and the probability of every label."""
    response = client.system_one(state=state, questions={QUESTION: Choice(instructions=INSTRUCTIONS, criteria=criteria)})
    answer = response.choices.get(QUESTION)
    if answer is None:
        raise DecisionError(f"Jev response has no {QUESTION!r} answer.")
    if answer.choice not in labels:
        raise DecisionError(f"Jev chose {answer.choice!r}, which is not one of {labels}.")
    return answer.choice, {name: answer.probabilities.get(name, 0.0) for name in labels}
