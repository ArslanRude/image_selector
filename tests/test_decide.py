import json
from types import SimpleNamespace

import httpx2
import pytest
from typesafe_sdk import Choice, ChoiceAnswer, TypeSafeClient

from image_selector.decide import QUESTION, DecisionError, Verdict, pick_winner
from image_selector.vision import ImageDescription, RequirementCheck


def describe(n):
    return ImageDescription(
        composition=f"composition {n}",
        defects=f"defects {n}",
        prompt_match=f"match {n}",
        notes=f"notes {n}",
        requirements=(RequirementCheck("a fox", "yes", f"evidence {n}"),),
    )


def response(choice, probabilities):
    answer = ChoiceAnswer(choice=choice, confidence=probabilities[choice], probabilities=probabilities)
    return SimpleNamespace(choices={QUESTION: answer})


def fake_client(mocker, *responses):
    client = mocker.Mock()
    client.system_one.side_effect = list(responses)
    return client


def test_pick_winner_builds_one_criterion_per_image_in_both_orders(mocker):
    descriptions = [describe(n) for n in range(1, 4)]
    probabilities = {"image_1": 0.2, "image_2": 0.7, "image_3": 0.1}
    client = fake_client(mocker, response("image_2", probabilities), response("image_2", probabilities))

    pick_winner("a fox", descriptions, client=client)

    forward, backward = (call.kwargs for call in client.system_one.call_args_list)
    assert forward["state"] == backward["state"] == {"user_prompt": "a fox"}
    question = forward["questions"][QUESTION]
    assert isinstance(question, Choice)
    assert question.criteria == {f"image_{n}": describe(n).as_dict() for n in range(1, 4)}
    assert list(question.criteria) == ["image_1", "image_2", "image_3"]
    assert list(backward["questions"][QUESTION].criteria) == ["image_3", "image_2", "image_1"]


def test_pick_winner_averages_orderings(mocker):
    client = fake_client(
        mocker,
        response("image_3", {"image_1": 0.1, "image_2": 0.3, "image_3": 0.6}),
        response("image_3", {"image_1": 0.1, "image_2": 0.1, "image_3": 0.8}),
    )

    verdict = pick_winner("a fox", [describe(n) for n in range(1, 4)], client=client)

    assert verdict == Verdict(winner_index=2, confidence=pytest.approx(0.7), probabilities=pytest.approx((0.1, 0.2, 0.7)), consistent=True)


def test_pick_winner_flags_disagreement_between_orderings(mocker):
    client = fake_client(
        mocker,
        response("image_1", {"image_1": 0.55, "image_2": 0.45}),
        response("image_2", {"image_1": 0.35, "image_2": 0.65}),
    )

    verdict = pick_winner("a fox", [describe(1), describe(2)], client=client)

    assert (verdict.winner_index, verdict.consistent) == (1, False)
    assert verdict.confidence == pytest.approx(0.55)


def test_pick_winner_without_debias_asks_once(mocker):
    client = fake_client(mocker, response("image_1", {"image_1": 0.6, "image_2": 0.4}))
    verdict = pick_winner("a fox", [describe(1), describe(2)], client=client, debias=False)
    assert client.system_one.call_count == 1
    assert (verdict.winner_index, verdict.confidence) == (0, 0.6)


def test_pick_winner_sends_instructions_and_past_feedback(mocker):
    lessons = [{"prompt": "an owl", "user_preferred": describe(1).as_dict(), "user_rejected": [], "earlier_pick_was_right": False}]
    probabilities = {"image_1": 0.6, "image_2": 0.4}
    client = fake_client(mocker, response("image_1", probabilities), response("image_1", probabilities))

    pick_winner("a fox", [describe(1), describe(2)], client=client, instructions=" no text ", lessons=lessons)

    assert client.system_one.call_args.kwargs["state"] == {"user_prompt": "a fox", "user_instructions": "no text", "past_feedback": lessons}


def test_pick_winner_round_trips_through_real_sdk_client():
    """Exercise the SDK's actual request encoding and response decoding, with no network."""
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        answer = {"type": "choice", "choice": "image_1", "confidence": 0.9, "probabilities": {"image_1": 0.9, "image_2": 0.1}}
        return httpx2.Response(200, json={"model": "jev-latest", "usage": {}, "answers": {QUESTION: answer}})

    with TypeSafeClient(api_key="test-key", transport=httpx2.MockTransport(handler)) as client:
        verdict = pick_winner("a fox", [describe(1), describe(2)], client=client)

    assert verdict == Verdict(winner_index=0, confidence=0.9, probabilities=(0.9, 0.1), consistent=True)
    assert len(requests) == 2
    body = requests[0]
    assert body["state"] == {"user_prompt": "a fox"}
    assert body["questions"][QUESTION]["type"] == "choice"
    assert body["questions"][QUESTION]["criteria"]["image_2"] == describe(2).as_dict()


def test_pick_winner_requires_two_images(mocker):
    client = mocker.Mock()
    with pytest.raises(ValueError, match="two images"):
        pick_winner("a fox", [describe(1)], client=client)
    client.system_one.assert_not_called()


def test_pick_winner_rejects_unknown_label(mocker):
    client = fake_client(mocker, response("image_9", {"image_9": 0.9}))
    with pytest.raises(DecisionError, match="image_9"):
        pick_winner("a fox", [describe(1), describe(2)], client=client)
