import pytest

from image_selector import app
from image_selector.app import (
    Judgment,
    JudgeError,
    Run,
    chosen_label,
    judge,
    labelled_results,
    number_captions,
    render,
    render_stats,
    save_feedback,
)
from image_selector.decide import Verdict
from image_selector.memory import FeedbackMemory, Stats
from image_selector.vision import ImageDescription, RequirementCheck, VisionError

CHECKLIST = ["a fox", "snow"]
DESCRIPTION = ImageDescription(
    composition="c",
    defects="none",
    prompt_match="good",
    notes="",
    requirements=(RequirementCheck("a fox", "yes", "fox in center"), RequirementCheck("snow", "no", "grass")),
)
VERDICT = Verdict(winner_index=1, confidence=0.8, probabilities=(0.2, 0.8))


@pytest.fixture
def stubs(mocker):
    return (
        mocker.patch.object(app.vision, "build_checklist", return_value=CHECKLIST),
        mocker.patch.object(app.vision, "describe_image", return_value=DESCRIPTION),
        mocker.patch.object(app.decide, "pick_winner", return_value=VERDICT),
    )


def run(image_paths, prompt="a fox", **kwargs):
    return judge(image_paths, prompt, vision_client=object(), vision_model="m", decision_client=object(), **kwargs)


def test_judge_builds_checklist_grades_each_image_then_picks_winner(stubs, tmp_path):
    build_checklist, describe, pick = stubs
    reference = tmp_path / "ref.png"
    reference.write_bytes(b"reference bytes")
    lessons = [{"prompt": "an owl"}]

    judgment = run(["a.png", "b.png"], " a fox ", instructions=" no text ", reference_paths=[str(reference)], lessons=lessons)

    assert judgment == Judgment(checklist=tuple(CHECKLIST), descriptions=(DESCRIPTION, DESCRIPTION), verdict=VERDICT)
    assert build_checklist.call_args.args == ("a fox",)
    assert build_checklist.call_args.kwargs["instructions"] == "no text"
    assert build_checklist.call_args.kwargs["references"] == [b"reference bytes"]
    assert sorted(call.args[0] for call in describe.call_args_list) == ["a.png", "b.png"]
    for call in describe.call_args_list:
        assert call.kwargs["checklist"] == CHECKLIST
        assert call.kwargs["references"] == [b"reference bytes"]
        assert call.kwargs["instructions"] == "no text"
    assert pick.call_args.args == ("a fox", [DESCRIPTION, DESCRIPTION])
    assert pick.call_args.kwargs["instructions"] == "no text"
    assert pick.call_args.kwargs["lessons"] == lessons


def test_judge_names_the_image_that_failed(stubs):
    _, describe, pick = stubs

    def fail_on_b(path, *_args, **_kwargs):
        if path == "b.png":
            raise VisionError("Unsupported image format")
        return DESCRIPTION

    describe.side_effect = fail_on_b
    with pytest.raises(JudgeError, match="image 2 failed: Unsupported image format"):
        run(["a.png", "b.png"])
    pick.assert_not_called()


def test_judge_surfaces_checklist_errors(stubs):
    build_checklist, describe, _ = stubs
    build_checklist.side_effect = VisionError("did not return valid JSON")
    with pytest.raises(JudgeError, match="Building the checklist failed: did not return valid JSON"):
        run(["a.png", "b.png"])
    describe.assert_not_called()


def test_judge_surfaces_decision_errors(stubs):
    stubs[2].side_effect = RuntimeError("401 invalid key")
    with pytest.raises(JudgeError, match="Picking the winner failed: 401 invalid key"):
        run(["a.png", "b.png"])


def test_judge_accepts_any_number_of_references(stubs, tmp_path):
    build_checklist, describe, _ = stubs
    paths = []
    for number in range(9):
        path = tmp_path / f"ref{number}.png"
        path.write_bytes(f"reference {number}".encode())
        paths.append(str(path))

    run(["a.png", "b.png"], reference_paths=paths)

    expected = [f"reference {number}".encode() for number in range(9)]
    assert build_checklist.call_args.kwargs["references"] == expected
    assert all(call.kwargs["references"] == expected for call in describe.call_args_list)


def test_judge_reports_unreadable_reference(stubs, tmp_path):
    with pytest.raises(JudgeError, match="Could not read reference image 1"):
        run(["a.png", "b.png"], reference_paths=[str(tmp_path / "missing.png")])
    stubs[0].assert_not_called()


@pytest.mark.parametrize(
    ("paths", "prompt", "references", "message"),
    [
        (["a.png"], "a fox", [], "two images"),
        (["a.png", "b.png"], "  ", [], "prompt"),
    ],
)
def test_judge_validates_input_before_calling_apis(stubs, paths, prompt, references, message):
    with pytest.raises(JudgeError, match=message):
        run(paths, prompt, reference_paths=references)
    stubs[0].assert_not_called()


def test_save_feedback_rewards_right_pick(tmp_path):
    memory = FeedbackMemory(tmp_path / "memory.jsonl")
    message = save_feedback(memory, Run("a fox", (DESCRIPTION, DESCRIPTION), predicted_index=1), correct_index=1)
    assert message.startswith("Saved: reward +1")
    assert memory.stats().reward == 1


def test_save_feedback_penalizes_and_remembers_wrong_pick_with_reason(tmp_path):
    memory = FeedbackMemory(tmp_path / "memory.jsonl")
    run_ = Run("a fox", (DESCRIPTION, DESCRIPTION), predicted_index=1, instructions="no text")
    message = save_feedback(memory, run_, correct_index=0, reason="image 2 has a watermark")
    assert "reward -1" in message and "Image 1 was right, not Image 2" in message
    [lesson] = memory.lessons()
    assert (lesson["earlier_pick_was_right"], lesson["user_reason"], lesson["instructions"]) == (False, "image 2 has a watermark", "no text")


@pytest.mark.parametrize(("run_", "correct_index"), [(None, 0), (Run("a fox", (DESCRIPTION, DESCRIPTION), 0), None)])
def test_save_feedback_records_nothing_without_a_verdict_and_answer(tmp_path, run_, correct_index):
    memory = FeedbackMemory(tmp_path / "memory.jsonl")
    save_feedback(memory, run_, correct_index)
    assert memory.load() == []


def test_render_stats():
    assert "no feedback yet" in render_stats(Stats())
    assert render_stats(Stats(right=3, wrong=1)) == "**Score:** 3 right, 1 wrong · reward +2 · accuracy 75%"


def test_render_shows_checklist_grades_and_winner():
    text = render(Judgment(tuple(CHECKLIST), (DESCRIPTION, DESCRIPTION), Verdict(0, 0.75, (0.75, 0.25))))
    assert text.startswith("## Image 1 wins (confidence: 75%)")
    assert "Close call" not in text
    assert "1. a fox\n2. snow" in text
    assert "### Image 1: 75% — winner · 1/2 requirements met" in text
    assert "- ✓ a fox — fox in center" in text and "- ✗ snow — grass" in text


def test_render_flags_close_call_and_reference_match():
    description = ImageDescription("c", "none", "good", "", reference_match="same palette as the reference")
    text = render(Judgment(tuple(CHECKLIST), (description, description), Verdict(1, 0.52, (0.48, 0.52), consistent=False)))
    assert "**Close call:**" in text
    assert "- **Reference match:** same palette as the reference" in text


def test_number_captions_labels_images_in_order():
    assert number_captions([("a.png", None), ("b.png", "old")], "Image") == [("a.png", "Image 1"), ("b.png", "Image 2")]


def test_number_captions_renumbers_after_removal():
    assert number_captions([("b.png", "Image 2"), ("c.png", "Image 3")], "Image") == [("b.png", "Image 1"), ("c.png", "Image 2")]


@pytest.mark.parametrize("items", [None, [], [("a.png", "Reference 1"), ("b.png", "Reference 2")]])
def test_number_captions_returns_none_when_already_right(items):
    assert number_captions(items, "Reference") is None


def test_labelled_results_marks_the_chosen_image():
    verdict = Verdict(winner_index=1, confidence=0.64, probabilities=(0.3, 0.64, 0.06))
    assert chosen_label(verdict) == "Chosen: Image 2 (64% confidence)"
    assert labelled_results(["a.png", "b.png", "c.png"], verdict) == [
        ("a.png", "Image 1 (30%)"),
        ("b.png", "★ Image 2 (64%)"),
        ("c.png", "Image 3 (6%)"),
    ]


def test_judge_reports_progress_for_each_step(stubs):
    updates = []
    run(["a.png", "b.png"], on_progress=lambda fraction, message: updates.append((fraction, message)))

    assert [message for _, message in updates] == [
        "Building a checklist from your prompt (DeepSeek)…",
        "Grading 2 images against the checklist (DeepSeek)…",
        "Graded 1 of 2 images…",
        "Graded 2 of 2 images…",
        "Jev is choosing the best image…",
    ]
    fractions = [fraction for fraction, _ in updates]
    assert fractions == sorted(fractions) and 0 < fractions[0] and fractions[-1] < 1
