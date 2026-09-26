import pytest

from image_selector.memory import FeedbackMemory, Stats
from image_selector.vision import ImageDescription


def describe(n):
    return ImageDescription(composition=f"composition {n}", defects="none", prompt_match=f"match {n}", notes="")


DESCRIPTIONS = [describe(1), describe(2), describe(3)]


@pytest.fixture
def memory(tmp_path):
    return FeedbackMemory(tmp_path / "memory.jsonl")


def test_empty_memory(memory):
    assert memory.load() == []
    assert memory.stats() == Stats()
    assert memory.stats().accuracy is None
    assert memory.lessons() == []


def test_reward_is_positive_for_right_pick_and_negative_for_wrong(memory):
    assert memory.record("fox", DESCRIPTIONS, predicted_index=1, correct_index=1).reward == 1
    assert memory.record("fox", DESCRIPTIONS, predicted_index=0, correct_index=2).reward == -1


def test_feedback_persists_across_instances(memory):
    memory.record("fox", DESCRIPTIONS, predicted_index=0, correct_index=2)

    [entry] = FeedbackMemory(memory.path).load()
    assert (entry.user_prompt, entry.descriptions, entry.predicted_index, entry.correct_index) == ("fox", tuple(DESCRIPTIONS), 0, 2)


def test_stats_totals_reward_and_accuracy(memory):
    for predicted, correct in [(0, 0), (1, 1), (2, 1), (0, 0)]:
        memory.record("fox", DESCRIPTIONS, predicted, correct)

    stats = memory.stats()
    assert (stats.right, stats.wrong, stats.reward, stats.accuracy) == (3, 1, 2, 0.75)


def test_lesson_names_preferred_and_rejected_images(memory):
    memory.record("fox", DESCRIPTIONS, predicted_index=0, correct_index=2)

    assert memory.lessons() == [
        {
            "prompt": "fox",
            "user_preferred": describe(3).as_dict(),
            "user_rejected": [describe(1).as_dict(), describe(2).as_dict()],
            "earlier_pick_was_right": False,
        }
    ]


def test_lessons_put_recent_mistakes_first_and_respect_limit(memory):
    memory.record("old mistake", DESCRIPTIONS, 0, 1)
    memory.record("confirmed", DESCRIPTIONS, 0, 0)
    memory.record("new mistake", DESCRIPTIONS, 2, 0)

    assert [lesson["prompt"] for lesson in memory.lessons()] == ["new mistake", "old mistake", "confirmed"]
    assert [lesson["prompt"] for lesson in memory.lessons(limit=2)] == ["new mistake", "old mistake"]


def test_record_rejects_out_of_range_answer(memory):
    with pytest.raises(ValueError, match="out of range"):
        memory.record("fox", DESCRIPTIONS, predicted_index=0, correct_index=3)
    assert memory.load() == []


def test_load_skips_corrupt_lines(memory):
    memory.record("fox", DESCRIPTIONS, 0, 0)
    with memory.path.open("a", encoding="utf-8") as file:
        file.write("{not json\n")
    memory.record("owl", DESCRIPTIONS, 1, 1)

    assert [entry.user_prompt for entry in memory.load()] == ["fox", "owl"]


def test_lessons_rank_related_requests_first(memory):
    memory.record("red sports car on a racetrack", DESCRIPTIONS, 0, 1)
    memory.record("watercolor fox in the snow", DESCRIPTIONS, 0, 0)
    memory.record("portrait of an old man", DESCRIPTIONS, 2, 1)

    prompts = [lesson["prompt"] for lesson in memory.lessons("a fox sleeping in snow")]
    assert prompts[0] == "watercolor fox in the snow"
    # Unrelated requests keep the mistakes-first, newest-first order behind it.
    assert prompts[1:] == ["portrait of an old man", "red sports car on a racetrack"]


def test_reason_and_instructions_persist_into_lessons(memory):
    memory.record("fox", DESCRIPTIONS, 0, 1, instructions=" no text ", reason=" image 1 has a watermark ")

    [entry] = FeedbackMemory(memory.path).load()
    assert (entry.instructions, entry.reason) == ("no text", "image 1 has a watermark")
    [lesson] = memory.lessons()
    assert (lesson["instructions"], lesson["user_reason"]) == ("no text", "image 1 has a watermark")


def test_lessons_omit_empty_reason_and_instructions(memory):
    memory.record("fox", DESCRIPTIONS, 0, 1)
    assert {"instructions", "user_reason"}.isdisjoint(memory.lessons()[0])


def test_load_reads_lines_saved_before_checklists_and_reasons(memory):
    old = '{"user_prompt": "fox", "descriptions": [{"composition": "c", "defects": "d", "prompt_match": "p", "notes": "n"}, ' \
        '{"composition": "c2", "defects": "d2", "prompt_match": "p2", "notes": "n2"}], "predicted_index": 0, "correct_index": 1, ' \
        '"reward": -1, "created_at": "2026-09-25T10:00:00+00:00"}\n'
    memory.path.write_text(old, encoding="utf-8")

    [entry] = memory.load()
    assert (entry.correct_index, entry.reason, entry.descriptions[1].prompt_match) == (1, "", "p2")
