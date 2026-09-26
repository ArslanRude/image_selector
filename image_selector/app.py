"""Gradio web UI: paste or upload 2+ images and a prompt, get back the image that matches best.

Run with `python -m image_selector.app`.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import gradio as gr
from openai import OpenAI
from typesafe_sdk import TypeSafeClient

from image_selector import decide, vision
from image_selector.config import ConfigError, Settings
from image_selector.decide import Verdict
from image_selector.memory import FeedbackMemory, Stats
from image_selector.vision import ImageDescription

MAX_PARALLEL_VISION_CALLS = 8

MET_SYMBOLS = {"yes": "✓", "partial": "~", "no": "✗"}

SUBMIT_LABEL = "Pick the best image"
BUSY_LABEL = "Selecting… please wait"

ProgressCallback = Callable[[float, str], None]


class JudgeError(RuntimeError):
    """A user-facing failure; the message is shown in the UI as-is."""


@dataclass(frozen=True)
class Judgment:
    checklist: tuple[str, ...]
    descriptions: tuple[ImageDescription, ...]
    verdict: Verdict


@dataclass(frozen=True)
class Run:
    """The latest verdict, kept so the user can rate it."""

    user_prompt: str
    descriptions: tuple[ImageDescription, ...]
    predicted_index: int
    instructions: str = ""


def judge(
    image_paths: Sequence[str],
    user_prompt: str,
    *,
    instructions: str = "",
    reference_paths: Sequence[str] = (),
    vision_client: OpenAI,
    vision_model: str,
    decision_client: TypeSafeClient,
    lessons: Sequence[dict] = (),
    on_progress: ProgressCallback = lambda fraction, message: None,
) -> Judgment:
    """Build a shared checklist, grade every image against it with DeepSeek, then have Jev pick the winner.

    `on_progress(fraction, message)` is called from the calling thread as each step starts or finishes.
    """
    user_prompt, instructions = user_prompt.strip(), instructions.strip()
    if not user_prompt:
        raise JudgeError("Enter a prompt to judge the images against.")
    if len(image_paths) < 2:
        raise JudgeError("Add at least two images.")
    # Read references once rather than once per candidate.
    references = [_read(path, f"reference image {number}") for number, path in enumerate(reference_paths, start=1)]

    on_progress(0.05, "Building a checklist from your prompt (DeepSeek)…")
    try:
        checklist = vision.build_checklist(user_prompt, instructions=instructions, references=references, client=vision_client, model=vision_model)
    except Exception as error:
        raise JudgeError(f"Building the checklist failed: {error}") from error

    def describe(path: str) -> ImageDescription:
        return vision.describe_image(
            path,
            user_prompt,
            checklist=checklist,
            instructions=instructions,
            references=references,
            client=vision_client,
            model=vision_model,
        )

    total = len(image_paths)
    on_progress(0.2, f"Grading {total} images against the checklist (DeepSeek)…")
    results: dict[int, ImageDescription] = {}
    with ThreadPoolExecutor(max_workers=min(total, MAX_PARALLEL_VISION_CALLS)) as pool:
        futures = {pool.submit(describe, path): index for index, path in enumerate(image_paths)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as error:
                pool.shutdown(wait=False, cancel_futures=True)
                raise JudgeError(f"Describing image {index + 1} failed: {error}") from error
            on_progress(0.2 + 0.6 * len(results) / total, f"Graded {len(results)} of {total} images…")
    descriptions = [results[index] for index in range(total)]

    on_progress(0.85, "Jev is choosing the best image…")
    try:
        verdict = decide.pick_winner(user_prompt, descriptions, client=decision_client, instructions=instructions, lessons=lessons)
    except Exception as error:
        raise JudgeError(f"Picking the winner failed: {error}") from error
    return Judgment(checklist=tuple(checklist), descriptions=tuple(descriptions), verdict=verdict)


def render(judgment: Judgment) -> str:
    """Format the verdict, the checklist, and every image's grades as Markdown."""
    verdict = judgment.verdict
    lines = [f"## Image {verdict.winner_index + 1} wins (confidence: {verdict.confidence:.0%})", ""]
    if not verdict.consistent:
        lines += ["> **Close call:** Jev's pick changed when the images were reordered. Double-check this one.", ""]
    lines += ["**Checklist**", *[f"{number}. {item}" for number, item in enumerate(judgment.checklist, start=1)], ""]

    for index, (description, probability) in enumerate(zip(judgment.descriptions, verdict.probabilities)):
        marker = " — winner" if index == verdict.winner_index else ""
        met = sum(check.met == "yes" for check in description.requirements)
        lines += [f"### Image {index + 1}: {probability:.0%}{marker} · {met}/{len(description.requirements)} requirements met"]
        lines += [f"- {MET_SYMBOLS.get(check.met, '?')} {check.requirement} — {check.evidence}" for check in description.requirements]
        if description.reference_match:
            lines.append(f"- **Reference match:** {description.reference_match}")
        lines += [
            f"- **Defects:** {description.defects}",
            f"- **Overall:** {description.prompt_match}",
            f"- **Composition:** {description.composition}",
        ]
        if description.notes:
            lines.append(f"- **Notes:** {description.notes}")
        lines.append("")
    return "\n".join(lines)


def number_captions(items: Sequence[tuple[str, str | None]] | None, prefix: str) -> list[tuple[str, str]] | None:
    """Caption each gallery image "<prefix> N" in order, or return None when the captions are already right."""
    items = list(items or [])
    captioned = [(path, f"{prefix} {number}") for number, (path, _caption) in enumerate(items, start=1)]
    return None if [caption for _, caption in items] == [caption for _, caption in captioned] else captioned


def chosen_label(verdict: Verdict) -> str:
    return f"Chosen: Image {verdict.winner_index + 1} ({verdict.confidence:.0%} confidence)"


def labelled_results(image_paths: Sequence[str], verdict: Verdict) -> list[tuple[str, str]]:
    """Every candidate captioned with its number and probability, with the chosen one marked."""
    return [
        (path, f"{'★ ' if index == verdict.winner_index else ''}Image {index + 1} ({probability:.0%})")
        for index, (path, probability) in enumerate(zip(image_paths, verdict.probabilities))
    ]


def render_stats(stats: Stats) -> str:
    if not stats.total:
        return "**Score:** no feedback yet. Rate a verdict to start teaching the picker."
    return f"**Score:** {stats.right} right, {stats.wrong} wrong · reward {stats.reward:+d} · accuracy {stats.accuracy:.0%}"


def save_feedback(memory: FeedbackMemory, run: Run | None, correct_index: int | None, reason: str = "") -> str:
    """Record the user's rating of `run` and describe the reward it earned."""
    if run is None:
        return "Nothing to rate yet. Pick the best image first."
    if correct_index is None:
        return "Select which image was actually best."
    feedback = memory.record(
        run.user_prompt,
        run.descriptions,
        run.predicted_index,
        correct_index,
        instructions=run.instructions,
        reason=reason,
    )
    if feedback.was_right:
        return f"Saved: reward +1. Image {correct_index + 1} was the right pick."
    return (
        f"Saved: reward -1. Remembered that Image {correct_index + 1} was right, not Image {run.predicted_index + 1}; "
        "future picks will learn from this."
    )


def _read(path: str, name: str) -> bytes:
    try:
        with open(path, "rb") as file:
            return file.read()
    except OSError as error:
        raise JudgeError(f"Could not read {name}: {error}") from error


def build_ui(settings: Settings) -> gr.Blocks:
    vision_client = vision.make_client(settings)
    decision_client = decide.make_client(settings)
    memory = FeedbackMemory(settings.memory_file)

    def start_run():
        return (
            gr.Button(value=BUSY_LABEL, interactive=False),
            gr.Image(value=None, visible=False),
            gr.Gallery(value=None, visible=False),
            "",  # left empty so the progress bar drawn over it is readable
            gr.Column(visible=False),
            "",
            None,
        )

    def on_submit(gallery, references_gallery, user_prompt: str, instructions: str, progress=gr.Progress()):
        image_paths = [path for path, _caption in gallery or []]
        try:
            judgment = judge(
                image_paths,
                user_prompt,
                instructions=instructions,
                reference_paths=[path for path, _caption in references_gallery or []],
                vision_client=vision_client,
                vision_model=settings.vision_model,
                decision_client=decision_client,
                lessons=memory.lessons(f"{user_prompt} {instructions}"),
                on_progress=lambda fraction, message: progress(fraction, desc=message),
            )
        except JudgeError as error:
            hidden = gr.Image(value=None, visible=False), gr.Gallery(value=None, visible=False)
            return *hidden, f"**Error:** {error}", gr.Column(visible=False), gr.Radio(), "", None
        verdict = judgment.verdict
        choices = [(f"Image {index + 1}", index) for index in range(len(judgment.descriptions))]
        run = Run(user_prompt.strip(), judgment.descriptions, verdict.winner_index, instructions.strip())
        return (
            gr.Image(value=image_paths[verdict.winner_index], label=chosen_label(verdict), visible=True),
            gr.Gallery(value=labelled_results(image_paths, verdict), visible=True),
            render(judgment),
            gr.Column(visible=True),
            gr.Radio(choices=choices, value=judgment.verdict.winner_index),
            "",
            run,
        )

    def on_save(correct_index: int | None, reason: str, run: Run | None):
        message = save_feedback(memory, run, correct_index, reason)
        saved = run is not None and correct_index is not None
        return (
            message,
            render_stats(memory.stats()),
            gr.Column(visible=not saved),
            None if saved else run,
            "" if saved else reason,
        )

    with gr.Blocks(title="Image Selector") as ui:
        gr.Markdown("# Image Selector\nPaste or upload two or more images, describe what you want, and see which image matches best. Images are numbered left to right.")
        stats = gr.Markdown(render_stats(memory.stats()))
        last_run = gr.State(None)
        with gr.Row():
            with gr.Column():
                gallery = gr.Gallery(
                    label="Candidate images",
                    type="filepath",
                    file_types=["image"],
                    sources=["upload", "clipboard"],
                    interactive=True,
                    columns=3,
                )
                with gr.Accordion("Reference images (optional, any number)", open=False):
                    references = gr.Gallery(
                        label="Reference images: examples of the look, subject, or style you want",
                        type="filepath",
                        file_types=["image"],
                        sources=["upload", "clipboard"],
                        interactive=True,
                        columns=4,
                    )
                prompt = gr.Textbox(label="Prompt", lines=3, placeholder="e.g. a red fox sitting in fresh snow at dusk")
                instructions = gr.Textbox(
                    label="Extra instructions (optional)",
                    lines=2,
                    placeholder="e.g. no text or watermarks; realistic anatomy matters most; prefer warm lighting",
                )
                submit = gr.Button(SUBMIT_LABEL, variant="primary")
            with gr.Column():
                chosen = gr.Image(label="Chosen image", type="filepath", interactive=False, visible=False, height=320)
                results_gallery = gr.Gallery(label="All candidates (★ = chosen)", interactive=False,visible=False, columns=4, height="auto")
                result = gr.Markdown(min_height=120)
                with gr.Column(visible=False) as feedback_panel:
                    correct = gr.Radio(label="Which image was actually best? Change it if the pick was wrong.")
                    reason = gr.Textbox(
                        label="Why? (optional, helps it learn)",
                        placeholder="e.g. image 1 has a sixth finger; I prefer softer colors",
                    )
                    save = gr.Button("Save feedback")
                feedback_status = gr.Markdown()
        # Lock the button and clear the last verdict first, run with a live progress bar, then unlock.
        submit.click(
            start_run,
            outputs=[submit, chosen, results_gallery, result, feedback_panel, feedback_status, last_run],
            show_progress="hidden",
            queue=False,
        ).then(
            on_submit,
            inputs=[gallery, references, prompt, instructions],
            outputs=[chosen, results_gallery, result, feedback_panel, correct, feedback_status, last_run],
            show_progress="full",
            show_progress_on=result,
        ).then(lambda: gr.Button(value=SUBMIT_LABEL, interactive=True), outputs=submit, show_progress="hidden", queue=False)
        # Number images as soon as they are added, pasted, or removed, so "Image N" in the verdict is easy to find.
        gallery.change(lambda items: number_captions(items, "Image") or gr.skip(), inputs=gallery, outputs=gallery)
        references.change(lambda items: number_captions(items, "Reference") or gr.skip(), inputs=references, outputs=references)
        save.click(on_save, inputs=[correct, reason, last_run], outputs=[feedback_status, stats, feedback_panel, last_run, reason])
        ui.load(lambda: render_stats(memory.stats()), outputs=stats)
    return ui


def main() -> None:
    try:
        settings = Settings.load()
    except ConfigError as error:
        raise SystemExit(str(error)) from error
    build_ui(settings).launch()


if __name__ == "__main__":
    main()
