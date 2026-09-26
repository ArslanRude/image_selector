"""Grade images against a prompt using DeepSeek's vision model (OpenAI-compatible API).

Grading happens in two steps so every candidate is judged on the same terms:
1. `build_checklist` turns the prompt, optional instructions, and optional reference images into a short
   list of concrete, visually checkable requirements.
2. `describe_image` grades one candidate against that checklist, alongside the reference images.
"""

import base64
import io
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from openai import OpenAI
from PIL import Image

from image_selector.config import Settings

ImageInput = str | Path | bytes

VERDICTS = ("yes", "partial", "no")
# DeepSeek downsizes every image to at most 384 tokens, so larger uploads only bloat the request (48 MiB cap).
# Files above this size are shrunk to fit MAX_IMAGE_SIDE before sending, which keeps many references practical.
MAX_UNSCALED_BYTES = 1_000_000
MAX_IMAGE_SIDE = 1024
# One retry covers the occasional reply that wraps the JSON in prose or truncates it.
JSON_ATTEMPTS = 2

CHECKLIST_PROMPT = """You are preparing to judge candidate images against a request.

Request: {user_prompt}
{instructions_block}{references_block}
Break the request into 3 to 8 concrete requirements that can be checked by looking at an image: subject,
attributes, count, setting, lighting, style, composition, text, and anything the instructions demand.
{reference_rule}Order them from most to least important. Do not add requirements the request does not imply.

Reply with ONLY a JSON object: {{"requirements": ["...", "..."]}}"""

DESCRIBE_PROMPT = """You are grading the CANDIDATE image against a request.

Request: {user_prompt}
{instructions_block}
Checklist:
{checklist}
{references_block}
Reply with ONLY a JSON object, no other text, with these keys:
- "requirements": one object per checklist item, in the same order:
  {{"requirement": "<checklist text>", "met": "yes" | "partial" | "no", "evidence": "<what in the image shows it>"}}
  Be strict: "yes" only when clearly visible and correct.
{reference_key}- "defects": technical problems (blur, artifacts, distorted anatomy or hands, garbled text, bad crops), or "none".
- "composition": subject, framing, layout, colors, style, in one or two sentences.
- "prompt_match": a one-sentence overall verdict on how well the image fits the request.
- "notes": anything else that matters for the request, or ""."""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

# Formats DeepSeek accepts, detected from file signatures rather than trusting file extensions.
_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)

T = TypeVar("T")


class VisionError(RuntimeError):
    """The image could not be read or DeepSeek's reply could not be used."""


@dataclass(frozen=True)
class RequirementCheck:
    requirement: str
    met: str
    """One of `VERDICTS`, or "unclear" when the model answered something else."""
    evidence: str


@dataclass(frozen=True)
class ImageDescription:
    composition: str
    defects: str
    prompt_match: str
    notes: str
    requirements: tuple[RequirementCheck, ...] = ()
    reference_match: str = ""
    """How closely the image matches the reference images; empty when none were given."""

    def as_dict(self) -> dict:
        """JSON-ready form, omitting fields that carry no information."""
        data = asdict(self)
        data["requirements"] = list(data["requirements"])
        return {key: value for key, value in data.items() if value not in ("", [])}

    @classmethod
    def from_dict(cls, data: dict) -> "ImageDescription":
        """Inverse of `as_dict`; also reads descriptions saved before checklists existed."""
        return cls(
            composition=data["composition"],
            defects=data["defects"],
            prompt_match=data["prompt_match"],
            notes=data.get("notes", ""),
            requirements=tuple(RequirementCheck(**check) for check in data.get("requirements", ())),
            reference_match=data.get("reference_match", ""),
        )


def make_client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)


def build_checklist(
    user_prompt: str,
    *,
    instructions: str = "",
    references: Sequence[ImageInput] = (),
    client: OpenAI,
    model: str,
) -> list[str]:
    """Turn the request into the shared requirements every candidate is graded on."""
    text = CHECKLIST_PROMPT.format(
        user_prompt=user_prompt,
        instructions_block=_instructions_block(instructions),
        references_block=f"\n{len(references)} reference image(s) follow; they show what the user wants.\n" if references else "",
        reference_rule="Include requirements for what the reference images establish (style, subject identity, palette, layout).\n" if references else "",
    )
    content = [{"type": "text", "text": text}, *_labelled_images("Reference image", references)]
    return _ask_json(client, model, content, parse_checklist)


def describe_image(
    image: ImageInput,
    user_prompt: str,
    *,
    checklist: Sequence[str],
    instructions: str = "",
    references: Sequence[ImageInput] = (),
    client: OpenAI,
    model: str,
) -> ImageDescription:
    """Grade one candidate image against the shared checklist and any reference images."""
    text = DESCRIBE_PROMPT.format(
        user_prompt=user_prompt,
        instructions_block=_instructions_block(instructions),
        checklist="\n".join(f"{number}. {item}" for number, item in enumerate(checklist, start=1)),
        references_block=f"\nThe first {len(references)} image(s) are REFERENCES showing what the user wants. The last image is the CANDIDATE to grade.\n"
        if references
        else "\nThe image below is the CANDIDATE to grade.\n",
        reference_key='- "reference_match": how closely the candidate matches the references in subject, style, palette, and layout, naming the differences.\n'
        if references
        else "",
    )
    content = [
        {"type": "text", "text": text},
        *_labelled_images("Reference image", references),
        {"type": "text", "text": "CANDIDATE image:"},
        _image_part(image),
    ]
    return _ask_json(client, model, content, parse_description)


def parse_checklist(text: str) -> list[str]:
    payload = _load_json_object(text)
    items = payload.get("requirements")
    if not isinstance(items, list) or not items:
        raise VisionError(f"Checklist reply has no requirements: {text[:200]!r}")
    return [str(item).strip() for item in items if str(item).strip()]


def parse_description(text: str) -> ImageDescription:
    """Parse the model's JSON reply, tolerating a surrounding Markdown code fence."""
    payload = _load_json_object(text)
    missing = [key for key in ("composition", "defects", "prompt_match") if key not in payload]
    if missing:
        raise VisionError(f"Vision model reply is missing key(s): {', '.join(missing)}")

    raw_checks = payload.get("requirements") or []
    if not isinstance(raw_checks, list):
        raise VisionError(f"Vision model returned requirements that are not a list: {text[:200]!r}")
    checks = []
    for raw in raw_checks:
        if not isinstance(raw, dict):
            continue
        met = str(raw.get("met", "")).strip().lower()
        checks.append(
            RequirementCheck(
                requirement=str(raw.get("requirement", "")).strip(),
                met=met if met in VERDICTS else "unclear",
                evidence=str(raw.get("evidence", "")).strip(),
            )
        )
    return ImageDescription(
        composition=str(payload["composition"]).strip(),
        defects=str(payload["defects"]).strip(),
        prompt_match=str(payload["prompt_match"]).strip(),
        notes=str(payload.get("notes", "")).strip(),
        requirements=tuple(checks),
        reference_match=str(payload.get("reference_match", "")).strip(),
    )


def _ask_json(client: OpenAI, model: str, content: list[dict], parse: Callable[[str], T]) -> T:
    for attempt in range(JSON_ATTEMPTS):
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": content}],
        )
        try:
            return parse(response.choices[0].message.content or "")
        except VisionError:
            if attempt == JSON_ATTEMPTS - 1:
                raise
    raise AssertionError("unreachable")


def _load_json_object(text: str) -> dict:
    try:
        payload = json.loads(_FENCE.sub("", text.strip()))
    except json.JSONDecodeError as error:
        raise VisionError(f"Vision model did not return valid JSON: {text[:200]!r}") from error
    if not isinstance(payload, dict):
        raise VisionError(f"Vision model returned JSON that is not an object: {text[:200]!r}")
    return payload


def _instructions_block(instructions: str) -> str:
    return f"User instructions (follow them strictly): {instructions.strip()}\n" if instructions.strip() else ""


def _labelled_images(label: str, images: Sequence[ImageInput]) -> list[dict]:
    parts = []
    for number, image in enumerate(images, start=1):
        parts += [{"type": "text", "text": f"{label} {number}:"}, _image_part(image)]
    return parts


def _image_part(image: ImageInput) -> dict:
    data, mime = shrink(image if isinstance(image, bytes) else Path(image).read_bytes())
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"}}


def shrink(data: bytes) -> tuple[bytes, str]:
    """Return the image and its MIME type, re-encoded at most MAX_IMAGE_SIDE pixels wide or tall if the file is large."""
    mime = _mime_type(data)
    if len(data) <= MAX_UNSCALED_BYTES:
        return data, mime
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            output = io.BytesIO()
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                image.convert("RGBA").save(output, format="PNG", optimize=True)
                return output.getvalue(), "image/png"
            image.convert("RGB").save(output, format="JPEG", quality=90)
            return output.getvalue(), "image/jpeg"
    except OSError as error:
        raise VisionError(f"Could not read image: {error}") from error


def _mime_type(data: bytes) -> str:
    for signature, mime in _SIGNATURES:
        if data.startswith(signature):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise VisionError("Unsupported image format; use JPEG, PNG, GIF, or WebP.")
