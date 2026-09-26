import base64
import io
import json
import os
from types import SimpleNamespace

import pytest
from PIL import Image

from image_selector.vision import (
    MAX_IMAGE_SIDE,
    MAX_UNSCALED_BYTES,
    ImageDescription,
    RequirementCheck,
    VisionError,
    build_checklist,
    describe_image,
    parse_description,
    shrink,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 8

REPLY = {
    "requirements": [
        {"requirement": "a fox", "met": "yes", "evidence": "fox in center"},
        {"requirement": "dusk lighting", "met": "No", "evidence": "bright daylight"},
        {"requirement": "snow", "met": "mostly", "evidence": "patchy snow"},
    ],
    "defects": "none",
    "composition": "A fox centered in snow.",
    "prompt_match": "Right subject, wrong time of day.",
    "notes": "",
}
DESCRIPTION = ImageDescription(
    composition="A fox centered in snow.",
    defects="none",
    prompt_match="Right subject, wrong time of day.",
    notes="",
    requirements=(
        RequirementCheck("a fox", "yes", "fox in center"),
        RequirementCheck("dusk lighting", "no", "bright daylight"),
        RequirementCheck("snow", "unclear", "patchy snow"),
    ),
)
CHECKLIST = ["a fox", "dusk lighting", "snow"]


def fake_client(mocker, *contents):
    client = mocker.Mock()
    client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))]) for content in contents
    ]
    return client


def sent_content(client, call=0):
    return client.chat.completions.create.call_args_list[call].kwargs["messages"][0]["content"]


def data_url(data, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def test_build_checklist_includes_instructions_and_labelled_references(mocker):
    client = fake_client(mocker, json.dumps({"requirements": CHECKLIST}))

    checklist = build_checklist("a fox at dusk", instructions="no watermarks", references=[PNG, JPEG], client=client, model="deepseek-flash")

    assert checklist == CHECKLIST
    kwargs = client.chat.completions.create.call_args.kwargs
    assert (kwargs["model"], kwargs["temperature"]) == ("deepseek-flash", 0)
    text, label_1, image_1, label_2, image_2 = sent_content(client)
    assert "a fox at dusk" in text["text"] and "no watermarks" in text["text"] and "reference images" in text["text"]
    assert (label_1["text"], image_1["image_url"]["url"]) == ("Reference image 1:", data_url(PNG))
    assert (label_2["text"], image_2["image_url"]["url"]) == ("Reference image 2:", data_url(JPEG, "image/jpeg"))


def test_build_checklist_without_extras_sends_text_only(mocker):
    client = fake_client(mocker, json.dumps({"requirements": CHECKLIST}))
    build_checklist("a fox", client=client, model="m")
    [text] = sent_content(client)
    assert "User instructions" not in text["text"] and "reference" not in text["text"].lower()


def test_describe_image_grades_candidate_after_references(mocker):
    client = fake_client(mocker, json.dumps(REPLY))

    description = describe_image(PNG, "a fox at dusk", checklist=CHECKLIST, instructions="no watermarks", references=[JPEG], client=client, model="m")

    assert description == DESCRIPTION
    text, ref_label, ref_image, candidate_label, candidate = sent_content(client)
    assert "1. a fox\n2. dusk lighting\n3. snow" in text["text"]
    assert "no watermarks" in text["text"] and '"reference_match"' in text["text"]
    assert (ref_label["text"], ref_image["image_url"]["url"]) == ("Reference image 1:", data_url(JPEG, "image/jpeg"))
    assert (candidate_label["text"], candidate["image_url"]["url"]) == ("CANDIDATE image:", data_url(PNG))


def test_describe_image_without_references_asks_for_no_reference_match(mocker):
    client = fake_client(mocker, json.dumps(REPLY))
    describe_image(PNG, "a fox", checklist=CHECKLIST, client=client, model="m")
    text, candidate_label, candidate = sent_content(client)
    assert "reference_match" not in text["text"]


@pytest.mark.parametrize(("data", "mime"), [(JPEG, "image/jpeg"), (WEBP, "image/webp"), (b"GIF89a" + b"\x00" * 8, "image/gif")])
def test_describe_image_detects_mime_type(mocker, data, mime):
    client = fake_client(mocker, json.dumps(REPLY))
    describe_image(data, "prompt", checklist=CHECKLIST, client=client, model="m")
    assert sent_content(client)[-1]["image_url"]["url"].startswith(f"data:{mime};base64,")


def test_describe_image_reads_file_path(mocker, tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(PNG)
    client = fake_client(mocker, json.dumps(REPLY))
    assert describe_image(str(path), "prompt", checklist=CHECKLIST, client=client, model="m") == DESCRIPTION


def test_describe_image_retries_once_on_invalid_json(mocker):
    client = fake_client(mocker, "Sure! This image shows a fox.", json.dumps(REPLY))
    assert describe_image(PNG, "a fox", checklist=CHECKLIST, client=client, model="m") == DESCRIPTION
    assert client.chat.completions.create.call_count == 2


def test_describe_image_gives_up_after_second_invalid_reply(mocker):
    client = fake_client(mocker, "prose", "more prose")
    with pytest.raises(VisionError, match="valid JSON"):
        describe_image(PNG, "a fox", checklist=CHECKLIST, client=client, model="m")
    assert client.chat.completions.create.call_count == 2


def test_describe_image_rejects_unsupported_format_without_calling_api(mocker):
    client = fake_client(mocker)
    with pytest.raises(VisionError, match="Unsupported image format"):
        describe_image(b"BM not supported", "prompt", checklist=CHECKLIST, client=client, model="m")
    client.chat.completions.create.assert_not_called()


def test_any_number_of_references_is_sent(mocker):
    client = fake_client(mocker, json.dumps({"requirements": CHECKLIST}), json.dumps(REPLY))
    references = [PNG] * 9

    build_checklist("a fox", references=references, client=client, model="m")
    describe_image(JPEG, "a fox", checklist=CHECKLIST, references=references, client=client, model="m")

    checklist_labels = [part["text"] for part in sent_content(client, 0) if part["type"] == "text"][1:]
    assert checklist_labels == [f"Reference image {n}:" for n in range(1, 10)]
    describe_parts = sent_content(client, 1)
    assert sum(part["type"] == "image_url" for part in describe_parts) == 10
    assert describe_parts[-2]["text"] == "CANDIDATE image:"


def noisy_image(width, height, mode="RGB"):
    """An incompressible image, so its encoded size is predictable."""
    return Image.frombytes(mode, (width, height), os.urandom(width * height * len(mode)))


def encode(image, fmt):
    output = io.BytesIO()
    image.save(output, format=fmt)
    return output.getvalue()


def test_shrink_leaves_small_files_untouched():
    assert shrink(PNG) == (PNG, "image/png")


def test_shrink_downsizes_large_photos_to_jpeg():
    data = encode(noisy_image(2400, 1200), "PNG")
    assert len(data) > MAX_UNSCALED_BYTES

    shrunk, mime = shrink(data)

    assert mime == "image/jpeg" and len(shrunk) < len(data)
    assert Image.open(io.BytesIO(shrunk)).size == (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE // 2)


def test_shrink_keeps_transparency_as_png():
    data = encode(noisy_image(1400, 1400, "RGBA"), "PNG")
    assert len(data) > MAX_UNSCALED_BYTES

    shrunk, mime = shrink(data)

    assert mime == "image/png"
    assert Image.open(io.BytesIO(shrunk)).mode == "RGBA"


def test_shrink_reports_corrupt_large_file():
    with pytest.raises(VisionError, match="Could not read image"):
        shrink(PNG + b"\x00" * (MAX_UNSCALED_BYTES + 1))


def test_parse_description_strips_code_fence():
    assert parse_description(f"```json\n{json.dumps(REPLY)}\n```") == DESCRIPTION


def test_parse_description_rejects_missing_keys():
    with pytest.raises(VisionError, match="defects"):
        parse_description(json.dumps({"composition": "x", "prompt_match": "y"}))


def test_description_dict_round_trip_omits_empty_fields():
    data = DESCRIPTION.as_dict()
    assert "notes" not in data and "reference_match" not in data
    assert data["requirements"][1] == {"requirement": "dusk lighting", "met": "no", "evidence": "bright daylight"}
    assert ImageDescription.from_dict(data) == DESCRIPTION


def test_from_dict_reads_descriptions_saved_before_checklists():
    old = {"composition": "c", "defects": "d", "prompt_match": "p", "notes": "n"}
    assert ImageDescription.from_dict(old) == ImageDescription(composition="c", defects="d", prompt_match="p", notes="n")
