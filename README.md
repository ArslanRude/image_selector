# Image Selector

Paste or upload two or more images plus a text prompt, and get back which image matches the prompt best.

1. **Checklist:** DeepSeek (`deepseek-flash`) turns the prompt, any extra instructions, and any reference images into 3 to 8 concrete requirements, most important first.
2. **Grading:** DeepSeek grades every candidate against that same checklist (yes, partial, or no, with evidence), next to the reference images. It also notes defects and how closely each candidate matches the references.
3. **Decision:** Jev (TypeSafe AI, `Choice` primitive) picks the winner from those grades. It decides twice, once with the images in reverse order, and averages the two. If the two runs disagree, the verdict is flagged as a close call. Jev never sees image bytes.
4. **Feedback memory:** after each verdict, tell it which image was actually best and, optionally, why. A right pick scores +1 and a wrong one -1. Every rating is saved to `image_selector_memory.jsonl` in the project folder. Each decision gets the 8 past ratings most relevant to the current prompt, corrections first, so Jev learns your taste. Delete the file to reset.

**Optional inputs:**
- **Extra instructions:** rules the checklist, grading, and decision must follow, such as "no text or watermarks" or "anatomy matters most".
- **Reference images:** any number of examples of the look, subject, or style you want. Each reference is sent with every candidate, so many references mean bigger, slower DeepSeek calls. Files over 1 MB are shrunk to 1024 px on the longest side before sending.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env   # then fill in DEEPSEEK_API_KEY and TYPESAFE_API_KEY
```

## Run

```bash
.venv/Scripts/python -m image_selector.app
```

Open http://localhost:7860. Images are numbered left to right in the order they appear in the gallery.

## Test

```bash
.venv/Scripts/python -m pytest
```

Tests mock both APIs and make no network calls.
