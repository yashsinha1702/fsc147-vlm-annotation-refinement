import os
import io
import json
import time
import math

import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Vertex AI setup — credentials are NEVER stored in this repo.
#   GOOGLE_CLOUD_PROJECT            your GCP project id            (required)
#   GOOGLE_CLOUD_LOCATION           Vertex AI region (default us-central1)
#   GOOGLE_APPLICATION_CREDENTIALS  path to a service-account key JSON kept
#                                   outside the repo (or use
#                                   `gcloud auth application-default login`)
# ---------------------------------------------------------------------------
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION   = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
if not PROJECT_ID:
    raise RuntimeError(
        "Set GOOGLE_CLOUD_PROJECT (and GOOGLE_APPLICATION_CREDENTIALS or "
        "Application Default Credentials) before running the Gemini stages."
    )
vertexai.init(project=PROJECT_ID, location=LOCATION)

MODEL_PRO   = "gemini-2.5-pro"
MODEL_FLASH = "gemini-2.5-flash"

BATCH_SIZE = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_part(img: Image.Image) -> Part:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return Part.from_data(data=buf.getvalue(), mime_type="image/jpeg")


def _repair_truncated_json(text: str):
    text = text.strip()
    for suffix in ["", "]", "}]", '"}]', '"}}]']:
        try:
            return json.loads(text + suffix)
        except Exception:
            pass
    return None


def _extract_json(text: str, expect_list=True):
    if expect_list:
        start = text.find("[")
        end   = text.rfind("]") + 1
    else:
        start = text.find("{")
        end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        chunk = text[start:end]
        try:
            return json.loads(chunk)
        except Exception:
            return _repair_truncated_json(chunk)
    return _repair_truncated_json(text)


def query_gemini(images, prompt: str, max_tokens: int = 1024, use_flash: bool = False, retries: int = 3):
    model_name = MODEL_FLASH if use_flash else MODEL_PRO
    model  = GenerativeModel(model_name)
    parts  = [_pil_to_part(img) for img in images] + [prompt]
    config = GenerationConfig(max_output_tokens=max_tokens)
    for attempt in range(retries):
        try:
            resp = model.generate_content(parts, generation_config=config)
            return resp.text
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  [gemini] attempt {attempt+1} failed: {e} — retry in {wait}s")
                time.sleep(wait)
            else:
                raise
    return None


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _draw_existing(img: Image.Image, existing_points, radius=6, color="lime") -> Image.Image:
    out  = img.copy()
    draw = ImageDraw.Draw(out)
    for x, y in existing_points:
        x, y = int(x), int(y)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     outline=color, width=2)
    return out


def _draw_candidate_crosshair(img: Image.Image, cx, cy, color="cyan", size=12) -> Image.Image:
    out  = img.copy()
    draw = ImageDraw.Draw(out)
    cx, cy = int(cx), int(cy)
    draw.line([cx - size, cy, cx + size, cy], fill=color, width=2)
    draw.line([cx, cy - size, cx, cy + size], fill=color, width=2)
    return out


def _draw_numbered_candidates(img: Image.Image, candidates) -> Image.Image:
    out  = img.copy()
    draw = ImageDraw.Draw(out)
    for i, cand in enumerate(candidates):
        x, y = int(cand["point"][0]), int(cand["point"][1])
        r = 9
        draw.ellipse([x - r, y - r, x + r, y + r], outline="red", width=2)
        draw.text((x + r + 2, y - 8), str(i + 1), fill="red")
    return out


def _zoom_crop(img: Image.Image, cx, cy, pad=60) -> Image.Image:
    W, H  = img.size
    x1    = max(0, int(cx) - pad)
    y1    = max(0, int(cy) - pad)
    x2    = min(W, int(cx) + pad)
    y2    = min(H, int(cy) + pad)
    crop  = img.crop((x1, y1, x2, y2))
    scale = 256 / max(crop.width, crop.height)
    new_w = max(1, int(crop.width  * scale))
    new_h = max(1, int(crop.height * scale))
    return crop.resize((new_w, new_h), Image.LANCZOS)


def extract_exemplar_crops(ann, img_pil: Image.Image, max_crops=3):
    boxes = ann.get("box_examples_coordinates", [])
    crops = []
    for box in boxes[:max_crops]:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        pad = 10
        W, H = img_pil.size
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(W, x2 + pad)
        y2 = min(H, y2 + pad)
        if x2 > x1 and y2 > y1:
            crop = img_pil.crop((x1, y1, x2, y2))
            crop = crop.resize((96, 96), Image.LANCZOS)
            crops.append(crop)
    return crops


# ---------------------------------------------------------------------------
# Phase 1 – advisory count gate (non-gating)
# ---------------------------------------------------------------------------

def count_gate_advisory(img_pil: Image.Image, ann, use_flash=False) -> dict:
    n_existing = len(ann.get("points", []))
    exemplars  = extract_exemplar_crops(ann, img_pil)
    img_marked = _draw_existing(img_pil, ann.get("points", []))

    prompt = f"""This image has {n_existing} annotated objects (shown as green circles).
Exemplar objects are provided separately.

Roughly, do you think there are more objects of the same type that are NOT circled?
Answer with JSON:
{{"more_exist": true/false, "confidence": "high/medium/low", "estimate_missed": <integer or 0>}}

Only JSON, no other text."""

    images = [img_marked] + exemplars
    try:
        text   = query_gemini(images, prompt, max_tokens=256, use_flash=use_flash)
        parsed = _extract_json(text, expect_list=False)
        if parsed:
            return parsed
    except Exception as e:
        print(f"  [count_gate] error: {e}")
    return {"more_exist": True, "confidence": "low", "estimate_missed": 0}


# ---------------------------------------------------------------------------
# Phase 2 – batch verify candidates (used by stage3 pipeline)
# ---------------------------------------------------------------------------

def batch_verify_candidates(img_pil: Image.Image, ann, candidates: list, use_flash=False) -> list:
    """
    candidates: list of [x, y]
    Returns list of booleans (True = keep).
    """
    if not candidates:
        return []

    existing   = ann.get("points", [])
    exemplars  = extract_exemplar_crops(ann, img_pil)
    img_marked = _draw_existing(img_pil, existing)

    cand_dicts = [{"point": c} for c in candidates]
    img_num    = _draw_numbered_candidates(img_marked, cand_dicts)

    prompt = f"""The image shows {len(existing)} already-annotated objects (green circles).
{len(candidates)} candidate points (red numbered circles) may be missed objects.
Exemplar images show what the objects look like.

For each candidate (1 to {len(candidates)}), decide if it is a genuine missed object of the SAME TYPE as the exemplars.

Respond with a JSON array of exactly {len(candidates)} booleans:
[true, false, ...]

true  = genuine missed object
false = wrong / different type / already annotated

Only the JSON array, no other text."""

    images = [img_num] + exemplars
    try:
        text   = query_gemini(images, prompt, max_tokens=512, use_flash=use_flash)
        parsed = _extract_json(text, expect_list=True)
        if isinstance(parsed, list):
            results = [bool(v) for v in parsed]
            if len(results) < len(candidates):
                results += [True] * (len(candidates) - len(results))
            return results[:len(candidates)]
    except Exception as e:
        print(f"  [batch_verify] error: {e}")
    return [True] * len(candidates)


# ---------------------------------------------------------------------------
# Phase 3 – zoomed single candidate confirm
# ---------------------------------------------------------------------------

def confirm_candidate_zoomed(img_pil: Image.Image, ann, candidate_point, use_flash=False) -> dict:
    cx, cy    = candidate_point
    existing  = ann.get("points", [])
    exemplars = extract_exemplar_crops(ann, img_pil)

    img_marked = _draw_existing(img_pil, existing)
    img_marked = _draw_candidate_crosshair(img_marked, cx, cy)
    zoom       = _zoom_crop(img_marked, cx, cy, pad=80)

    prompt = f"""You are verifying a single candidate annotation (cyan crosshair) in a counting dataset.
Green circles = already annotated objects.
Exemplar images show what the target object looks like.

Answer strictly with JSON:
{{
  "is_valid_object": true/false,
  "on_object_or_gap": "object" | "gap" | "background",
  "same_type_as_exemplar": true/false,
  "confidence": "high" | "medium" | "low",
  "reason": "one sentence"
}}

Reject if:
- The crosshair falls on empty space / gap between objects
- The crosshair falls on background pixels
- The object is a clearly different type from the exemplars
- The object is in a different state (e.g. unlit candle vs lit candle)
- It is part of an already-annotated object's body

Only JSON, no other text."""

    images = [zoom, img_pil] + exemplars
    try:
        text   = query_gemini(images, prompt, max_tokens=512, use_flash=use_flash)
        parsed = _extract_json(text, expect_list=False)
        if parsed and "is_valid_object" in parsed:
            return parsed
    except Exception as e:
        print(f"  [confirm_zoomed] error: {e}")
    return {
        "is_valid_object": False,
        "on_object_or_gap": "unknown",
        "same_type_as_exemplar": False,
        "confidence": "low",
        "reason": "verification error",
    }


# ---------------------------------------------------------------------------
# Rejection filter – Flash conservative batch rejection
# ---------------------------------------------------------------------------

def batch_reject_obvious(img_pil: Image.Image, ann, candidates: list, use_pro=False) -> list:
    """
    candidates: list of dicts with 'point' key (and optionally others).
    Default policy: KEEP.  Only rejects if CLEARLY wrong type.
    Returns list of {keep: bool, reason: str} dicts, one per candidate.
    """
    if not candidates:
        return []

    existing   = ann.get("points", [])
    exemplars  = extract_exemplar_crops(ann, img_pil)
    img_marked = _draw_existing(img_pil, existing)
    img_num    = _draw_numbered_candidates(img_marked, candidates)

    prompt = f"""You are helping build a gold-standard counting annotation dataset.

The image shows existing annotations (green circles) and {len(candidates)} candidate missed objects (red numbered circles).
Exemplar images show what the target object class looks like.

For each numbered candidate, decide KEEP or REJECT.

Rules (very conservative — when in doubt, KEEP):
- KEEP  if the candidate looks like it could be the same type of object as the exemplars
- KEEP  if it is ambiguous or hard to see
- REJECT only if it is CLEARLY a completely different type of object
- Do NOT reject due to occlusion, blur, or small size

Respond with a JSON array of exactly {len(candidates)} objects:
[
  {{"keep": true, "reason": "brief reason"}},
  ...
]

Only JSON, no other text."""

    images = [img_num] + exemplars[:3]
    try:
        text   = query_gemini(images, prompt, max_tokens=2048, use_flash=not use_pro)
        parsed = _extract_json(text, expect_list=True)
        if isinstance(parsed, list):
            results = []
            for item in parsed:
                if isinstance(item, dict):
                    results.append({
                        "keep":   bool(item.get("keep", True)),
                        "reason": str(item.get("reason", "")),
                    })
                else:
                    results.append({"keep": True, "reason": "parse fallback"})
            if len(results) < len(candidates):
                results += [{"keep": True, "reason": "no response"}] * (len(candidates) - len(results))
            return results[:len(candidates)]
    except Exception as e:
        print(f"  [batch_reject] error: {e}")

    return [{"keep": True, "reason": "filter error, default keep"} for _ in candidates]
