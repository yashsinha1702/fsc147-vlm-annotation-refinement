"""
pro_refilter.py
Runs Gemini Pro over Flash-kept candidates that the user hasn't reviewed yet.
Marks obvious rejects directly in decisions.json so the UI skips them.

Usage:
    python scripts/pro_refilter.py
    python scripts/pro_refilter.py --pilot 20   # test on first 20 images
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from gemini_client import query_gemini, extract_exemplar_crops, _extract_json, BATCH_SIZE
from gemini_client import _draw_existing as draw_existing_only, _draw_numbered_candidates

from config import ANNOTATIONS_FILE, POTENTIAL_FILE, DECISIONS_FILE, IMAGES_DIR, REFILTER_LOG


def candidate_key(img_name: str, point) -> str:
    return f"{img_name}__{int(round(point[0]))}_{int(round(point[1]))}"


def load_decisions() -> dict:
    if DECISIONS_FILE.exists():
        with open(DECISIONS_FILE) as f:
            return json.load(f)
    return {}


def save_decisions(new_decisions: dict):
    """
    Merge-write: reload disk first, then add only keys not already present.
    This ensures user decisions made concurrently are never overwritten.
    """
    on_disk = load_decisions()
    # Only add Pro decisions for keys the user hasn't touched yet
    for k, v in new_decisions.items():
        if k not in on_disk:
            on_disk[k] = v
    tmp = str(DECISIONS_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(on_disk, f, indent=2)
    os.replace(tmp, DECISIONS_FILE)


def pro_batch_filter(img_pil: Image.Image, ann: dict, candidates: list) -> list:
    """
    Stricter Pro filter. Returns list of {keep: bool, reason: str}.
    Stricter than Flash — rejects if not clearly the same object type.
    """
    existing  = ann.get("points", [])
    exemplars = extract_exemplar_crops(ann, img_pil)

    img_marked = draw_existing_only(img_pil, existing)
    cand_dicts = [{"point": c["point"]} for c in candidates]
    img_num    = _draw_numbered_candidates(img_marked, cand_dicts)

    prompt = f"""You are a quality-control expert reviewing annotation candidates for a counting dataset.

The target object class is shown in the exemplar images.
Green circles = already correctly annotated objects.
Red numbered circles = {len(candidates)} candidates to evaluate.

For each numbered candidate, decide KEEP or REJECT.

Be stricter than usual:
- KEEP only if the candidate is clearly the same object type as the exemplars
- REJECT if it is a different object type, part of an already-annotated object, background, or gap
- REJECT if it is clearly not countable as a separate instance
- KEEP if genuinely ambiguous due to occlusion or blur (not wrong type)

Respond with a JSON array of exactly {len(candidates)} objects:
[
  {{"keep": true/false, "reason": "brief reason"}},
  ...
]

Only JSON, no other text."""

    images = [img_num] + exemplars[:3]
    try:
        text   = query_gemini(images, prompt, max_tokens=2048, use_flash=False)
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
        print(f"  [pro_filter] error: {e}")

    return [{"keep": True, "reason": "error fallback"} for _ in candidates]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, default=None)
    args = parser.parse_args()

    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)

    with open(POTENTIAL_FILE) as f:
        potential = json.load(f)

    decisions = load_decisions()

    # Build list of (img, [unreviewed candidates]) — skip already decided by user
    todo_images = []
    for img_name in sorted(potential.keys()):
        unreviewed = [
            c for c in potential[img_name]
            if decisions.get(candidate_key(img_name, c["point"])) not in
               ("accept", "reject", "uncertain")
        ]
        if unreviewed:
            todo_images.append((img_name, unreviewed))

    if args.pilot:
        todo_images = todo_images[:args.pilot]

    total_unreviewed = sum(len(c) for _, c in todo_images)
    print(f"Images to refilter  : {len(todo_images)}")
    print(f"Candidates to filter: {total_unreviewed}\n")

    refilter_log  = {}
    total_kept    = 0
    total_rejected = 0

    for idx, (img_name, candidates) in enumerate(todo_images, 1):
        img_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"[{idx}] {img_name} — image not found, skipping")
            continue

        ann     = annotations.get(img_name, {})
        img_pil = Image.open(img_path).convert("RGB")

        print(f"[{idx}/{len(todo_images)}] {img_name}  ({len(candidates)} unreviewed)")

        img_kept = 0
        img_rejected = 0
        img_log  = []

        for chunk_start in range(0, len(candidates), BATCH_SIZE):
            chunk = candidates[chunk_start: chunk_start + BATCH_SIZE]
            print(f"  batch {chunk_start // BATCH_SIZE + 1}: {len(chunk)} candidates …", end=" ", flush=True)

            results = pro_batch_filter(img_pil, ann, chunk)

            for cand, res in zip(chunk, results):
                key = candidate_key(img_name, cand["point"])
                if not res["keep"]:
                    decisions[key] = "reject"
                    img_rejected += 1
                else:
                    img_kept += 1
                img_log.append({
                    "point":  cand["point"],
                    "keep":   res["keep"],
                    "reason": res["reason"],
                })

            print(f"kept {img_kept}, rejected {img_rejected} so far")

            # Save decisions after every batch
            save_decisions(decisions)

        total_kept     += img_kept
        total_rejected += img_rejected
        refilter_log[img_name] = img_log
        print(f"  → kept {img_kept}, rejected {img_rejected}  "
              f"(running: kept={total_kept}, rejected={total_rejected})\n")

        # Save log incrementally
        tmp = str(REFILTER_LOG) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(refilter_log, f)
        os.replace(tmp, REFILTER_LOG)

    print("=== Done ===")
    print(f"Total kept     : {total_kept}")
    print(f"Total rejected : {total_rejected}")
    rejection_rate = total_rejected / max(total_kept + total_rejected, 1) * 100
    print(f"Rejection rate : {rejection_rate:.1f}%")


if __name__ == "__main__":
    main()
