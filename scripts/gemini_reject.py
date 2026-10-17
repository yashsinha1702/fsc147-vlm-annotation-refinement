"""
gemini_reject.py
Conservative Gemini Flash rejection filter.
Reads unannotated_candidates_clean.json, runs batch rejection per image,
checkpoints per-image results, outputs review_pipeline/potential_candidates.json.

Usage:
    python scripts/gemini_reject.py              # all images
    python scripts/gemini_reject.py --pilot 30   # first 30 images only
    python scripts/gemini_reject.py --pro        # use Pro instead of Flash
"""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

from gemini_client import batch_reject_obvious, count_gate_advisory, BATCH_SIZE
from gemini_client import extract_exemplar_crops

from config import ANNOTATIONS_FILE, CANDIDATES_FILE, IMAGES_DIR
from config import REVIEW_DIR, PER_IMAGE_DIR, POTENTIAL_FILE, FILTER_LOG_FILE


def load_per_image_checkpoint(img_name: str) -> dict | None:
    p = PER_IMAGE_DIR / f"{img_name}.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def save_per_image_checkpoint(img_name: str, data: dict):
    p = PER_IMAGE_DIR / f"{img_name}.json"
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, p)


def process_image(img_name: str, candidates: list, annotations: dict, use_pro: bool) -> dict:
    img_path = os.path.join(IMAGES_DIR, img_name)
    if not os.path.exists(img_path):
        print(f"  [skip] image not found: {img_path}")
        return {"kept": candidates, "rejected": [], "advisory": {}}

    ann     = annotations.get(img_name, {})
    img_pil = Image.open(img_path).convert("RGB")

    # Phase 1 – advisory (non-gating)
    advisory = {}
    try:
        advisory = count_gate_advisory(img_pil, ann, use_flash=not use_pro)
        print(f"  advisory: more_exist={advisory.get('more_exist')}, "
              f"confidence={advisory.get('confidence')}, "
              f"est_missed={advisory.get('estimate_missed')}")
    except Exception as e:
        print(f"  advisory error: {e}")

    # Phase 2 – batch rejection in chunks
    cand_dicts = [{"point": pt} for pt in candidates]
    keep_results = []

    for chunk_start in range(0, len(cand_dicts), BATCH_SIZE):
        chunk = cand_dicts[chunk_start: chunk_start + BATCH_SIZE]
        print(f"  batch {chunk_start // BATCH_SIZE + 1}: {len(chunk)} candidates …", end=" ", flush=True)
        results = batch_reject_obvious(img_pil, ann, chunk, use_pro=use_pro)
        keep_results.extend(results)
        kept_in_chunk = sum(1 for r in results if r["keep"])
        print(f"kept {kept_in_chunk}/{len(chunk)}")

    kept     = []
    rejected = []
    for pt, res in zip(candidates, keep_results):
        entry = {"point": pt, "keep": res["keep"], "reason": res["reason"]}
        if res["keep"]:
            kept.append(entry)
        else:
            rejected.append(entry)

    return {"kept": kept, "rejected": rejected, "advisory": advisory}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, default=None,
                        help="Process only first N images with candidates")
    parser.add_argument("--pro",   action="store_true",
                        help="Use Gemini Pro instead of Flash")
    args = parser.parse_args()

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    PER_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading annotations …")
    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)

    print("Loading candidates …")
    with open(CANDIDATES_FILE) as f:
        candidates_map = json.load(f)

    images = sorted(candidates_map.keys())
    if args.pilot:
        images = images[: args.pilot]
        print(f"Pilot mode: processing {len(images)} images")

    model_label = "Pro" if args.pro else "Flash"
    print(f"Using Gemini {model_label}")
    print(f"Total images to process: {len(images)}\n")

    # Resume: load already-processed checkpoints
    already_done = set()
    potential    = {}   # img -> list of kept candidate dicts
    filter_log   = {}   # img -> {kept, rejected, advisory}

    for img_name in images:
        ckpt = load_per_image_checkpoint(img_name)
        if ckpt is not None:
            already_done.add(img_name)
            filter_log[img_name] = ckpt
            if ckpt["kept"]:
                potential[img_name] = ckpt["kept"]

    todo = [img for img in images if img not in already_done]
    print(f"Already done: {len(already_done)}, remaining: {len(todo)}\n")

    total_kept     = sum(len(v) for v in potential.values())
    total_rejected = sum(len(v["rejected"]) for v in filter_log.values() if "rejected" in v)

    for idx, img_name in enumerate(todo, 1):
        candidates = candidates_map[img_name]
        print(f"[{idx}/{len(todo)}] {img_name}  ({len(candidates)} candidates)")

        try:
            result = process_image(img_name, candidates, annotations, use_pro=args.pro)
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {"kept": [{"point": pt, "keep": True, "reason": "error fallback"}
                               for pt in candidates],
                      "rejected": [], "advisory": {}}

        save_per_image_checkpoint(img_name, result)
        filter_log[img_name] = result

        if result["kept"]:
            potential[img_name] = result["kept"]

        total_kept     += len(result["kept"])
        total_rejected += len(result["rejected"])
        print(f"  → kept {len(result['kept'])}, rejected {len(result['rejected'])}  "
              f"(running: kept={total_kept}, rejected={total_rejected})\n")

        # Incremental save
        _atomic_json(POTENTIAL_FILE, potential)
        _atomic_json(FILTER_LOG_FILE, filter_log)

    print("\n=== Done ===")
    print(f"Total kept     : {total_kept}")
    print(f"Total rejected : {total_rejected}")
    print(f"Potential file : {POTENTIAL_FILE}")
    print(f"Filter log     : {FILTER_LOG_FILE}")


def _atomic_json(path, data):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
