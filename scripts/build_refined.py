"""
build_refined.py
Merges human-accepted candidates from decisions.json into the original
annotation file to produce annotation_FSC147_384_refined_human.json.

Usage:
    python scripts/build_refined.py
"""

import json
import os
from collections import defaultdict
from pathlib import Path

from config import ANNOTATIONS_FILE, DECISIONS_FILE, POTENTIAL_FILE, SUMMARY_FILE
from config import REFINED_FILE as OUTPUT_FILE


def candidate_key(img_name: str, point) -> str:
    return f"{img_name}__{int(round(point[0]))}_{int(round(point[1]))}"


def main():
    print("Loading annotations …")
    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)

    print("Loading decisions …")
    with open(DECISIONS_FILE) as f:
        decisions = json.load(f)

    print("Loading potential candidates …")
    with open(POTENTIAL_FILE) as f:
        potential = json.load(f)

    # Group accepted points by image
    accepted_by_image = defaultdict(list)
    for img_name, candidates in potential.items():
        for cand in candidates:
            key = candidate_key(img_name, cand["point"])
            if decisions.get(key) == "accept":
                accepted_by_image[img_name].append(cand["point"])

    # Build refined annotations
    import copy
    refined     = copy.deepcopy(annotations)
    stats       = {
        "images_modified":   0,
        "total_added":       0,
        "decisions_accept":  sum(1 for d in decisions.values() if d == "accept"),
        "decisions_reject":  sum(1 for d in decisions.values() if d == "reject"),
        "decisions_uncertain": sum(1 for d in decisions.values() if d == "uncertain"),
        "decisions_skip":    sum(1 for d in decisions.values() if d == "skip"),
        "decisions_undecided": sum(1 for d in decisions.values() if d == "undecided"),
        "per_image": {},
    }

    for img_name, new_points in accepted_by_image.items():
        if img_name not in refined:
            print(f"  [warn] {img_name} not in original annotations, skipping")
            continue
        original_count = len(refined[img_name]["points"])
        for pt in new_points:
            refined[img_name]["points"].append(pt)
        added = len(new_points)
        stats["images_modified"] += 1
        stats["total_added"]     += added
        stats["per_image"][img_name] = {
            "original_count": original_count,
            "added":          added,
            "new_count":      original_count + added,
        }
        print(f"  {img_name}: {original_count} → {original_count + added} (+{added})")

    print(f"\nTotal images modified : {stats['images_modified']}")
    print(f"Total annotations added: {stats['total_added']}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(refined, f)
    print(f"\nSaved refined annotations → {OUTPUT_FILE}")

    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved summary           → {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
