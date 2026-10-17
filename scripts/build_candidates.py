"""
build_candidates.py
Reads model_predictions_final.json and annotation_FSC147_384.json,
filters predicted points to find genuine unannotated candidates,
outputs unannotated_candidates_clean.json.
"""

import json
import numpy as np
from scipy.spatial import cKDTree

from config import ANNOTATIONS_FILE, PREDICTIONS_FILE
from config import CANDIDATES_FILE as OUTPUT_FILE

OOB_MARGIN             = 2
ELONGATED_ASPECT_THRESH = 2.0
ELONGATED_WIDTH_FACTOR  = 0.7
ELONGATED_HEIGHT_FACTOR = 3.5


# ---------------------------------------------------------------------------
# Object size / shape from exemplar boxes
# ---------------------------------------------------------------------------

def compute_object_sizes(ann: dict) -> dict:
    boxes = ann.get("box_examples_coordinates", [])
    widths, heights = [], []
    for box in boxes:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        w  = max(xs) - min(xs)
        h  = max(ys) - min(ys)
        if w > 0 and h > 0:
            widths.append(w)
            heights.append(h)

    if not widths:
        return {
            "mean_size": 20.0, "mean_w": 20.0, "mean_h": 20.0,
            "is_elongated": False,
            "elongated_w": 20.0, "elongated_h": 20.0, "max_aspect": 1.0,
        }

    mean_w = float(np.mean(widths))
    mean_h = float(np.mean(heights))
    mean_size = (mean_w + mean_h) / 2.0

    # Per-box aspect ratio — fixes the cross-box max/min bug
    max_per_box_aspect = 1.0
    elongated_w, elongated_h = mean_w, mean_h
    for w, h in zip(widths, heights):
        a = max(w, h) / min(w, h)
        if a > max_per_box_aspect:
            max_per_box_aspect = a
            elongated_w, elongated_h = float(w), float(h)

    return {
        "mean_size":    mean_size,
        "mean_w":       mean_w,
        "mean_h":       mean_h,
        "is_elongated": max_per_box_aspect > ELONGATED_ASPECT_THRESH,
        "elongated_w":  elongated_w,
        "elongated_h":  elongated_h,
        "max_aspect":   max_per_box_aspect,
    }


# ---------------------------------------------------------------------------
# Adaptive proximity threshold
# ---------------------------------------------------------------------------

def compute_nn_distance(points) -> float:
    if len(points) < 2:
        return 20.0
    pts  = np.array(points, dtype=float)
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    return float(np.median(dists[:, 1]))


def compute_adaptive_threshold(existing_points, size_info: dict) -> float:
    nn        = compute_nn_distance(existing_points)
    mean_size = size_info["mean_size"]
    return max(0.8 * nn, 0.5 * mean_size)


# ---------------------------------------------------------------------------
# Elongated-body vertical-strip filter
# ---------------------------------------------------------------------------

def is_elongated_body(cand_pt, existing_points, size_info: dict) -> bool:
    if not size_info["is_elongated"]:
        return False

    ew         = size_info["elongated_w"]
    eh         = size_info["elongated_h"]
    short_side = min(ew, eh)
    long_side  = max(ew, eh)

    half_w = (ELONGATED_WIDTH_FACTOR  * short_side) / 2.0
    half_h = (ELONGATED_HEIGHT_FACTOR * long_side)  / 2.0

    cx, cy = cand_pt
    for px, py in existing_points:
        if abs(cx - px) <= half_w and abs(cy - py) <= half_h:
            return True
    return False


# ---------------------------------------------------------------------------
# NMS among remaining candidates
# ---------------------------------------------------------------------------

def nms_candidates(candidates, min_dist: float):
    kept = []
    for pt in candidates:
        too_close = any(
            np.hypot(pt[0] - k[0], pt[1] - k[1]) < min_dist
            for k in kept
        )
        if not too_close:
            kept.append(pt)
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading annotations …")
    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)

    print("Loading predictions …")
    with open(PREDICTIONS_FILE) as f:
        predictions = json.load(f)

    output = {}
    stats  = {
        "total_preds":     0,
        "oob":             0,
        "matched_existing": 0,
        "elongated_body":  0,
        "nms_dups":        0,
        "kept":            0,
        "images_with_candidates": 0,
    }

    for img, preds in predictions.items():
        if img not in annotations:
            continue

        ann      = annotations[img]
        H, W     = ann["H"], ann["W"]
        existing = [[float(p[0]), float(p[1])] for p in ann["points"]]

        size_info = compute_object_sizes(ann)
        thresh    = (
            compute_adaptive_threshold(existing, size_info)
            if existing
            else size_info["mean_size"]
        )

        # Build KDTree once per image
        tree = cKDTree(existing) if existing else None

        candidates = []
        for pred in preds:
            x, y = float(pred[0]), float(pred[1])
            stats["total_preds"] += 1

            # Out-of-bounds
            if x < OOB_MARGIN or y < OOB_MARGIN or x > W - OOB_MARGIN or y > H - OOB_MARGIN:
                stats["oob"] += 1
                continue

            # Too close to existing annotation
            if tree is not None:
                dist, _ = tree.query([x, y], k=1)
                if dist < thresh:
                    stats["matched_existing"] += 1
                    continue

            # Elongated-body strip
            if is_elongated_body([x, y], existing, size_info):
                stats["elongated_body"] += 1
                continue

            candidates.append([x, y])

        # NMS among candidates themselves
        before_nms = len(candidates)
        nms_dist   = max(thresh * 0.5, size_info["mean_size"] * 0.4)
        candidates = nms_candidates(candidates, nms_dist)
        stats["nms_dups"] += before_nms - len(candidates)

        if candidates:
            output[img] = candidates
            stats["kept"] += len(candidates)
            stats["images_with_candidates"] += 1

    print("\n=== Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nImages with candidates : {stats['images_with_candidates']}")
    print(f"Total kept candidates  : {stats['kept']}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f)
    print(f"\nSaved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
