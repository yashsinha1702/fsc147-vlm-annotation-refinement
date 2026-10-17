"""
config.py
Central path configuration for the refinement pipeline.

All paths are relative to the repository root by default and can be
overridden with environment variables:

    FSC147_DIR          folder holding annotation_FSC147_384.json and images_384_VarV2/
                        (default: <repo>/data/FSC147)
    VLMBTP_OUTPUT_DIR   folder for pipeline outputs (default: <repo>/outputs)
    VLMBTP_PREDICTIONS  counting-model point predictions
                        (default: <VLMBTP_OUTPUT_DIR>/model_predictions_final.json)
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR   = Path(os.environ.get("FSC147_DIR", REPO_ROOT / "data" / "FSC147"))
OUTPUT_DIR = Path(os.environ.get("VLMBTP_OUTPUT_DIR", REPO_ROOT / "outputs"))

# FSC-147 dataset (download separately, see README)
ANNOTATIONS_FILE = DATA_DIR / "annotation_FSC147_384.json"
IMAGES_DIR       = DATA_DIR / "images_384_VarV2"

# Pipeline inputs / outputs
PREDICTIONS_FILE = Path(os.environ.get("VLMBTP_PREDICTIONS",
                                       OUTPUT_DIR / "model_predictions_final.json"))
CANDIDATES_FILE  = OUTPUT_DIR / "unannotated_candidates_clean.json"
REFINED_FILE     = OUTPUT_DIR / "annotation_FSC147_384_refined_human.json"

REVIEW_DIR       = OUTPUT_DIR / "review_pipeline"
PER_IMAGE_DIR    = REVIEW_DIR / "per_image"
POTENTIAL_FILE   = REVIEW_DIR / "potential_candidates.json"
FILTER_LOG_FILE  = REVIEW_DIR / "gemini_filter_log.json"
DECISIONS_FILE   = REVIEW_DIR / "decisions.json"
REFILTER_LOG     = REVIEW_DIR / "pro_refilter_log.json"
SUMMARY_FILE     = REVIEW_DIR / "merge_summary.json"
