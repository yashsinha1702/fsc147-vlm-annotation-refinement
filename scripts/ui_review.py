"""
ui_review.py  –  Streamlit human-review UI for annotation candidates.

Launch (from the repository root):
    streamlit run scripts/ui_review.py

Reads  : review_pipeline/potential_candidates.json
Writes : review_pipeline/decisions.json  (atomic, after every click)
"""

import json
import os
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from config import ANNOTATIONS_FILE, POTENTIAL_FILE, DECISIONS_FILE, IMAGES_DIR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def candidate_key(img_name: str, point) -> str:
    return f"{img_name}__{int(round(point[0]))}_{int(round(point[1]))}"


def load_decisions() -> dict:
    if DECISIONS_FILE.exists():
        with open(DECISIONS_FILE) as f:
            return json.load(f)
    return {}


def save_one_decision(key: str, value: str):
    """Merge-write a single decision — never overwrites other keys on disk."""
    DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    on_disk = load_decisions()
    on_disk[key] = value
    tmp = str(DECISIONS_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(on_disk, f, indent=2)
    os.replace(tmp, DECISIONS_FILE)


def save_decisions(decisions: dict):
    """Bulk save — merges with disk so concurrent writes aren't lost."""
    DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    on_disk = load_decisions()
    on_disk.update(decisions)
    tmp = str(DECISIONS_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(on_disk, f, indent=2)
    os.replace(tmp, DECISIONS_FILE)


@st.cache_data
def load_data():
    with open(POTENTIAL_FILE) as f:
        potential = json.load(f)
    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)
    return potential, annotations


@st.cache_data
def open_image(img_name: str) -> Image.Image:
    return Image.open(os.path.join(IMAGES_DIR, img_name)).convert("RGB")


def image_progress(img_name, candidates, decisions):
    keys    = [candidate_key(img_name, c["point"]) for c in candidates]
    decided = sum(1 for k in keys if decisions.get(k) in ("accept", "reject", "uncertain"))
    return decided, len(keys)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_existing_only(img_pil, existing, radius=6):
    out  = img_pil.copy()
    draw = ImageDraw.Draw(out)
    for x, y in existing:
        x, y = int(x), int(y)
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], outline="lime", width=2)
    return out


def draw_with_candidate(img_pil, existing, point, radius=6):
    out  = draw_existing_only(img_pil, existing, radius)
    draw = ImageDraw.Draw(out)
    cx, cy = int(point[0]), int(point[1])
    sz = 14
    draw.line([cx-sz, cy, cx+sz, cy], fill="cyan", width=3)
    draw.line([cx, cy-sz, cx, cy+sz], fill="cyan", width=3)
    draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill="cyan")
    return out


def zoom_crop(img_pil, cx, cy, pad=70):
    W, H = img_pil.size
    x1 = max(0, int(cx)-pad);  y1 = max(0, int(cy)-pad)
    x2 = min(W, int(cx)+pad);  y2 = min(H, int(cy)+pad)
    crop  = img_pil.crop((x1, y1, x2, y2))
    scale = 300 / max(crop.width, crop.height, 1)
    return crop.resize((max(1, int(crop.width*scale)), max(1, int(crop.height*scale))), Image.LANCZOS)


def exemplar_strip(ann, img_pil, size=96):
    boxes = ann.get("box_examples_coordinates", [])
    crops = []
    for box in boxes[:4]:
        xs = [p[0] for p in box];  ys = [p[1] for p in box]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        W, H = img_pil.size
        x1 = max(0, x1-8);  y1 = max(0, y1-8)
        x2 = min(W, x2+8);  y2 = min(H, y2+8)
        if x2 > x1 and y2 > y1:
            crops.append(img_pil.crop((x1,y1,x2,y2)).resize((size,size), Image.LANCZOS))
    if not crops:
        return None
    strip = Image.new("RGB", (len(crops)*(size+4), size), (40,40,40))
    for i, c in enumerate(crops):
        strip.paste(c, (i*(size+4), 0))
    return strip


# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Annotation Review", layout="wide")
st.title("Annotation Candidate Review")

if not POTENTIAL_FILE.exists():
    st.error(f"Not found: {POTENTIAL_FILE}")
    st.stop()

potential, annotations = load_data()

# Single source of truth for navigation
if "img_idx"  not in st.session_state: st.session_state.img_idx  = 0
if "cand_idx" not in st.session_state: st.session_state.cand_idx = 0
if "decisions" not in st.session_state: st.session_state.decisions = load_decisions()

all_images  = sorted(potential.keys())
all_pairs   = [(img, c) for img in all_images for c in potential[img]]
total_pairs = len(all_pairs)

# Clamp indices
st.session_state.img_idx  = max(0, min(st.session_state.img_idx,  len(all_images) - 1))

decisions = st.session_state.decisions

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Images")

    total_decided = sum(
        1 for (img, c) in all_pairs
        if decisions.get(candidate_key(img, c["point"])) in ("accept", "reject", "uncertain")
    )
    st.progress(total_decided / max(total_pairs, 1),
                text=f"{total_decided} / {total_pairs} decided")

    # Selectbox: keep in sync with img_idx via pre-setting session state key
    st.session_state["_img_select"] = all_images[st.session_state.img_idx]

    def _on_img_select():
        new = st.session_state["_img_select"]
        st.session_state.img_idx  = all_images.index(new)
        st.session_state.cand_idx = 0

    st.selectbox(
        "Jump to image",
        options=all_images,
        key="_img_select",
        on_change=_on_img_select,
        format_func=lambda img: (
            f"✅ {img}" if image_progress(img, potential[img], decisions)[0] == len(potential[img])
            else f"◐ {img}"  if image_progress(img, potential[img], decisions)[0] > 0
            else img
        ),
    )

    st.divider()
    accepted  = sum(1 for d in decisions.values() if d == "accept")
    rejected  = sum(1 for d in decisions.values() if d == "reject")
    uncertain = sum(1 for d in decisions.values() if d == "uncertain")
    st.metric("Accepted",  accepted)
    st.metric("Rejected",  rejected)
    st.metric("Uncertain", uncertain)

    st.divider()
    view_mode = st.radio("View mode", ["Single", "Grid (4-wide)"])

    st.divider()
    if st.button("Reload decisions from disk"):
        st.session_state.decisions = load_decisions()
        st.rerun()

# ---------------------------------------------------------------------------
# Current image
# ---------------------------------------------------------------------------
selected_img = all_images[st.session_state.img_idx]
image_pairs  = [(img, c) for (img, c) in all_pairs if img == selected_img]

st.session_state.cand_idx = max(0, min(st.session_state.cand_idx, len(image_pairs) - 1))

done_img, total_img = image_progress(selected_img, potential[selected_img], decisions)
st.subheader(f"{selected_img}  —  {done_img}/{total_img} decided")

if not image_pairs:
    st.info("No candidates for this image.")
    st.stop()

# ---------------------------------------------------------------------------
# Single mode
# ---------------------------------------------------------------------------
if view_mode == "Single":

    # Nav bar
    nav1, nav2, nav3 = st.columns([1, 6, 1])
    with nav1:
        if st.button("◀ Prev"):
            if st.session_state.cand_idx > 0:
                st.session_state.cand_idx -= 1
            elif st.session_state.img_idx > 0:
                st.session_state.img_idx -= 1
                st.session_state.cand_idx = len([c for img,c in all_pairs if img == all_images[st.session_state.img_idx]]) - 1
            st.rerun()
    with nav2:
        st.markdown(
            f"<div style='text-align:center;padding-top:6px'>"
            f"Candidate <b>{st.session_state.cand_idx+1}</b> / {len(image_pairs)}</div>",
            unsafe_allow_html=True,
        )
    with nav3:
        if st.button("Next ▶"):
            if st.session_state.cand_idx < len(image_pairs) - 1:
                st.session_state.cand_idx += 1
            elif st.session_state.img_idx < len(all_images) - 1:
                st.session_state.img_idx  += 1
                st.session_state.cand_idx  = 0
            st.rerun()

    img_name, cand = image_pairs[st.session_state.cand_idx]
    key              = candidate_key(img_name, cand["point"])
    current_decision = decisions.get(key, "undecided")

    ann      = annotations.get(img_name, {})
    img_pil  = open_image(img_name)
    existing = [[p[0], p[1]] for p in ann.get("points", [])]

    # Three columns: existing-only | with-candidate | zoom+controls
    col1, col2, col3 = st.columns([3, 3, 1])

    with col1:
        st.image(draw_existing_only(img_pil, existing),
                 caption="Existing annotations only", use_container_width=True)

    with col2:
        st.image(draw_with_candidate(img_pil, existing, cand["point"]),
                 caption="With candidate (cyan)", use_container_width=True)

    with col3:
        st.image(zoom_crop(draw_with_candidate(img_pil, existing, cand["point"]),
                           cand["point"][0], cand["point"][1]),
                 caption="Zoom", use_container_width=True)

        strip = exemplar_strip(ann, img_pil)
        if strip:
            st.image(strip, caption="Exemplars", use_container_width=True)

        if cand.get("reason"):
            st.caption(f"Gemini: {cand['reason']}")

        st.write(f"Decision: **{current_decision}**")

        def _decide(decision: str):
            st.session_state.decisions[key] = decision
            save_one_decision(key, decision)
            # Auto-advance
            if st.session_state.cand_idx < len(image_pairs) - 1:
                st.session_state.cand_idx += 1
            elif st.session_state.img_idx < len(all_images) - 1:
                st.session_state.img_idx  += 1
                st.session_state.cand_idx  = 0

        b1, b2 = st.columns(2)
        b3, b4 = st.columns(2)
        if b1.button("✅ Accept",  key="btn_a"): _decide("accept");    st.rerun()
        if b2.button("❌ Reject",  key="btn_r"): _decide("reject");    st.rerun()
        if b3.button("❓ Unsure",  key="btn_u"): _decide("uncertain"); st.rerun()
        if b4.button("⏭ Skip",    key="btn_s"): _decide("skip");      st.rerun()

        st.divider()
        if st.button("❌ Reject ALL remaining"):
            for _, c2 in image_pairs:
                k2 = candidate_key(selected_img, c2["point"])
                if decisions.get(k2) not in ("accept", "reject", "uncertain"):
                    st.session_state.decisions[k2] = "reject"
                    save_one_decision(k2, "reject")
            st.rerun()

# ---------------------------------------------------------------------------
# Grid mode
# ---------------------------------------------------------------------------
else:
    GRID_COLS = 4
    GRID_ROWS = 3
    page_size = GRID_COLS * GRID_ROWS
    n_pages   = max(1, (len(image_pairs) + page_size - 1) // page_size)
    page      = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1) - 1
    page_pairs = image_pairs[page*page_size : (page+1)*page_size]

    ann      = annotations.get(selected_img, {})
    img_pil  = open_image(selected_img)
    existing = [[p[0], p[1]] for p in ann.get("points", [])]

    for row_pairs in [page_pairs[i:i+GRID_COLS] for i in range(0, len(page_pairs), GRID_COLS)]:
        cols = st.columns(GRID_COLS)
        for col, (img_name, cand) in zip(cols, row_pairs):
            key         = candidate_key(img_name, cand["point"])
            dec         = decisions.get(key, "undecided")
            icon        = {"accept":"✅","reject":"❌","uncertain":"❓","skip":"⏭"}.get(dec,"⬜")
            crop_vis    = zoom_crop(draw_with_candidate(img_pil, existing, cand["point"]),
                                    cand["point"][0], cand["point"][1], pad=60)
            with col:
                st.image(crop_vis,
                         caption=f"{icon} ({int(cand['point'][0])},{int(cand['point'][1])})",
                         use_container_width=True)
                if cand.get("reason"):
                    st.caption(cand["reason"][:60])
                b1, b2, b3, b4 = st.columns(4)
                if b1.button("✅", key=f"ga_{key}"):
                    st.session_state.decisions[key] = "accept";    save_one_decision(key, "accept");    st.rerun()
                if b2.button("❌", key=f"gr_{key}"):
                    st.session_state.decisions[key] = "reject";    save_one_decision(key, "reject");    st.rerun()
                if b3.button("❓", key=f"gu_{key}"):
                    st.session_state.decisions[key] = "uncertain"; save_one_decision(key, "uncertain"); st.rerun()
                if b4.button("⏭", key=f"gs_{key}"):
                    st.session_state.decisions[key] = "skip";      save_one_decision(key, "skip");      st.rerun()
