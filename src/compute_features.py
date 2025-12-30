from __future__ import annotations
import csv
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from fontTools.ttLib import TTFont

from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis
from skimage.measure import find_contours

import freetype
from PIL import Image

# -----------------------------
# Glyph sets
# -----------------------------
ROUND_GLYPHS = ["O", "o", "0", "e", "c"]
APERTURE_GLYPHS = ["c", "e", "S", "s", "C"]
STEM_VERT_GLYPHS = ["H", "I", "n", "h", "l"]
STEM_HORZ_GLYPHS = ["E", "H", "e", "t", "T"]
SERIF_PROBE_GLYPHS = ["H", "I", "n", "l", "E", "T"]

# -----------------------------
# Basic font metrics
# -----------------------------
def safe_get(obj, attr: str, default=None):
    return getattr(obj, attr, default) if obj is not None else default

def upm(font: TTFont) -> int:
    head = font["head"] if "head" in font else None
    return int(safe_get(head, "unitsPerEm", 1000) or 1000)

def get_typo_metrics(font: TTFont) -> Dict[str, Optional[float]]:
    os2 = font["OS/2"] if "OS/2" in font else None
    hhea = font["hhea"] if "hhea" in font else None
    post = font["post"] if "post" in font else None

    U = float(upm(font))

    asc = safe_get(os2, "sTypoAscender", None)
    desc = safe_get(os2, "sTypoDescender", None)
    gap = safe_get(os2, "sTypoLineGap", None)
    if asc in (None, 0):
        asc = safe_get(hhea, "ascent", 0)
    if desc in (None, 0):
        desc = safe_get(hhea, "descent", 0)
    if gap in (None, 0):
        gap = safe_get(hhea, "lineGap", 0)

    sxh = safe_get(os2, "sxHeight", None)
    sch = safe_get(os2, "sCapHeight", None)

    italic_angle = safe_get(post, "italicAngle", 0.0)
    try:
        italic_angle = float(italic_angle)
    except Exception:
        italic_angle = 0.0

    weight_class = safe_get(os2, "usWeightClass", None)
    width_class = safe_get(os2, "usWidthClass", None)

    return {
        "upm": float(U),
        "ascender": float(asc) if asc is not None else None,
        "descender": float(desc) if desc is not None else None,
        "line_gap": float(gap) if gap is not None else None,
        "x_height": float(sxh) if (sxh not in (None, 0)) else None,
        "cap_height": float(sch) if (sch not in (None, 0)) else None,
        "italic_angle": float(italic_angle),
        "weight_class": float(weight_class) if weight_class is not None else None,
        "width_class": float(width_class) if width_class is not None else None,
    }

# -----------------------------
# FreeType rasterization
# -----------------------------
def render_glyph_mask_freetype(
    font_path: str,
    ch: str,
    ppem: int = 1024,
    canvas: int = 1024,
    thresh: int = 16,
) -> Optional[np.ndarray]:
    """
    Render a glyph to a square binary mask using FreeType.
    - ppem: pixels-per-em used by FreeType (controls resolution)
    - canvas: output square size
    """
    face = freetype.Face(font_path)
    face.set_char_size(ppem * 64)  # 26.6 fixed point

    try:
        face.load_char(ch, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
    except Exception:
        return None

    bmp = face.glyph.bitmap
    if bmp.width == 0 or bmp.rows == 0:
        return None

    buf = np.array(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.width)

    # Place in square canvas
    out = np.zeros((canvas, canvas), dtype=np.uint8)
    h, w = buf.shape
    if h >= canvas or w >= canvas:
        # Downscale if glyph is bigger than canvas
        scale = min((canvas - 2) / max(1, w), (canvas - 2) / max(1, h))
        buf = np.array(Image.fromarray(buf).resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR))
        h, w = buf.shape

    y0 = (canvas - h) // 2
    x0 = (canvas - w) // 2
    out[y0:y0 + h, x0:x0 + w] = buf

    mask = (out > thresh).astype(np.uint8)
    if mask.sum() < 50:
        return None
    return mask

# -----------------------------
# Morphology helpers
# -----------------------------
def skeleton_and_radii(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    inside = mask.astype(bool)
    skel, dist = medial_axis(inside, return_distance=True)
    skel = skel.astype(np.uint8)
    radii = dist.astype(np.float32) * skel
    return skel, radii

def estimate_stem_thickness(mask: np.ndarray) -> Optional[float]:
    skel, radii = skeleton_and_radii(mask)
    vals = radii[skel > 0]
    vals = vals[vals > 0.75]
    if vals.size < 30:
        return None
    return 2.0 * float(np.median(vals))

def estimate_contrast_ratio(mask: np.ndarray) -> Optional[float]:
    skel, radii = skeleton_and_radii(mask)
    vals = radii[skel > 0]
    vals = vals[vals > 0.75]
    if vals.size < 50:
        return None
    thick = float(np.percentile(vals, 92))
    thin = float(np.percentile(vals, 18))
    if thin <= 1e-6:
        return None
    return thick / thin

def boundary_points(mask: np.ndarray) -> Optional[np.ndarray]:
    cs = find_contours(mask.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=lambda a: a.shape[0])
    return c.astype(np.float32)

def curvature_stats(contour: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if contour is None or contour.shape[0] < 120:
        return None, None, None
    pts = contour
    p_prev = np.roll(pts, 1, axis=0)
    p_next = np.roll(pts, -1, axis=0)

    v1 = pts - p_prev
    v2 = p_next - pts
    n1 = v1 / (np.linalg.norm(v1, axis=1)[:, None] + 1e-6)
    n2 = v2 / (np.linalg.norm(v2, axis=1)[:, None] + 1e-6)

    dot = np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0)
    ang = np.arccos(dot)
    ds = (np.linalg.norm(v1, axis=1) + np.linalg.norm(v2, axis=1)) * 0.5 + 1e-6
    k = np.abs(ang / ds)

    return float(np.mean(k)), float(np.percentile(k, 90)), float(np.mean(k > np.percentile(k, 95)))

def roundness(mask: np.ndarray) -> Optional[float]:
    area = float(mask.sum())
    if area < 200:
        return None
    c = boundary_points(mask)
    if c is None:
        return None
    dif = np.diff(np.vstack([c, c[:1]]), axis=0)
    per = float(np.sum(np.linalg.norm(dif, axis=1)))
    if per <= 1e-6:
        return None
    return float(4.0 * math.pi * area / (per * per))

def stress_angle(mask: np.ndarray) -> Optional[float]:
    c = boundary_points(mask)
    if c is None or c.shape[0] < 160:
        return None

    # SDF-ish thickness proxy near boundary: distance inside
    dist_in = distance_transform_edt(mask.astype(bool)).astype(np.float32)
    ys = np.clip(c[:, 0].astype(int), 0, dist_in.shape[0] - 1)
    xs = np.clip(c[:, 1].astype(int), 0, dist_in.shape[1] - 1)
    thick = dist_in[ys, xs]
    thr = np.percentile(thick, 85)
    sel = c[thick >= thr]
    if sel.shape[0] < 50:
        return None

    pts = sel
    p_prev = np.roll(pts, 1, axis=0)
    p_next = np.roll(pts, -1, axis=0)
    t = p_next - p_prev
    tn = t / (np.linalg.norm(t, axis=1)[:, None] + 1e-6)
    # normal
    n = np.stack([-tn[:, 1], tn[:, 0]], axis=1)  # (dy, dx)

    cov = (n.T @ n) / max(1, n.shape[0])
    w, v = np.linalg.eigh(cov)
    dom = v[:, np.argmax(w)]
    ang = math.degrees(math.atan2(dom[0], dom[1]))
    while ang <= -90:
        ang += 180
    while ang > 90:
        ang -= 180
    return float(ang)

def aperture(mask: np.ndarray) -> Optional[float]:
    c = boundary_points(mask)
    if c is None or c.shape[0] < 120:
        return None
    dif = np.diff(np.vstack([c, c[:1]]), axis=0)
    per = float(np.sum(np.linalg.norm(dif, axis=1)))

    ys, xs = np.where(mask > 0)
    if ys.size < 100:
        return None
    y0, y1 = float(ys.min()), float(ys.max())
    x0, x1 = float(xs.min()), float(xs.max())
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    bbox_per = 2.0 * (bw + bh)

    area = float(mask.sum())
    fill = area / max(1.0, bw * bh)
    per_ratio = per / max(1e-6, bbox_per)

    score = (1.0 - fill) * np.clip(per_ratio / 2.0, 0.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))

def skeleton_endpoints(mask: np.ndarray) -> List[Tuple[int, int]]:
    skel, _ = skeleton_and_radii(mask)
    ys, xs = np.where(skel > 0)
    pts = set(zip(ys.tolist(), xs.tolist()))
    def neigh(y, x):
        out = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                if (y + dy, x + dx) in pts:
                    out += 1
        return out
    return [(y, x) for (y, x) in pts if neigh(y, x) == 1]

def terminal_angle_mean_abs(mask: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    skel, _ = skeleton_and_radii(mask)
    ys, xs = np.where(skel > 0)
    sk = set(zip(ys.tolist(), xs.tolist()))
    def neighbors(y, x):
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                p = (y + dy, x + dx)
                if p in sk:
                    out.append(p)
        return out

    endpoints = [(y, x) for (y, x) in sk if len(neighbors(y, x)) == 1]
    if len(endpoints) < 2:
        return None, float(len(endpoints))

    angles = []
    for (y0, x0) in endpoints:
        prev = (y0, x0)
        cur = neighbors(y0, x0)[0]
        for _ in range(6):
            nxts = [p for p in neighbors(*cur) if p != prev]
            if not nxts:
                break
            prev, cur = cur, nxts[0]
        dy = float(cur[0] - y0)
        dx = float(cur[1] - x0)
        if abs(dx) + abs(dy) < 1e-6:
            continue
        ang = math.degrees(math.atan2(dy, dx))
        while ang <= -90:
            ang += 180
        while ang > 90:
            ang -= 180
        angles.append(abs(ang))
    if not angles:
        return None, float(len(endpoints))
    return float(np.mean(angles)), float(len(endpoints))

def serifness(mask: np.ndarray) -> Optional[float]:
    skel, radii = skeleton_and_radii(mask)
    ys, xs = np.where(skel > 0)
    sk = set(zip(ys.tolist(), xs.tolist()))
    def neighbors(y, x):
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                p = (y + dy, x + dx)
                if p in sk:
                    out.append(p)
        return out
    endpoints = [(y, x) for (y, x) in sk if len(neighbors(y, x)) == 1]
    if len(endpoints) < 2:
        return 0.0

    flare = []
    for (y0, x0) in endpoints:
        r0 = float(radii[y0, x0])
        if r0 <= 0.75:
            continue
        prev = (y0, x0)
        cur = neighbors(y0, x0)[0]
        for _ in range(10):
            nxts = [p for p in neighbors(*cur) if p != prev]
            if not nxts:
                break
            prev, cur = cur, nxts[0]
        r_in = float(radii[cur[0], cur[1]])
        if r_in <= 0.75:
            continue
        flare.append(r0 / max(1e-6, r_in))
    if len(flare) < 2:
        return None
    med = float(np.median(flare))
    score = (med - 1.0) / 0.5  # 1.5 -> 1.0
    return float(np.clip(score, 0.0, 1.0))

def symmetry_lr(mask: np.ndarray) -> Optional[float]:
    m = mask.astype(bool)
    if m.sum() < 200:
        return None
    mir = np.fliplr(m)
    inter = np.logical_and(m, mir).sum()
    union = np.logical_or(m, mir).sum()
    return float(inter / union) if union else None


def median_or_none(xs: List[Optional[float]]) -> Optional[float]:
    ys = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(ys)) if ys else None


# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(font_path: str, canvas: int = 1024, ppem: int = 1024, debug: bool = False) -> Dict[str, Optional[float]]:
    font = TTFont(str(font_path), recalcBBoxes=False, recalcTimestamp=False)
    U = float(upm(font))

    out: Dict[str, Optional[float]] = {}
    out["font_path"] = str(font_path)

    m = get_typo_metrics(font)
    out.update(m)

    out["x_height_ratio"] = (m["x_height"] / U) if m["x_height"] is not None else None
    out["cap_height_ratio"] = (m["cap_height"] / U) if m["cap_height"] is not None else None
    out["ascender_ratio"] = (m["ascender"] / U) if m["ascender"] is not None else None
    out["descender_ratio"] = (m["descender"] / U) if m["descender"] is not None else None

    # Morphology aggregates
    vert_stems = []
    horz_stems = []
    contrasts = []
    stresses = []
    apertures = []
    roundnesses = []
    curv_mean = []
    curv_p90 = []
    curv_hi = []
    terminals = []
    term_counts = []
    serifs = []
    symO = []
    symH = []

    def msk(ch: str) -> Optional[np.ndarray]:
        return render_glyph_mask_freetype(str(font_path), ch, ppem=ppem, canvas=canvas)

    # Debug: verify masks exist
    if debug:
        for ch in ["H", "O", "o", "e", "c", "n"]:
            mm = msk(ch)
            print(f"[debug] {ch}: {'None' if mm is None else int(mm.sum())} ink_pixels")

    # stems
    for ch in STEM_VERT_GLYPHS:
        mm = msk(ch)
        if mm is None:
            continue
        t = estimate_stem_thickness(mm)
        if t is not None:
            vert_stems.append(t)

    for ch in STEM_HORZ_GLYPHS:
        mm = msk(ch)
        if mm is None:
            continue
        t = estimate_stem_thickness(mm)
        if t is not None:
            horz_stems.append(t)

    # rounds/aperture/curvature/terminals/serifs/symmetry
    probe_chars = sorted(set(ROUND_GLYPHS + APERTURE_GLYPHS + SERIF_PROBE_GLYPHS + ["O", "H"]))
    for ch in probe_chars:
        mm = msk(ch)
        if mm is None:
            continue

        if ch in ROUND_GLYPHS:
            r = roundness(mm)
            if r is not None:
                roundnesses.append(r)
            c = estimate_contrast_ratio(mm)
            if c is not None:
                contrasts.append(c)
            s = stress_angle(mm)
            if s is not None:
                stresses.append(s)

        if ch in APERTURE_GLYPHS:
            a = aperture(mm)
            if a is not None:
                apertures.append(a)

        cont = boundary_points(mm)
        km, kp, kh = curvature_stats(cont)
        if km is not None: curv_mean.append(km)
        if kp is not None: curv_p90.append(kp)
        if kh is not None: curv_hi.append(kh)

        ta, tc = terminal_angle_mean_abs(mm)
        if ta is not None: terminals.append(ta)
        if tc is not None: term_counts.append(tc)

        if ch in SERIF_PROBE_GLYPHS:
            sf = serifness(mm)
            if sf is not None:
                serifs.append(sf)

        if ch == "O":
            so = symmetry_lr(mm)
            if so is not None: symO.append(so)
        if ch == "H":
            sh = symmetry_lr(mm)
            if sh is not None: symH.append(sh)

    out["vertical_stem_px"] = median_or_none(vert_stems)
    out["horizontal_stem_px"] = median_or_none(horz_stems)

    # normalize stems by H pixel height in this canvas (stable across fonts at same ppem/canvas)
    Hm = msk("H")
    if Hm is not None:
        ys, _ = np.where(Hm > 0)
        Hh = float(ys.max() - ys.min() + 1) if ys.size else None
    else:
        Hh = None

    out["vertical_stem_rel_to_H"] = (out["vertical_stem_px"] / Hh) if (Hh and out["vertical_stem_px"] is not None) else None
    out["horizontal_stem_rel_to_H"] = (out["horizontal_stem_px"] / Hh) if (Hh and out["horizontal_stem_px"] is not None) else None

    out["contrast_ratio"] = median_or_none(contrasts)
    out["stress_angle_deg"] = median_or_none(stresses)
    out["aperture_score"] = median_or_none(apertures)
    out["roundness_score"] = median_or_none(roundnesses)

    out["curvature_mean"] = median_or_none(curv_mean)
    out["curvature_p90"] = median_or_none(curv_p90)
    out["sharpness_high_curv_frac"] = median_or_none(curv_hi)

    out["terminal_angle_mean_abs_deg"] = median_or_none(terminals)
    out["terminal_count_est"] = median_or_none(term_counts)

    out["serifness_score"] = median_or_none(serifs)
    out["symmetry_O_lr"] = median_or_none(symO)
    out["symmetry_H_lr"] = median_or_none(symH)

    if contrasts:
        out["contrast_iqr"] = float(np.percentile(contrasts, 75) - np.percentile(contrasts, 25))
    else:
        out["contrast_iqr"] = None
    if curv_mean:
        out["curvature_iqr"] = float(np.percentile(curv_mean, 75) - np.percentile(curv_mean, 25))
    else:
        out["curvature_iqr"] = None

    return out

stats = {}
def main():
    with open("../dataset/curated_fonts.csv", "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            font_path = row["font_path"]
            print("Analyzing font: ", font_path)
            features = extract_features(font_path)
            for key in features:
                if features[key] is None:
                    if key in stats:
                        stats[key] += 1
                    else:
                        stats[key] = 1
            if i > 20:
                break

    for key in stats:
        print(key, stats[key])

if __name__ == "__main__":
    main()