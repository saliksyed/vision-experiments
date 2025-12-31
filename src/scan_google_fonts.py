#!/usr/bin/env python3
"""
Scan a local clone of https://github.com/google/fonts and produce:

1) A structured index of "usable" fonts (TTF/OTF) with key metrics/features.
2) De-duplication / near-dup clustering using a robust outline-based fingerprint.
3) A curated subset (balanced across serif/sans/display/script-ish proxies).

USAGE
-----
python scan_google_fonts.py /path/to/google/fonts \
  --out_dir ./gf_dataset \
  --min_glyphs 200 \
  --require_basic_latin \
  --target_families 400

OUTPUTS
-------
out_dir/
  fonts_index.csv             # all scanned fonts + features
  usable_fonts.csv            # filtered "usable" fonts
  dedup_clusters.json         # clusters of near-identical fonts
  curated_fonts.csv           # selected subset (one per dedup cluster, balanced)
  curated_copy_manifest.csv   # copy instructions (src -> dst)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import os
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.boundsPen import BoundsPen
from constants import GOOGLE_FONTS_DIR, GOOGLE_FONTS_METADATA_DIR, BASE_DATA_DIR

# ----------------------------
# Helpers: parsing METADATA.pb
# ----------------------------

_RE_FAMILY_NAME = re.compile(r'family:\s*"([^"]+)"')
_RE_DESIGNER = re.compile(r'designer:\s*"([^"]+)"')
_RE_CATEGORY = re.compile(r'category:\s*"([^"]+)"')
_RE_SUBSETS = re.compile(r'subsets:\s*"([^"]+)"')
_RE_LICENSE = re.compile(r'license:\s*"([^"]+)"')
_RE_FONT_BLOCK = re.compile(r'fonts\s*{\s*(.*?)\s*}', re.DOTALL)
_RE_STYLE = re.compile(r'style:\s*"([^"]+)"')
_RE_WEIGHT = re.compile(r'weight:\s*(\d+)')
_RE_FILENAME = re.compile(r'filename:\s*"([^"]+)"')
_RE_FULL_NAME = re.compile(r'full_name:\s*"([^"]+)"')


def parse_metadata_pb_text(path: Path) -> Dict[str, object]:
    """
    Google Fonts' METADATA.pb in the repo is typically protobuf text format.
    We parse only the fields we care about, with regex (no protobuf dependency).
    """
    data: Dict[str, object] = {
        "family": None,
        "designer": None,
        "category": None,
        "subsets": [],
        "license": None,
        "fonts": [],  # list of dicts {filename, style, weight, full_name}
    }
    if not path.exists():
        return data

    txt = path.read_text(encoding="utf-8", errors="ignore")
    m = _RE_FAMILY_NAME.search(txt)
    if m:
        data["family"] = m.group(1)
    m = _RE_DESIGNER.search(txt)
    if m:
        data["designer"] = m.group(1)
    m = _RE_CATEGORY.search(txt)
    if m:
        data["category"] = m.group(1)
    data["subsets"] = _RE_SUBSETS.findall(txt)
    m = _RE_LICENSE.search(txt)
    if m:
        data["license"] = m.group(1)

    fonts: List[Dict[str, object]] = []
    for block_m in _RE_FONT_BLOCK.finditer(txt):
        block = block_m.group(1)
        fn = _RE_FILENAME.search(block)
        st = _RE_STYLE.search(block)
        wt = _RE_WEIGHT.search(block)
        full = _RE_FULL_NAME.search(block)
        fonts.append(
            {
                "filename": fn.group(1) if fn else None,
                "style": st.group(1) if st else None,
                "weight": int(wt.group(1)) if wt else None,
                "full_name": full.group(1) if full else None,
            }
        )
    data["fonts"] = fonts
    return data


# ---------------------------------------
# Font metrics + outline-based fingerprint
# ---------------------------------------

BASIC_LATIN_CODEPOINTS = list(range(0x20, 0x7F))  # space .. ~
CONTROL_CHARS = ["H", "O", "n", "o", "a", "e", "s"]  # can degrade gracefully
FALLBACK_CHARS = ["I", "l", "h", "c", "r", "u", "m", "x", "p", "q", "t", "S", "E"]


@dataclass
class FontRow:
    # identity
    font_path: str
    family_dir: str
    family_name: str
    designer: str
    category: str
    license: str
    subsets: str

    # tables/metrics
    upm: int
    weight_class: int
    width_class: int
    italic_angle: float
    ascender: int
    descender: int
    line_gap: int
    x_height: Optional[int]
    cap_height: Optional[int]

    # coverage / usability
    glyph_count: int
    has_basic_latin: bool
    is_variable: bool

    # outline signature
    signature_sha256: str
    simhash64: int  # for near-dup clustering


def safe_getattr(obj, name: str, default=None):
    return getattr(obj, name, default) if obj is not None else default


def get_best_cmap(font: TTFont) -> Dict[int, str]:
    cmap = font.getBestCmap()
    return cmap or {}


def glyph_name_for_char(font: TTFont, ch: str) -> Optional[str]:
    cmap = get_best_cmap(font)
    return cmap.get(ord(ch))


def glyph_bounds(font: TTFont, glyph_name: str) -> Optional[Tuple[int, int, int, int]]:
    glyph_set = font.getGlyphSet()
    if glyph_name not in glyph_set:
        return None
    pen = BoundsPen(glyph_set)
    glyph_set[glyph_name].draw(pen)
    return pen.bounds


def glyph_height(font: TTFont, glyph_name: str) -> Optional[int]:
    b = glyph_bounds(font, glyph_name)
    if not b:
        return None
    xMin, yMin, xMax, yMax = b
    return yMax - yMin


def is_variable_font(font: TTFont) -> bool:
    return "fvar" in font


def has_basic_latin(font: TTFont) -> bool:
    cmap = get_best_cmap(font)
    # require that most printable ASCII map (excluding control chars)
    present = 0
    for cp in BASIC_LATIN_CODEPOINTS:
        if cp in cmap:
            present += 1
    return present >= int(0.90 * len(BASIC_LATIN_CODEPOINTS))


def pick_signature_glyphs(font: TTFont, max_glyphs: int = 7) -> List[str]:
    """
    Choose a stable glyph list for signature hashing. Prefer CONTROL_CHARS;
    fallback to any available among FALLBACK_CHARS, then any cmap glyphs.
    """
    chosen: List[str] = []
    for ch in CONTROL_CHARS:
        gn = glyph_name_for_char(font, ch)
        if gn and gn not in chosen:
            chosen.append(gn)
        if len(chosen) >= max_glyphs:
            return chosen

    for ch in FALLBACK_CHARS:
        gn = glyph_name_for_char(font, ch)
        if gn and gn not in chosen:
            chosen.append(gn)
        if len(chosen) >= max_glyphs:
            return chosen

    # last resort: first few glyphs by glyphOrder (skip .notdef)
    for gn in font.getGlyphOrder():
        if gn == ".notdef":
            continue
        chosen.append(gn)
        if len(chosen) >= max_glyphs:
            break
    return chosen

def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    if _is_num(x):
        return float(x)
    if isinstance(x, str):
        # Some backends may provide numeric strings; most of your errors are non-numeric.
        try:
            return float(x)
        except Exception:
            return None
    return None

def _flatten_pts(pts):
    """
    Yield a flat stream of numeric coordinates from RecordingPen points.

    Handles:
      - (x, y, x, y, ...)
      - ((x,y), (x,y), ...)
      - mixed nesting
      - None values
      - non-numeric strings / tokens (skipped)
    """
    # pts might be empty tuple
    for v in pts:
        if v is None:
            continue

        # Common case: (x, y) point
        if isinstance(v, (tuple, list)) and len(v) == 2:
            fx = _to_float(v[0])
            fy = _to_float(v[1])
            if fx is not None and fy is not None:
                yield fx
                yield fy
            # else skip silently
            continue

        # Nested list/tuple of points (e.g., ((x,y),(x,y),...))
        if isinstance(v, (tuple, list)):
            # recurse
            for u in _flatten_pts(v):
                yield u
            continue

        # Scalar-ish
        f = _to_float(v)
        if f is not None:
            yield f
        # else skip silently (strings like "n", "Comp_H", etc.)


def normalize_recording(rec, upm: int):
    q = 2048.0
    out = []
    upm_f = float(upm) if upm else 1000.0
    for op, pts in rec:
        ints = []
        for v in _flatten_pts(pts):
            ints.append(int(round((v / upm_f) * q)))
        out.append((op, tuple(ints)))
    return out

def outline_signature(font: TTFont) -> Tuple[str, int]:
    """
    Returns:
      - signature_sha256: strong hash for exact duplicates
      - simhash64: locality-sensitive hash for near duplicates
    """
    upm = int(safe_getattr(font["head"], "unitsPerEm", 1000))
    glyph_set = font.getGlyphSet()
    glyphs = pick_signature_glyphs(font, max_glyphs=7)

    token_counts: Dict[int, int] = {}

    h = hashlib.sha256()
    for gn in glyphs:
        if gn not in glyph_set:
            continue
        pen = RecordingPen()
        glyph_set[gn].draw(pen)
        norm = normalize_recording(pen.value, upm=upm)

        # feed exact signature bytes
        h.update(gn.encode("utf-8"))
        for op, ints in norm:
            h.update(op.encode("utf-8"))
            # ints are already normalized; include as bytes
            h.update((",".join(map(str, ints))).encode("utf-8"))
            h.update(b";")

            # also create cheap tokens for simhash
            # token = hash(op + first few coords) (keeps it stable)
            key = f"{op}:{ints[:8]}"
            tok = int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big")
            token_counts[tok] = token_counts.get(tok, 0) + 1
        h.update(b"|")

    sig_sha = h.hexdigest()
    sim = simhash64(token_counts)
    return sig_sha, sim


def simhash64(token_counts: Dict[int, int]) -> int:
    """
    Weighted SimHash over 64 bits. token_counts keys are 64-bit-ish ints.
    """
    v = [0] * 64
    for tok, w in token_counts.items():
        # ensure 64-bit token
        x = tok & ((1 << 64) - 1)
        for i in range(64):
            bit = (x >> i) & 1
            v[i] += w if bit else -w

    out = 0
    for i in range(64):
        if v[i] >= 0:
            out |= (1 << i)
    return out


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def extract_font_row(font_path: Path) -> FontRow:
    font = TTFont(str(font_path), recalcBBoxes=False, recalcTimestamp=False)

    head = font["head"] if "head" in font else None
    os2 = font["OS/2"] if "OS/2" in font else None
    hhea = font["hhea"] if "hhea" in font else None
    post = font["post"] if "post" in font else None

    upm = int(safe_getattr(head, "unitsPerEm", 1000))

    # prefer OS/2 typo metrics; fallback to hhea
    asc = safe_getattr(os2, "sTypoAscender", None)
    desc = safe_getattr(os2, "sTypoDescender", None)
    gap = safe_getattr(os2, "sTypoLineGap", None)
    if asc in (None, 0):
        asc = safe_getattr(hhea, "ascent", 0)
    if desc in (None, 0):
        desc = safe_getattr(hhea, "descent", 0)
    if gap in (None, 0):
        gap = safe_getattr(hhea, "lineGap", 0)

    # optional heights
    sxh = safe_getattr(os2, "sxHeight", None)
    sch = safe_getattr(os2, "sCapHeight", None)

    # geometry fallback
    gn_x = glyph_name_for_char(font, "x") or glyph_name_for_char(font, "n")
    gn_H = glyph_name_for_char(font, "H") or glyph_name_for_char(font, "I")
    xh = sxh if (sxh not in (None, 0)) else (glyph_height(font, gn_x) if gn_x else None)
    ch = sch if (sch not in (None, 0)) else (glyph_height(font, gn_H) if gn_H else None)

    weight_class = int(safe_getattr(os2, "usWeightClass", 0) or 0)
    width_class = int(safe_getattr(os2, "usWidthClass", 0) or 0)
    italic_angle = float(safe_getattr(post, "italicAngle", 0.0) or 0.0)

    glyph_count = len(font.getGlyphOrder())
    basic_latin = has_basic_latin(font)
    variable = is_variable_font(font)

    sig_sha, sim = outline_signature(font)

    # Fill metadata later in the caller (from directory METADATA.pb)
    return FontRow(
        font_path=str(font_path),
        family_dir=str(font_path.parent),
        family_name="",
        designer="",
        category="",
        license="",
        subsets="",
        upm=upm,
        weight_class=weight_class,
        width_class=width_class,
        italic_angle=italic_angle,
        ascender=int(asc or 0),
        descender=int(desc or 0),
        line_gap=int(gap or 0),
        x_height=int(xh) if xh is not None else None,
        cap_height=int(ch) if ch is not None else None,
        glyph_count=glyph_count,
        has_basic_latin=basic_latin,
        is_variable=variable,
        signature_sha256=sig_sha,
        simhash64=sim,
    )


# -----------------------
# Filtering + de-dup logic
# -----------------------

def is_probably_usable(row: FontRow, min_glyphs: int, require_basic_latin: bool) -> bool:
    if row.glyph_count < min_glyphs:
        return False
    if require_basic_latin and not row.has_basic_latin:
        return False
    # Basic sanity: UPM should be reasonable
    if row.upm < 256 or row.upm > 4096:
        return False
    return True


def cluster_near_duplicates(rows: List[FontRow], ham_threshold: int = 3) -> List[List[int]]:
    """
    Cluster near-identical fonts using SimHash Hamming distance.
    This is approximate, but works well for forks/renames / byte-different-but-same outlines.
    Steps:
      1) Bucket by high 16 bits to reduce comparisons
      2) Union-find within buckets using Hamming <= threshold
    """
    # bucket by top bits
    buckets: Dict[int, List[int]] = {}
    for i, r in enumerate(rows):
        key = (r.simhash64 >> 48) & 0xFFFF
        buckets.setdefault(key, []).append(i)

    parent = list(range(len(rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for idxs in buckets.values():
        # O(k^2) within bucket; buckets are small in practice
        for a_i in range(len(idxs)):
            for b_i in range(a_i + 1, len(idxs)):
                a = idxs[a_i]
                b = idxs[b_i]
                if hamming64(rows[a].simhash64, rows[b].simhash64) <= ham_threshold:
                    union(a, b)

    clusters: Dict[int, List[int]] = {}
    for i in range(len(rows)):
        clusters.setdefault(find(i), []).append(i)

    # sort cluster members by "preference" (we'll pick best representative later)
    out = []
    for members in clusters.values():
        out.append(members)
    out.sort(key=len, reverse=True)
    return out


def score_representative(row: FontRow) -> Tuple[int, int, int]:
    """
    Higher is better. We prefer:
      - non-variable (optional choice; you can flip this)
      - has x_height & cap_height
      - larger glyph_count
    """
    has_heights = int(row.x_height is not None) + int(row.cap_height is not None)
    return (has_heights, row.glyph_count, -int(row.is_variable))


def pick_cluster_representatives(rows: List[FontRow], clusters: List[List[int]]) -> List[int]:
    reps: List[int] = []
    for members in clusters:
        members_sorted = sorted(members, key=lambda i: score_representative(rows[i]), reverse=True)
        reps.append(members_sorted[0])
    return reps


# -----------------------
# Curation (balanced subset)
# -----------------------

def normalize_category(cat: str) -> str:
    c = (cat or "").strip().lower()
    # Google Fonts categories: "SANS_SERIF", "SERIF", "DISPLAY", "HANDWRITING", "MONOSPACE"
    if "sans" in c:
        return "sans"
    if "serif" in c:
        return "serif"
    if "mono" in c:
        return "mono"
    if "hand" in c or "script" in c:
        return "script"
    if "display" in c:
        return "display"
    return "other"


def curate(
    rows: List[FontRow],
    rep_idxs: List[int],
    target_families: int = 400,
) -> List[int]:
    """
    Pick a curated set of representatives (one per near-dup cluster) with rough balance
    across category and across x-height bins.

    This is a heuristic "boutique / luxury-adjacent" starter set:
    - keep serif/sans/display/script/mono balanced
    - prefer fonts with sane x-height + cap-height values
    - prefer moderate italic angles (not extreme)
    """
    reps = [rows[i] for i in rep_idxs]

    # Build bins
    def xh_ratio(r: FontRow) -> Optional[float]:
        if r.x_height is None:
            return None
        return float(r.x_height) / float(r.upm)

    # category buckets
    buckets: Dict[str, List[int]] = {}
    for idx in rep_idxs:
        c = normalize_category(rows[idx].category)
        buckets.setdefault(c, []).append(idx)

    # target proportions (tweak as you like)
    plan = {
        "serif": 0.28,
        "sans": 0.35,
        "display": 0.18,
        "script": 0.07,
        "mono": 0.07,
        "other": 0.05,
    }

    # x-height bins encourage variety (low/editorial vs high/tech)
    xbins = [0.38, 0.45, 0.52, 0.60]  # boundaries; yields 5 bins

    def xbin(r: FontRow) -> int:
        xr = xh_ratio(r)
        if xr is None:
            return -1
        for j, b in enumerate(xbins):
            if xr < b:
                return j
        return len(xbins)

    # within each category, rank by: has heights, glyph_count, "editorial-ish" x-height diversity
    selected: List[int] = []
    for cat, frac in plan.items():
        cand = buckets.get(cat, [])
        if not cand:
            continue
        k = max(1, int(round(target_families * frac)))
        # stratify by x-height bin
        by_bin: Dict[int, List[int]] = {}
        for idx in cand:
            by_bin.setdefault(xbin(rows[idx]), []).append(idx)

        # Sort each bin by representative score
        for b, lst in by_bin.items():
            lst.sort(key=lambda i: score_representative(rows[i]), reverse=True)

        # round-robin pull from bins to preserve diversity
        pulled = []
        bins_order = sorted(by_bin.keys(), key=lambda t: (t == -1, t))  # prefer known x-height
        while len(pulled) < k and any(by_bin.get(b) for b in bins_order):
            for b in bins_order:
                lst = by_bin.get(b)
                if lst:
                    pulled.append(lst.pop(0))
                    if len(pulled) >= k:
                        break
        selected.extend(pulled)

    # if underfilled, top off from remaining reps
    selected_set = set(selected)
    if len(selected) < target_families:
        remaining = [i for i in rep_idxs if i not in selected_set]
        remaining.sort(key=lambda i: score_representative(rows[i]), reverse=True)
        selected.extend(remaining[: (target_families - len(selected))])

    # cap
    return selected[:target_families]


# -----------------------
# IO
# -----------------------

def write_csv(path: Path, rows: List[FontRow]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def write_clusters(path: Path, rows: List[FontRow], clusters: List[List[int]]):
    payload = []
    for members in clusters:
        payload.append(
            {
                "size": len(members),
                "members": [
                    {
                        "font_path": rows[i].font_path,
                        "family_name": rows[i].family_name,
                        "category": rows[i].category,
                        "license": rows[i].license,
                        "signature_sha256": rows[i].signature_sha256,
                        "simhash64": rows[i].simhash64,
                    }
                    for i in members
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_copy_manifest(curated_rows: List[FontRow], dst_root: Path) -> List[Dict[str, str]]:
    manifest: List[Dict[str, str]] = []
    for r in curated_rows:
        src = Path(r.font_path)
        # Destination path: category/family/fontfile
        cat = normalize_category(r.category)
        family = sanitize_name(r.family_name or src.parent.name)
        dst = dst_root / cat / family / src.name
        manifest.append({"src": str(src), "dst": str(dst)})
    return manifest


def sanitize_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-\s\.]+", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:120] if len(s) > 120 else s


def apply_copy_manifest(manifest: List[Dict[str, str]], dry_run: bool = True):
    for item in manifest:
        src = Path(item["src"])
        dst = Path(item["dst"])
        if dry_run:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# -----------------------
# Main scan
# -----------------------

def find_family_dirs(gf_root: Path) -> List[Path]:
    """
    Google fonts repo has fonts/ofl/<family>, fonts/apache/<family>, fonts/ufl/<family>.
    We'll find all directories containing a METADATA.pb or at least font files.
    """
    family_dirs: List[Path] = []
    for license_dir in [gf_root / "ofl", gf_root / "apache", gf_root / "ufl"]:
        if not license_dir.exists():
            continue
        for d in license_dir.iterdir():
            if d.is_dir():
                # heuristic: contains ttf/otf or METADATA.pb
                has_font = any(p.suffix.lower() in (".ttf", ".otf") for p in d.iterdir() if p.is_file())
                if has_font or (d / "METADATA.pb").exists():
                    family_dirs.append(d)
    return family_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_glyphs", type=int, default=200)
    ap.add_argument("--require_basic_latin", action="store_true")
    ap.add_argument("--ham_threshold", type=int, default=3, help="SimHash Hamming threshold for near-dup clustering (2-4 typical).")
    ap.add_argument("--target_families", type=int, default=400, help="How many representative families to curate.")
    ap.add_argument("--copy_to", type=str, default="", help="If set, generate a copy manifest into this directory (and optionally copy).")
    ap.add_argument("--do_copy", action="store_true", help="Actually copy fonts per manifest (otherwise dry-run).")
    args = ap.parse_args()
    base_data_dir = Path(BASE_DATA_DIR).expanduser().resolve()
    if not os.path.exists(GOOGLE_FONTS_DIR):
        os.makedirs(base_data_dir, exist_ok=True)
        google_fonts_dir = Path(GOOGLE_FONTS_DIR).expanduser().resolve()
        subprocess.run(["git", "clone", "--depth=1", "https://github.com/google/fonts.git", str(google_fonts_dir)])

    gf_root = Path(GOOGLE_FONTS_DIR).expanduser().resolve()
    out_dir = Path(GOOGLE_FONTS_METADATA_DIR).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Scanning Google Fonts repo...")
    family_dirs = find_family_dirs(gf_root)
    print(f"Found {len(family_dirs)} family directories under {gf_root}")

    all_rows: List[FontRow] = []

    for fam_dir in family_dirs:
        meta = parse_metadata_pb_text(fam_dir / "METADATA.pb")
        fam_name = str(meta.get("family") or fam_dir.name)
        designer = str(meta.get("designer") or "")
        category = str(meta.get("category") or "")
        license_name = str(meta.get("license") or fam_dir.parent.name.upper())
        subsets = ",".join(list(meta.get("subsets") or []))

        # Collect font binaries
        font_files = [p for p in fam_dir.iterdir() if p.is_file() and p.suffix.lower() in (".ttf", ".otf")]
        if not font_files:
            continue

        for fp in font_files:
            try:
                row = extract_font_row(fp)
            except Exception as e:
                print(f"SKIP (parse error): {fp} :: {e}")
                continue

            row.family_name = fam_name
            row.designer = designer
            row.category = category
            row.license = license_name
            row.subsets = subsets
            all_rows.append(row)

    # Write full index
    if all_rows:
        write_csv(out_dir / "fonts_index.csv", all_rows)
        print(f"Wrote: {out_dir / 'fonts_index.csv'} ({len(all_rows)} rows)")
    else:
        print("No fonts found / parsed. Check the path you passed in.")
        return

    # Filter usable
    usable = [r for r in all_rows if is_probably_usable(r, args.min_glyphs, args.require_basic_latin)]
    write_csv(out_dir / "usable_fonts.csv", usable)
    print(f"Wrote: {out_dir / 'usable_fonts.csv'} ({len(usable)} rows)")

    # De-dup / near-dup clustering on usable set
    clusters = cluster_near_duplicates(usable, ham_threshold=args.ham_threshold)
    write_clusters(out_dir / "dedup_clusters.json", usable, clusters)
    print(f"Wrote: {out_dir / 'dedup_clusters.json'} ({len(clusters)} clusters)")

    # Pick one representative per cluster
    rep_idxs = pick_cluster_representatives(usable, clusters)
    reps = [usable[i] for i in rep_idxs]

    # Curate a balanced set of representatives
    curated_rep_idxs = curate(usable, rep_idxs=rep_idxs, target_families=args.target_families)
    curated = [usable[i] for i in curated_rep_idxs]
    write_csv(out_dir / "curated_fonts.csv", curated)
    print(f"Wrote: {out_dir / 'curated_fonts.csv'} ({len(curated)} rows)")

    # Copy manifest (optional)
    if args.copy_to:
        dst_root = Path(args.copy_to).expanduser().resolve()
        manifest = make_copy_manifest(curated, dst_root=dst_root)
        man_path = out_dir / "curated_copy_manifest.csv"
        with man_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["src", "dst"])
            w.writeheader()
            w.writerows(manifest)
        print(f"Wrote: {man_path}")

        if args.do_copy:
            apply_copy_manifest(manifest, dry_run=False)
            print(f"Copied curated fonts into: {dst_root}")
        else:
            print("Dry-run: not copying files (pass --do_copy to actually copy).")


if __name__ == "__main__":
    main()
