#!/usr/bin/env python3
"""Extract burned-in Chinese dialogue subtitles from vertical short videos.

Optimized for Douyin-style hardsubs:
  - coarse pixel-diff on the subtitle band (efficiency)
  - densify only across change windows
  - full-frame macOS Vision OCR
  - keep lines by normalized Y position (dialogue band), not grayscale crops

Examples:
  python helpers/extract_burned_subs.py video.mp4
  python helpers/extract_burned_subs.py video.mp4 --edit-dir ./work
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import Vision
    import objc
    from Foundation import NSURL

    _VISION_OK = True
except Exception:
    _VISION_OK = False

UI_NOISE = re.compile(
    r"(关注|已关注|抖音|推荐|直播|点击|主页|分享|评论|点赞|收藏|"
    r"放大镜|搜索|合集|相关搜索|广告|每人限用|限用一次|限用一)"
)

try:
    from domain_lexicon import is_ocr_garbage
except ImportError:

    def is_ocr_garbage(text: str) -> bool:  # type: ignore
        return len((text or "").strip()) < 2


@dataclass
class OcrHit:
    text: str
    conf: float
    y: float  # bottom-left origin, normalized
    h: float
    w: float


@dataclass
class Cue:
    start: float
    end: float
    text: str
    conf: float
    uncertain: bool = False


def probe_duration(video: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        text=True,
    ).strip()
    return float(out)


def probe_size(video: Path) -> tuple[int, int]:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video),
        ],
        text=True,
    ).strip()
    w, h = out.split(",")
    return int(w), int(h)


def extract_frame(video: Path, t: float, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_path),
        ],
        check=True,
    )
    return out_path


def band_signature(
    frame_path: Path, y0_ratio: float = 0.62, y1_ratio: float = 0.82
) -> np.ndarray:
    """Signature of the dialogue band (top-origin ratios)."""
    img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to read frame: {frame_path}")
    h, w = img.shape[:2]
    y0 = int(h * y0_ratio)
    y1 = int(h * y1_ratio)
    band = img[y0:y1, int(w * 0.08) : int(w * 0.92)]
    small = cv2.resize(band, (64, 24), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def sig_changed(a: np.ndarray, b: np.ndarray, thresh: float = 10.0) -> bool:
    return float(np.mean(np.abs(a - b))) >= thresh


def vision_ocr(frame_path: Path) -> list[OcrHit]:
    if not _VISION_OK:
        raise RuntimeError("macOS Vision / pyobjc not available")
    url = NSURL.fileURLWithPath_(str(frame_path.resolve()))
    handler = objc.lookUpClass("VNImageRequestHandler").alloc().initWithURL_options_(
        url, None
    )
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(True)
    try:
        req.setRecognitionLanguages_(["zh-Hans", "zh-Hant", "en-US"])
    except Exception:
        pass
    ok, _err = handler.performRequests_error_([req], None)
    if not ok:
        return []
    hits: list[OcrHit] = []
    for obs in req.results() or []:
        bb = obs.boundingBox()
        y = float(bb.origin.y)
        h = float(bb.size.height)
        w = float(bb.size.width)
        cand = obs.topCandidates_(1)
        if not cand:
            continue
        text = str(cand[0].string() or "").strip()
        conf = float(cand[0].confidence())
        if text:
            hits.append(OcrHit(text=text, conf=conf, y=y, h=h, w=w))
    return hits


def normalize_text(text: str) -> str:
    text = text.strip().replace(" ", "").replace("\n", "")
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"[“”\"'‘’]", "", text)
    return text


def pick_dialogue_line(
    hits: list[OcrHit],
    y_min: float = 0.25,
    y_max: float = 0.40,
) -> tuple[str, float]:
    """Pick burned-in dialogue using Vision bbox Y (bottom-left origin).

    If multiple dialogue-band lines exist (stacked subs), join top→bottom
    reading order after filtering UI chrome.
    """
    band: list[tuple[float, str, float, float]] = []  # y_c, text, conf, w
    for hit in hits:
        text = normalize_text(hit.text)
        if len(text) < 2:
            continue
        y_c = hit.y + hit.h / 2.0
        if not (y_min <= y_c <= y_max):
            continue
        if UI_NOISE.search(text):
            continue
        # brand watermarks near dialogue band
        if re.fullmatch(r"沐.?[沄运]?尚品|肽.?简品", text):
            continue
        han = len(re.findall(r"[\u4e00-\u9fff]", text))
        if han == 0 and not re.search(r"\d", text):
            continue
        if han <= 2 and hit.conf < 0.6:
            continue
        band.append((y_c, text, hit.conf, hit.w))
    if not band:
        return "", 0.0

    # Stacked subtitle lines: higher y is visually higher on screen
    # (Vision origin is bottom-left), so sort by y descending for reading order.
    band.sort(key=lambda r: r[0], reverse=True)

    # Keep lines that look like dialogue continuation, drop tiny crumbs
    kept: list[tuple[float, str, float, float]] = []
    for y_c, text, conf, w in band:
        if kept and similar(kept[-1][1], text):
            # same line duplicate at slightly different box
            if len(text) > len(kept[-1][1]) or conf > kept[-1][2]:
                kept[-1] = (y_c, text, conf, w)
            continue
        kept.append((y_c, text, conf, w))

    if not kept:
        return "", 0.0

    # If one dominant long line and a short stacked complement, join them.
    if len(kept) == 1:
        return kept[0][1], kept[0][2]

    # Prefer joining 2 closest dialogue lines when both are substantial
    texts = [t for _y, t, _c, _w in kept[:3]]
    conf = max(c for _y, _t, c, _w in kept[:3])
    joined = "".join(texts)
    # Avoid joining unrelated far-apart fragments
    if len(kept) >= 2 and abs(kept[0][0] - kept[1][0]) <= 0.10:
        return joined if len(joined) <= 40 else kept[0][1], conf
    # otherwise take the longest / widest
    best = max(kept, key=lambda r: (len(r[1]), r[3], r[2]))
    return best[1], best[2]


def similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return (inter / union) >= 0.82 and abs(len(a) - len(b)) <= 4


def merge_cues(raw: list[tuple[float, str, float]], min_short: float = 0.20) -> list[Cue]:
    if not raw:
        return []
    cues: list[Cue] = []
    cur_start, cur_text, cur_conf = raw[0][0], raw[0][1], raw[0][2]
    last_t = raw[0][0]

    def flush(end_t: float) -> None:
        nonlocal cur_start, cur_text, cur_conf
        if not cur_text:
            return
        dur = max(0.0, end_t - cur_start)
        if dur < min_short and cues and similar(cues[-1].text, cur_text):
            return
        cues.append(
            Cue(
                start=round(cur_start, 2),
                end=round(max(end_t, cur_start + 0.2), 2),
                text=cur_text,
                conf=round(cur_conf, 3),
                uncertain=cur_conf < 0.4,
            )
        )

    for t, text, conf in raw[1:]:
        if similar(text, cur_text):
            if len(text) > len(cur_text) or (
                len(text) == len(cur_text) and conf > cur_conf
            ):
                cur_text, cur_conf = text, conf
            last_t = t
            continue
        flush(max(last_t + 0.12, cur_start + 0.2))
        cur_start, cur_text, cur_conf, last_t = t, text, conf, t
    flush(last_t + 0.35)

    cleaned: list[Cue] = []
    for c in cues:
        if cleaned and similar(cleaned[-1].text, c.text):
            prev = cleaned[-1]
            prev.end = max(prev.end, c.end)
            if len(c.text) > len(prev.text):
                prev.text = c.text
            prev.conf = max(prev.conf, c.conf)
            prev.uncertain = prev.uncertain and c.uncertain
            continue
        cleaned.append(c)
    return cleaned


def build_sample_times(
    duration: float, coarse: float, fine: float, change_pairs: list[tuple[float, float]]
) -> list[float]:
    times: set[float] = set()
    # always dense head/tail for hooks & CTA
    for x in np.arange(0.0, min(4.0, duration), fine):
        times.add(round(float(x), 3))
    for x in np.arange(max(0.0, duration - 4.0), duration, fine):
        times.add(round(float(x), 3))
    # densify only inside changed coarse intervals
    for a, b in change_pairs:
        x = a
        while x <= b + 1e-9:
            times.add(round(float(x), 3))
            x += fine
    # keep a light backbone so long static lines still get endpoints
    x = 0.0
    while x <= duration:
        times.add(round(float(x), 3))
        x += max(coarse, 0.8)
    times.add(round(max(0.0, duration - 0.08), 3))
    return sorted(t for t in times if 0.0 <= t <= duration)


def classify_subs_mode(cues: list[Cue], duration: float) -> dict:
    """Classify burned-in text: full dialogue vs sparse 花字 vs none."""
    if duration <= 0:
        duration = 1.0
    usable = [c for c in cues if not is_ocr_garbage(c.text)]
    n = len(usable)
    if n == 0:
        mode = "none"
    else:
        coverage = sum(max(0.0, c.end - c.start) for c in usable) / duration
        avg_len = sum(len(c.text) for c in usable) / n
        # Many short marketing words covering long spans => sparse 花字
        long_short = sum(1 for c in usable if len(c.text) <= 8 and (c.end - c.start) > 8)
        dialogueish = sum(
            1
            for c in usable
            if any(k in c.text for k in ("块", "代", "付", "券", "买", "你好", "左下角", "任选"))
        )
        # Short ads often have only 4–8 dialogue captions — still treat as dialogue
        if dialogueish >= 2 and avg_len >= 6:
            mode = "full_dialogue" if (n >= 4 or coverage >= 0.25) else "partial"
        elif n >= 12 and avg_len >= 8 and coverage >= 0.35:
            mode = "full_dialogue"
        elif n <= 8 or long_short >= max(1, n // 2) or avg_len < 8:
            mode = "sparse_huazi"
        else:
            mode = "partial"
    return {
        "mode": mode,
        "cue_count": n,
        "raw_cue_count": len(cues),
        "coverage": round(
            sum(max(0.0, c.end - c.start) for c in usable) / duration, 3
        ),
        "avg_len": round(sum(len(c.text) for c in usable) / n, 2) if n else 0.0,
    }


def extract_subtitles(
    video: Path,
    edit_dir: Path,
    coarse: float = 0.40,
    fine: float = 0.15,
    diff_thresh: float = 10.0,
    y_min: float = 0.24,
    y_max: float = 0.42,
) -> dict:
    if not _VISION_OK:
        raise RuntimeError("macOS Vision is required for this extractor")

    video = video.resolve()
    edit_dir = edit_dir.resolve()
    frames_dir = edit_dir / "sub_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    duration = probe_duration(video)
    width, height = probe_size(video)

    # Pass 1: coarse signatures
    coarse_times: list[float] = []
    t = 0.0
    while t <= duration - 0.05:
        coarse_times.append(round(t, 3))
        t += coarse
    coarse_times.append(round(max(0.0, duration - 0.1), 3))

    sigs: dict[float, np.ndarray] = {}
    for ct in coarse_times:
        fp = frames_dir / f"c_{int(ct * 1000):07d}.jpg"
        extract_frame(video, ct, fp)
        sigs[ct] = band_signature(fp)

    change_pairs: list[tuple[float, float]] = []
    for a, b in zip(coarse_times, coarse_times[1:]):
        if sig_changed(sigs[a], sigs[b], diff_thresh):
            change_pairs.append((a, b))

    sample_times = build_sample_times(duration, coarse, fine, change_pairs)

    raw_samples: list[tuple[float, str, float]] = []
    ocr_log: list[dict] = []
    prev_sig = None
    prev_text = ""
    ocr_calls = 0
    skipped = 0

    for tt in sample_times:
        fp = frames_dir / f"f_{int(tt * 1000):07d}.jpg"
        extract_frame(video, tt, fp)
        sig = band_signature(fp)

        if (
            prev_sig is not None
            and prev_text
            and not sig_changed(prev_sig, sig, diff_thresh)
        ):
            skipped += 1
            ocr_log.append({"t": tt, "skipped": True, "text": prev_text})
            raw_samples.append((tt, prev_text, max(0.9, 0.5)))
            prev_sig = sig
            continue

        hits = vision_ocr(fp)
        ocr_calls += 1
        text, conf = pick_dialogue_line(hits, y_min=y_min, y_max=y_max)
        ocr_log.append(
            {
                "t": tt,
                "skipped": False,
                "text": text,
                "conf": conf,
                "raw": [
                    {"text": h.text, "conf": h.conf, "y": round(h.y, 3), "h": round(h.h, 3)}
                    for h in hits
                    if 0.15 <= (h.y + h.h / 2) <= 0.55
                ][:12],
            }
        )
        if text:
            raw_samples.append((tt, text, conf))
            prev_text = text
        prev_sig = sig

    cues = merge_cues(raw_samples)

    # Mid-cue recheck for uncertain / low-conf lines
    for i, cue in enumerate(cues):
        if not cue.uncertain and cue.conf >= 0.45:
            continue
        mid = (cue.start + cue.end) / 2.0
        fp = frames_dir / f"re_{i:03d}.jpg"
        extract_frame(video, mid, fp)
        text, conf = pick_dialogue_line(vision_ocr(fp), y_min=y_min, y_max=y_max)
        ocr_calls += 1
        if text and (similar(text, cue.text) or len(text) >= len(cue.text)):
            if len(text) >= len(cue.text):
                cue.text = text
            cue.conf = max(cue.conf, conf)
            cue.uncertain = conf < 0.4

    # Drop OCR garbage; clamp absurdly long spans for short 花字
    filtered: list[Cue] = []
    for c in cues:
        if is_ocr_garbage(c.text):
            continue
        dur = max(0.0, c.end - c.start)
        if len(c.text) <= 6 and dur > 12:
            c.end = round(c.start + 3.0, 2)
            c.uncertain = True
        filtered.append(c)
    quality = classify_subs_mode(filtered, duration)

    payload = {
        "video": str(video),
        "duration": duration,
        "size": [width, height],
        "engine": "vision-bbox",
        "params": {
            "coarse": coarse,
            "fine": fine,
            "diff_thresh": diff_thresh,
            "y_min": y_min,
            "y_max": y_max,
        },
        "stats": {
            "coarse_frames": len(coarse_times),
            "change_intervals": len(change_pairs),
            "sample_times": len(sample_times),
            "ocr_calls": ocr_calls,
            "skipped": skipped,
            "cues": len(filtered),
            "cues_raw": len(cues),
        },
        "quality": quality,
        "cues": [asdict(c) for c in filtered],
        "script_text": "\n".join(c.text for c in filtered),
        "ocr_log": ocr_log,
    }

    (edit_dir / "burned_subs.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (edit_dir / "burned_subs.txt").write_text(
        "\n".join(f"[{c.start:5.2f}-{c.end:5.2f}] {c.text}" for c in cues),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract burned-in dialogue subtitles")
    ap.add_argument("video", type=Path)
    ap.add_argument("--edit-dir", type=Path, default=None)
    ap.add_argument("--coarse", type=float, default=0.40)
    ap.add_argument("--fine", type=float, default=0.15)
    ap.add_argument("--diff-thresh", type=float, default=10.0)
    ap.add_argument("--y-min", type=float, default=0.25)
    ap.add_argument("--y-max", type=float, default=0.38)
    args = ap.parse_args(argv)

    video = args.video.expanduser().resolve()
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 1
    edit_dir = args.edit_dir or (video.parent / f"edit-{video.stem}")
    edit_dir = edit_dir.expanduser().resolve()
    edit_dir.mkdir(parents=True, exist_ok=True)

    # fresh OCR outputs; keep frames cache if present
    for name in ("burned_subs.json", "burned_subs.txt"):
        p = edit_dir / name
        if p.exists():
            p.unlink()

    print(f"extracting burned-in subs from {video.name}")
    print(f"edit-dir: {edit_dir}")
    payload = extract_subtitles(
        video=video,
        edit_dir=edit_dir,
        coarse=args.coarse,
        fine=args.fine,
        diff_thresh=args.diff_thresh,
        y_min=args.y_min,
        y_max=args.y_max,
    )
    stats = payload["stats"]
    print(
        f"done: cues={stats['cues']} ocr_calls={stats['ocr_calls']} "
        f"skipped={stats['skipped']} samples={stats['sample_times']} "
        f"changes={stats['change_intervals']}"
    )
    print(f"wrote {edit_dir / 'burned_subs.json'}")
    print(f"wrote {edit_dir / 'burned_subs.txt'}")
    print("--- cues ---")
    print((edit_dir / "burned_subs.txt").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
