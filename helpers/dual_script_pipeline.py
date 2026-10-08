#!/usr/bin/env python3
"""Dual-channel original-script pipeline for Douyin-style videos.

Channels (run concurrently per video):
  A) FunASR Paraformer (+VAD/punc/spk)  -> transcripts/<stem>.json
  B) Burned-in subtitle extract (Vision) -> burned_subs.json

Then merge + diff + script_original.md

Concurrent-safe for colleagues:
  - each video uses an isolated edit dir
  - exclusive lock file prevents two people processing the same video
  - batch mode can run multiple videos with --jobs (default 1; ASR is heavy)

Usage:
  python helpers/dual_script_pipeline.py video.mp4
  python helpers/dual_script_pipeline.py a.mp4 b.mp4 --jobs 1
  python helpers/dual_script_pipeline.py video.mp4 --work-root ./work
  python helpers/dual_script_pipeline.py video.mp4 --skip-asr   # OCR only + merge if ASR exists
  python helpers/dual_script_pipeline.py video.mp4 --force      # ignore caches
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HELPERS = Path(__file__).resolve().parent
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))

from merge_asr_subs import merge_paths  # noqa: E402


def _try_fcntl():
    try:
        import fcntl

        return fcntl
    except Exception:
        return None


class WorkLock:
    """Exclusive lock so two colleagues don't process the same edit dir."""

    def __init__(self, edit_dir: Path, timeout_s: float = 0.0):
        self.lock_path = edit_dir / ".pipeline.lock"
        self.timeout_s = timeout_s
        self._fh = None
        self._fcntl = _try_fcntl()

    def acquire(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.lock_path, "a+", encoding="utf-8")
        if self._fcntl is None:
            # Best-effort on non-POSIX: write pid marker
            content = self._fh.read().strip()
            if content and content != str(os.getpid()):
                return False
            self._fh.seek(0)
            self._fh.truncate()
            self._fh.write(str(os.getpid()))
            self._fh.flush()
            return True

        start = time.time()
        while True:
            try:
                self._fcntl.flock(self._fh.fileno(), self._fcntl.LOCK_EX | self._fcntl.LOCK_NB)
                self._fh.seek(0)
                self._fh.truncate()
                self._fh.write(f"pid={os.getpid()} ts={time.time():.0f}\n")
                self._fh.flush()
                return True
            except BlockingIOError:
                if self.timeout_s <= 0 or (time.time() - start) >= self.timeout_s:
                    return False
                time.sleep(0.5)

    def release(self) -> None:
        if not self._fh:
            return
        try:
            if self._fcntl is not None:
                self._fcntl.flock(self._fh.fileno(), self._fcntl.LOCK_UN)
            self._fh.close()
        finally:
            self._fh = None


def default_edit_dir(video: Path, work_root: Path | None) -> Path:
    if work_root is not None:
        return (work_root / video.stem).resolve()
    return (video.parent / f"edit-{video.stem}").resolve()


def run_asr(
    video: Path,
    edit_dir: Path,
    language: str,
    force: bool,
    hotword: str | None = None,
) -> Path:
    from transcribe_funasr import transcribe_one

    out = edit_dir / "transcripts" / f"{video.stem}.json"
    if force and out.exists():
        out.unlink()
    return transcribe_one(
        video=video,
        edit_dir=edit_dir,
        language=language,
        verbose=True,
        hotword=hotword,
    )


def run_ocr(video: Path, edit_dir: Path, force: bool) -> Path:
    from extract_burned_subs import extract_subtitles

    out = edit_dir / "burned_subs.json"
    if force and out.exists():
        out.unlink()
        clean = edit_dir / "burned_subs_clean.json"
        if clean.exists():
            clean.unlink()
    if out.exists() and not force:
        print(f"cached: {out.name}")
        return out
    payload = extract_subtitles(video=video, edit_dir=edit_dir)
    return edit_dir / "burned_subs.json"


def maybe_clean_subs(edit_dir: Path) -> Path:
    """Optional lightweight OCR cleanup if clean file missing."""
    raw = edit_dir / "burned_subs.json"
    clean = edit_dir / "burned_subs_clean.json"
    if clean.exists():
        return clean
    if not raw.exists():
        return raw
    # Minimal pass-through: write clean cues with normalize only
    try:
        from extract_burned_subs import normalize_text, similar
    except Exception:
        return raw
    data = json.loads(raw.read_text(encoding="utf-8"))
    cues = data.get("cues") or []
    merged = []
    for c in cues:
        text = normalize_text(str(c.get("text") or ""))
        if len(text) < 2:
            continue
        if merged and similar(merged[-1]["text"], text):
            if len(text) > len(merged[-1]["text"]):
                merged[-1]["text"] = text
            merged[-1]["end"] = max(merged[-1]["end"], c.get("end", merged[-1]["end"]))
            continue
        merged.append(
            {
                "start": c.get("start", 0.0),
                "end": c.get("end", 0.0),
                "text": text,
                "conf": c.get("conf", 0.5),
                "uncertain": c.get("uncertain", False),
            }
        )
    payload = {"cues": merged, "source": str(raw)}
    clean.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return clean


def process_one(
    video: Path,
    work_root: Path | None,
    language: str = "zh",
    force: bool = False,
    skip_asr: bool = False,
    skip_ocr: bool = False,
    lock_timeout: float = 0.0,
    url: str | None = None,
    hotword: str | None = None,
) -> dict:
    video = video.resolve()
    if not video.exists():
        raise FileNotFoundError(video)

    edit_dir = default_edit_dir(video, work_root)
    edit_dir.mkdir(parents=True, exist_ok=True)
    lock = WorkLock(edit_dir, timeout_s=lock_timeout)
    if not lock.acquire():
        raise RuntimeError(
            f"edit dir locked by another process: {edit_dir / '.pipeline.lock'}\n"
            f"Another colleague may be processing this video. Retry later."
        )

    t0 = time.time()
    status = {
        "video": str(video),
        "edit_dir": str(edit_dir),
        "ok": False,
        "error": None,
        "artifacts": {},
    }
    try:
        asr_path = edit_dir / "transcripts" / f"{video.stem}.json"
        ocr_path = edit_dir / "burned_subs.json"

        futures = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            if not skip_asr:
                futures[
                    pool.submit(run_asr, video, edit_dir, language, force, hotword)
                ] = "asr"
            if not skip_ocr:
                futures[pool.submit(run_ocr, video, edit_dir, force)] = "ocr"

            errors = []
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    path = fut.result()
                    status["artifacts"][name] = str(path)
                    print(f"[ok] {name}: {path}")
                except Exception as e:
                    errors.append(f"{name}: {e}")
                    traceback.print_exc()
            if errors and not ((asr_path.exists() or skip_asr) and (ocr_path.exists() or skip_ocr)):
                raise RuntimeError("; ".join(errors))

        # Prefer cleaned subs when available
        subs_path = maybe_clean_subs(edit_dir) if ocr_path.exists() else ocr_path
        if not asr_path.exists() and not skip_asr:
            raise RuntimeError("ASR output missing")
        if not subs_path.exists() and not skip_ocr:
            raise RuntimeError("OCR output missing")

        # If one channel skipped/missing, still merge with whatever exists
        if not asr_path.exists():
            asr_path.write_text(
                json.dumps({"text": "", "words": [], "engine": "none"}, ensure_ascii=False),
                encoding="utf-8",
            )
        if not subs_path.exists():
            subs_path.write_text(json.dumps({"cues": []}, ensure_ascii=False), encoding="utf-8")

        vid = video.stem
        if url is None and vid.isdigit():
            url = f"https://www.douyin.com/video/{vid}"

        # Optional human-polished ASR / role script drops higher-quality prose in.
        # If present, keep auto merge/diff for QA, then prefer override as script_original.
        override = edit_dir / "asr_override.txt"
        merge_payload = merge_paths(
            asr_path=asr_path,
            subs_path=subs_path,
            out_dir=edit_dir,
            video_id=vid,
            url=url,
        )
        if override.exists():
            body = override.read_text(encoding="utf-8").strip() + "\n"
            if not body.startswith("对标"):
                head = f"对标：{url}\n\n" if url else f"对标：{vid}\n\n"
                body = head + body
            (edit_dir / "script_original.md").write_text(body, encoding="utf-8")
            merge_payload["asr_override"] = str(override)
            print(f"[ok] applied asr_override.txt -> script_original.md")

        # Manifest for colleagues / downstream
        manifest = {
            "video": str(video),
            "video_id": vid,
            "edit_dir": str(edit_dir),
            "url": url,
            "subs_mode": merge_payload.get("subs_mode"),
            "subs_quality": merge_payload.get("subs_quality"),
            "channels": {
                "asr": str(asr_path) if asr_path.exists() else None,
                "burned_subs": str(edit_dir / "burned_subs.json"),
                "burned_subs_clean": str(subs_path),
            },
            "outputs": {
                "merge": str(edit_dir / "merge.json"),
                "diff": str(edit_dir / "diff.json"),
                "script_original": str(edit_dir / "script_original.md"),
            },
            "diff_count": merge_payload.get("diff_count", 0),
            "turns": len(merge_payload.get("turns") or []),
            "elapsed_s": round(time.time() - t0, 2),
            "pid": os.getpid(),
        }
        (edit_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        status["ok"] = True
        status["manifest"] = manifest
        print(
            f"[done] {video.name} mode={manifest.get('subs_mode')} "
            f"turns={manifest['turns']} diffs={manifest['diff_count']} "
            f"in {manifest['elapsed_s']}s"
        )
        print(f"  script: {edit_dir / 'script_original.md'}")
        print(f"  review: {edit_dir / 'diff.json'}")
        return status
    except Exception as e:
        status["error"] = str(e)
        (edit_dir / "pipeline_error.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise
    finally:
        lock.release()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Dual-channel script pipeline (ASR ∥ OCR → merge → script)"
    )
    ap.add_argument("videos", nargs="+", type=Path, help="One or more video files")
    ap.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="Shared work root for colleagues (default: <video_dir>/edit-<stem>)",
    )
    ap.add_argument("--language", default="zh")
    ap.add_argument("--force", action="store_true", help="Ignore channel caches")
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--skip-ocr", action="store_true")
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel videos (default 1). Keep 1 on a single Mac when FunASR is heavy.",
    )
    ap.add_argument(
        "--lock-timeout",
        type=float,
        default=0.0,
        help="Seconds to wait for edit-dir lock (0=fail immediately if busy)",
    )
    ap.add_argument("--url", default=None, help="Override 对标 URL for single-video runs")
    ap.add_argument(
        "--hotword",
        default=None,
        help="Space-separated ASR hotwords (default: domain lexicon)",
    )
    args = ap.parse_args(argv)

    videos = [v.expanduser().resolve() for v in args.videos]
    missing = [str(v) for v in videos if not v.exists()]
    if missing:
        print("missing videos:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1

    work_root = args.work_root.expanduser().resolve() if args.work_root else None
    if work_root:
        work_root.mkdir(parents=True, exist_ok=True)

    results = []
    jobs = max(1, int(args.jobs))

    def _run(v: Path):
        return process_one(
            video=v,
            work_root=work_root,
            language=args.language,
            force=args.force,
            skip_asr=args.skip_asr,
            skip_ocr=args.skip_ocr,
            lock_timeout=args.lock_timeout,
            url=args.url if len(videos) == 1 else None,
            hotword=args.hotword,
        )

    if jobs == 1 or len(videos) == 1:
        for v in videos:
            try:
                results.append(_run(v))
            except Exception as e:
                print(f"[FAIL] {v.name}: {e}", file=sys.stderr)
                results.append({"video": str(v), "ok": False, "error": str(e)})
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_run, v): v for v in videos}
            for fut in as_completed(futs):
                v = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"[FAIL] {v.name}: {e}", file=sys.stderr)
                    results.append({"video": str(v), "ok": False, "error": str(e)})

    ok_n = sum(1 for r in results if r.get("ok"))
    print(f"\nsummary: {ok_n}/{len(results)} ok")
    # write batch summary if work root set
    if work_root:
        summary_path = work_root / f"batch_summary_{int(time.time())}.json"
        summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"batch summary: {summary_path}")
    return 0 if ok_n == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
