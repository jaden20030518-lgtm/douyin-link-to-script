"""Transcribe a video with FunASR (local, free alternative to ElevenLabs).

Extracts mono 16kHz audio via ffmpeg, transcribes with FunASR locally,
writes the output in ElevenLabs-compatible format to <edit_dir>/transcripts/<video_stem>.json.

Cached: if the output file already exists, transcription is skipped.

Usage:
    python helpers/transcribe_funasr.py <video_path>
    python helpers/transcribe_funasr.py <video_path> --edit-dir ./work
    python helpers/transcribe_funasr.py <video_path> --language zh
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    from funasr import AutoModel
except ImportError:
    sys.exit("FunASR not installed. Run: pip install funasr")

try:
    from domain_lexicon import DEFAULT_HOTWORDS, apply_asr_corrections
except ImportError:
    DEFAULT_HOTWORDS = ""
    def apply_asr_corrections(text: str) -> str:  # type: ignore
        return text

# Process-local model cache (avoid reload when batching sequentially)
_MODEL_CACHE: dict[str, object] = {}


def extract_audio(video_path: Path, dest: Path) -> None:
    """Extract mono 16kHz audio from video."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _get_model(language: str = "auto"):
    key = "zh" if language in ("zh", "auto") else "en"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if key == "zh":
        model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            disable_update=True,
        )
    else:
        model = AutoModel(
            model="paraformer-en",
            vad_model="fsmn-vad",
            punc_model="ct-punc-en",
            spk_model="cam++",
            disable_update=True,
        )
    _MODEL_CACHE[key] = model
    return model


def transcribe_with_funasr(
    audio_path: Path,
    language: str = "auto",
    hotword: str | None = None,
) -> dict:
    """Transcribe audio with FunASR and return structured transcript."""
    model = _get_model(language)
    hw = (hotword if hotword is not None else DEFAULT_HOTWORDS).strip()

    result = model.generate(
        input=str(audio_path),
        batch_size_s=300,
        hotword=hw,
    )

    words = []
    full_text = ""
    sentence_info = []

    if isinstance(result, list) and len(result) > 0:
        res = result[0]
        full_text = str(res.get("text") or "")

        # Prefer sentence_info (includes speaker labels when cam++ is enabled)
        raw_sentences = res.get("sentence_info") or []
        for s in raw_sentences:
            text = str(s.get("text") or "").strip()
            if not text:
                continue
            start_ms = float(s.get("start", 0.0))
            end_ms = float(s.get("end", start_ms))
            spk = s.get("spk", s.get("speaker", 0))
            spk_label = f"S{spk}" if not str(spk).startswith("S") else str(spk)
            sentence_info.append(
                {
                    "text": text,
                    "start": start_ms / 1000.0,
                    "end": end_ms / 1000.0,
                    "spk": spk,
                    "speaker": spk_label,
                    "timestamp": s.get("timestamp") or [],
                }
            )

        # Char/token timestamps from Paraformer (Chinese has no spaces)
        timestamps = res.get("timestamp") or []
        if timestamps and full_text:
            # Strip punctuation for alignment against timestamp tokens
            chars = [ch for ch in full_text if ch.strip() and ch not in "，。！？、；：""''（）()[]【】"]
            # Fallback: keep all non-space chars
            if len(chars) != len(timestamps):
                chars = [ch for ch in full_text if not ch.isspace()]
            for i, pair in enumerate(timestamps):
                start_ms, end_ms = pair[0], pair[1]
                token = chars[i] if i < len(chars) else ""
                if not token:
                    continue
                # find speaker from sentence covering this time
                spk_label = "S0"
                t_mid = ((start_ms + end_ms) / 2.0) / 1000.0
                for s in sentence_info:
                    if s["start"] <= t_mid <= s["end"] + 1e-3:
                        spk_label = s["speaker"]
                        break
                words.append(
                    {
                        "word": token,
                        "start": start_ms / 1000.0,
                        "end": end_ms / 1000.0,
                        "speaker": spk_label,
                    }
                )

        # If still empty, approximate from sentence_info
        if not words and sentence_info:
            for s in sentence_info:
                words.append(
                    {
                        "word": s["text"],
                        "start": s["start"],
                        "end": s["end"],
                        "speaker": s["speaker"],
                    }
                )

    # Domain corrections on text + sentence fragments
    full_text = apply_asr_corrections(full_text)
    for s in sentence_info:
        raw = s["text"]
        s["text_raw"] = raw
        s["text"] = apply_asr_corrections(raw)

    return {
        "text": full_text,
        "words": words,
        "sentence_info": sentence_info,
        "language": language,
        "engine": "funasr",
        "model": "paraformer-zh",
        "hotword": hw,
    }


def transcribe_one(
    video: Path,
    edit_dir: Path,
    language: str = "auto",
    verbose: bool = True,
    hotword: str | None = None,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio)
        size_mb = audio.stat().st_size / (1024 * 1024)

        if verbose:
            print(f"  transcribing {video.stem}.wav ({size_mb:.1f} MB) with FunASR", flush=True)

        payload = transcribe_with_funasr(audio, language, hotword=hotword)

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        if "words" in payload:
            print(f"    words: {len(payload['words'])}")
        if "sentence_info" in payload:
            print(f"    sentences: {len(payload['sentence_info'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video with FunASR")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default="auto",
        help="Language code: 'zh' (Chinese), 'en' (English), 'auto' (default)",
    )
    ap.add_argument(
        "--hotword",
        type=str,
        default=None,
        help="Space-separated hotwords (default: domain lexicon)",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        language=args.language,
        hotword=args.hotword,
    )


if __name__ == "__main__":
    main()
