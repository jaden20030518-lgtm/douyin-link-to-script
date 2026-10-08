#!/usr/bin/env python3
"""Merge Paraformer ASR + burned-in subtitle cues into a low-error script.

Rules (fixed):
  1. Body dialogue defaults to ASR
  2. Clear numbers/prices in subs override ASR when they conflict
  3. Brand / store names: subtitle hit in lexicon overrides ASR
  4. Short sub-only cues (<=8 chars) can be inserted if ASR gap
  5. Roles: ASR speaker labels if present; else Q/A heuristic

Outputs:
  diff.json           — only conflicts / inserts for human review
  script_original.md  — 对标-style script
  merge.json          — full merged segments
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

try:
    from domain_lexicon import BRAND_LEXICON, apply_asr_corrections, is_ocr_garbage
except ImportError:
    BRAND_LEXICON = ["沐沄尚品", "快鱼", "绿谷店", "361", "361度"]

    def apply_asr_corrections(text: str) -> str:
        return text

    def is_ocr_garbage(text: str) -> bool:
        return len((text or "").strip()) < 2

NUMBER_TOKEN = re.compile(
    r"(\d+\.\d+|\d+|九块九|九点九|二十|两百|两千|九十九|一百七十九|三百三十九|一百六十九)"
)
PRICEISH = re.compile(
    r"(\d+\s*块\s*\d*|\d+\s*代\s*\d+|任选\s*\d+|囤\s*\d+|代\s*\d+)"
)

QUESTION_HINT = re.compile(r"(吗|嘛|呢|啊|一下|确认|问一下|能不能|可以吗|是真的吗)[…。]?$")
STAFF_HINT = re.compile(r"^(不用问|不用确认|是真的|确定|建议你|我们就是|您抢到)")


@dataclass
class Seg:
    start: float
    end: float
    text: str
    source: str  # asr|sub|merged
    speaker: str = ""  # S0/S1 or 顾客/店员/出镜
    conf: float = 1.0


def _norm(s: str) -> str:
    s = s.strip().replace(" ", "").replace("\n", "")
    s = s.replace("（", "(").replace("）", ")")
    return s


def _sim(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    if a == b or a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _arabize_zh_nums(text: str) -> str:
    """Light normalization for shooting scripts."""
    repl = [
        ("九块九", "9块9"),
        ("九点九", "9.9"),
        ("二十张", "20张"),
        ("二十米", "20张"),  # common ASR/OCR confusion guard
        ("两千平", "2000平"),
        ("两千坪", "2000平"),
    ]
    for a, b in repl:
        text = text.replace(a, b)
    return text


def _coalesce_sentence_info(raw: list[dict]) -> list[Seg]:
    """Merge FunASR fragments into clause-sized segments by speaker + punct."""
    segs: list[Seg] = []
    buf_text = ""
    buf_start = 0.0
    buf_end = 0.0
    buf_spk = ""

    def flush():
        nonlocal buf_text, buf_start, buf_end, buf_spk
        text = _norm(buf_text)
        if text:
            segs.append(Seg(buf_start, buf_end, text, "asr", speaker=buf_spk or "S0"))
        buf_text = ""

    for s in raw:
        text = str(s.get("text") or "")
        if not text.strip():
            continue
        start = float(s.get("start", 0.0))
        end = float(s.get("end", start))
        if start > 1000 or end > 1000:
            start, end = start / 1000.0, end / 1000.0
        spk = str(s.get("speaker") or s.get("spk") or "S0")
        if not str(spk).startswith("S"):
            spk = f"S{spk}"

        if not buf_text:
            buf_text, buf_start, buf_end, buf_spk = text, start, end, spk
        elif spk != buf_spk:
            flush()
            buf_text, buf_start, buf_end, buf_spk = text, start, end, spk
        else:
            buf_text += text
            buf_end = end

        # flush on strong punctuation
        if any(buf_text.rstrip().endswith(p) for p in ("。", "！", "？", "…")):
            flush()
        # also flush moderately long clauses ending with comma if > 18 chars
        elif buf_text.rstrip().endswith(("，", "、")) and len(_norm(buf_text)) >= 18:
            flush()

    flush()
    return segs


def load_asr_segments(asr_path: Path) -> list[Seg]:
    data = json.loads(asr_path.read_text(encoding="utf-8"))
    segs: list[Seg] = []

    # Prefer sentence_info if present (FunASR + cam++)
    if isinstance(data.get("sentence_info"), list) and data["sentence_info"]:
        return _coalesce_sentence_info(data["sentence_info"])

    words = data.get("words") or []
    if words:
        # Group by speaker + pause gaps
        buf: list[dict] = []
        cur_spk = words[0].get("speaker", "S0")

        def flush():
            nonlocal buf
            if not buf:
                return
            text = _norm("".join(w.get("word", "") for w in buf))
            if text:
                segs.append(
                    Seg(
                        float(buf[0]["start"]),
                        float(buf[-1]["end"]),
                        text,
                        "asr",
                        speaker=str(cur_spk or "S0"),
                    )
                )
            buf = []

        for w in words:
            spk = w.get("speaker", "S0")
            gap = 0.0
            if buf:
                gap = float(w["start"]) - float(buf[-1]["end"])
            if spk != cur_spk or gap > 0.55:
                flush()
                cur_spk = spk
            buf.append(w)
        flush()
        if segs:
            return segs

    # Fallback: whole text as one block
    text = _norm(str(data.get("text") or ""))
    if text:
        segs.append(Seg(0.0, 0.0, text, "asr", speaker="S0"))
    return segs


def load_sub_payload(subs_path: Path) -> tuple[list[Seg], dict]:
    data = json.loads(subs_path.read_text(encoding="utf-8"))
    quality = data.get("quality") or {}
    cues = data.get("cues") or []
    segs: list[Seg] = []
    for c in cues:
        text = _norm(str(c.get("text") or ""))
        if len(text) < 2 or is_ocr_garbage(text):
            continue
        segs.append(
            Seg(
                float(c.get("start", 0.0)),
                float(c.get("end", 0.0)),
                text,
                "sub",
                conf=float(c.get("conf", 0.5)),
            )
        )
    if not quality:
        # infer when older burned_subs.json lacks quality block
        n = len(segs)
        dur = max((s.end for s in segs), default=1.0)
        avg_len = sum(len(s.text) for s in segs) / n if n else 0
        if n == 0:
            mode = "none"
        elif n >= 12 and avg_len >= 8:
            mode = "full_dialogue"
        else:
            mode = "sparse_huazi"
        quality = {"mode": mode, "cue_count": n, "avg_len": avg_len}
    return segs, quality


def load_sub_cues(subs_path: Path) -> list[Seg]:
    segs, _quality = load_sub_payload(subs_path)
    return segs


def _extract_numbers(text: str) -> set[str]:
    return set(NUMBER_TOKEN.findall(text)) | set(m.group(0) for m in PRICEISH.finditer(text))


def _brand_hits(text: str) -> list[str]:
    return [b for b in BRAND_LEXICON if b in text]


def _overlap(a: Seg, b: Seg) -> float:
    if a.end <= 0 and a.start <= 0:
        return 0.0
    left = max(a.start, b.start)
    right = min(a.end if a.end > a.start else a.start + 1.0, b.end if b.end > b.start else b.start + 1.0)
    return max(0.0, right - left)


def merge_segments(
    asr: list[Seg],
    subs: list[Seg],
    subs_mode: str = "full_dialogue",
) -> tuple[list[Seg], list[dict]]:
    """Return merged segments + diff items for human review.

    subs_mode:
      - full_dialogue: ASR body + number/brand overrides from subs
      - sparse_huazi / none: ASR-only body; subs only contribute lexicon hits
    """
    diffs: list[dict] = []
    sparse = subs_mode in ("sparse_huazi", "none", "partial")

    if not asr and not subs:
        return [], diffs
    if not asr:
        return [Seg(s.start, s.end, _arabize_zh_nums(s.text), "sub", s.speaker, s.conf) for s in subs], [
            {"type": "asr_missing", "note": "no ASR; used subtitles only"}
        ]
    if not subs or sparse:
        # ASR-primary path: apply domain corrections, keep high-value sub hits in diff
        merged = [
            Seg(s.start, s.end, _arabize_zh_nums(apply_asr_corrections(s.text)), "asr", s.speaker, s.conf)
            for s in asr
        ]
        if not subs:
            diffs.append({"type": "subs_missing", "note": "no subtitles; used ASR only"})
        else:
            diffs.append(
                {
                    "type": "subs_mode",
                    "mode": subs_mode,
                    "note": "sparse/no dialogue captions; ASR is sole body",
                }
            )
            for s in subs:
                brands = _brand_hits(s.text)
                if brands or (len(s.text) <= 10 and any(ch.isdigit() for ch in s.text)):
                    diffs.append(
                        {
                            "type": "sub_hint",
                            "t": [s.start, s.end],
                            "sub": s.text,
                            "brands": brands,
                            "action": "review_only",
                        }
                    )
                    # light brand injection into nearest ASR seg
                    if brands:
                        for a in merged:
                            if abs(((a.start + a.end) / 2) - ((s.start + s.end) / 2)) < 4.0:
                                if brands[0] not in a.text:
                                    a.text = apply_asr_corrections(
                                        a.text.replace("暮云上品", brands[0])
                                    )
                                    if brands[0] not in a.text:
                                        # replace common wrong brand-ish tokens already handled
                                        pass
                                break
        return merged, diffs

    merged: list[Seg] = []
    used_subs: set[int] = set()

    for a in asr:
        # find best overlapping / nearby sub
        best_i, best_score = -1, 0.0
        for i, s in enumerate(subs):
            if i in used_subs:
                continue
            ov = _overlap(a, s)
            # also allow near alignment by center distance
            ac = (a.start + (a.end or a.start + 1)) / 2
            sc = (s.start + s.end) / 2
            prox = max(0.0, 1.0 - abs(ac - sc) / 2.0)
            score = ov * 2.0 + prox * _sim(a.text, s.text)
            if score > best_score:
                best_score, best_i = score, i

        text = a.text
        source = "asr"
        if best_i >= 0 and best_score >= 0.35:
            s = subs[best_i]
            used_subs.add(best_i)
            sim = _sim(a.text, s.text)
            a_nums, s_nums = _extract_numbers(a.text), _extract_numbers(s.text)
            a_brands, s_brands = _brand_hits(a.text), _brand_hits(s.text)

            # number override
            if s_nums and a_nums != s_nums and sim >= 0.35:
                diffs.append(
                    {
                        "type": "number_override",
                        "t": [a.start, a.end],
                        "asr": a.text,
                        "sub": s.text,
                        "action": "prefer_sub_numbers",
                    }
                )
                # Prefer subtitle line when it carries clearer arabic prices
                sub_has_arabic = bool(re.search(r"\d", s.text))
                asr_spoken = bool(re.search(r"[幺么零一二三四五六七八九十百千]+", a.text))
                if sim >= 0.72 or (sub_has_arabic and (sim >= 0.45 or asr_spoken)):
                    # Keep longer ASR context if sub is truncated, but swap digits
                    if len(s.text) + 4 >= len(a.text) or sim >= 0.75:
                        text = s.text
                    else:
                        text = a.text
                        # map common spoken forms using digits found in subtitle
                        if "199" in s.text:
                            text = re.sub(r"幺九九|么九九|一九九|199", "199", text)
                        if "500" in s.text:
                            text = re.sub(r"五百|500", "500", text)
                        for m in re.findall(r"\d+(?:\.\d+)?", s.text):
                            # already injected key prices above
                            pass
                    source = "merged"
                    text = apply_asr_corrections(text)
                else:
                    text = apply_asr_corrections(a.text)
                    source = "asr"

            # brand override
            if s_brands and not a_brands:
                diffs.append(
                    {
                        "type": "brand_override",
                        "t": [a.start, a.end],
                        "asr": a.text,
                        "sub": s.text,
                        "brands": s_brands,
                        "action": "append_or_replace_brand",
                    }
                )
                if sim >= 0.6:
                    text = s.text if len(s.text) >= len(a.text) * 0.8 else (a.text + s_brands[0])
                    source = "merged"
                else:
                    for b in s_brands:
                        if b not in text:
                            text = text + b
                    source = "merged"

            # Strong subtitle rescue for short opening lines ASR often mangles
            # e.g. 不用忘了 -> 不用问9块9代20是真的
            if len(s.text) <= 24 and any(
                k in s.text for k in ("不用问", "9块9", "9.9", "代20", "囤20")
            ):
                if sim < 0.9:
                    diffs.append(
                        {
                            "type": "sub_rescue_short",
                            "t": [a.start, a.end],
                            "asr": a.text,
                            "sub": s.text,
                            "action": "prefer_sub",
                        }
                    )
                    text = s.text
                    source = "merged"

            if sim < 0.55 and best_score >= 0.5 and source == "asr":
                diffs.append(
                    {
                        "type": "low_similarity",
                        "t": [a.start, a.end],
                        "asr": a.text,
                        "sub": s.text,
                        "similarity": round(sim, 3),
                        "action": "kept_asr_review",
                    }
                )

        merged.append(
            Seg(
                a.start,
                a.end,
                _arabize_zh_nums(text),
                source,
                speaker=a.speaker,
                conf=a.conf,
            )
        )

    # subtitle-only short inserts in gaps
    for i, s in enumerate(subs):
        if i in used_subs:
            continue
        if len(s.text) > 8:
            # long orphan sub — flag only
            diffs.append(
                {
                    "type": "sub_only_long",
                    "t": [s.start, s.end],
                    "sub": s.text,
                    "action": "flag_review",
                }
            )
            continue
        # find gap
        inserted = False
        for j in range(len(merged) - 1):
            if merged[j].end <= s.start and s.end <= merged[j + 1].start + 0.15:
                merged.insert(
                    j + 1,
                    Seg(s.start, s.end, _arabize_zh_nums(s.text), "sub", conf=s.conf),
                )
                diffs.append(
                    {
                        "type": "sub_insert_short",
                        "t": [s.start, s.end],
                        "sub": s.text,
                        "action": "inserted",
                    }
                )
                inserted = True
                break
        if not inserted and s.start <= 3.0:
            # opening short cue
            merged.insert(0, Seg(s.start, s.end, _arabize_zh_nums(s.text), "sub", conf=s.conf))
            diffs.append(
                {
                    "type": "sub_insert_short",
                    "t": [s.start, s.end],
                    "sub": s.text,
                    "action": "inserted_opening",
                }
            )

    merged.sort(key=lambda x: (x.start, x.end))
    return merged, diffs


def _is_monologue(segs: list[Seg]) -> bool:
    """Host-only tour videos: no customer/staff skit markers."""
    blob = "".join(s.text for s in segs)
    skit_hits = 0
    for k in ("问一下", "不用问", "确认一下", "不用确认", "您抢到", "我抢到了"):
        if k in blob:
            skit_hits += 1
    # classic hook monologue openings
    mono_hooks = ("家人们", "挖到一家", "姐妹们", "宝子们", "还没选好吗")
    if skit_hits == 0 and any(h in blob[:40] for h in mono_hooks):
        return True
    if skit_hits == 0 and len(segs) >= 1:
        # no skit lexicon at all
        return True
    return False


def assign_roles(segs: list[Seg]) -> list[Seg]:
    """Map to 顾客/店员/出镜.

    Douyin shop skits are often one host voice-acting both sides, so cam++
    speaker IDs are treated as weak hints only. Content rules dominate.
    Pure monologue tours collapse to 出镜.
    """
    if _is_monologue(segs):
        return [
            Seg(s.start, s.end, apply_asr_corrections(s.text), s.source, "出镜", s.conf)
            for s in segs
        ]

    out: list[Seg] = []
    phase = "open"  # open Q/A, then host monologue
    open_turns = 0

    for i, s in enumerate(segs):
        text = apply_asr_corrections(s.text)
        role = "出镜"

        if phase == "open":
            if "问一下" in text or "确认一下" in text:
                role = "顾客"
                open_turns += 1
            elif STAFF_HINT.search(text) or text.startswith("不用") or "建议你" in text or "我们就是" in text:
                role = "店员"
                open_turns += 1
            elif open_turns == 0 and i <= 2 and ("美女" in text or text.endswith("……") or text.endswith("…")):
                role = "顾客"
                open_turns += 1
            elif "代金券" in text and open_turns > 0 and open_turns < 6 and len(text) < 40:
                role = "店员"
                open_turns += 1
            else:
                role = "出镜"
                phase = "host"
        else:
            role = "出镜"

        if "确认一下" in text:
            role = "顾客"
            phase = "open"
        if text.startswith("不用确认") or text.startswith("不用问"):
            role = "店员"
            phase = "open"
        if "像这" in text and ("块钱" in text or "衣服" in text):
            role = "出镜"
            phase = "host"
        if phase == "host" and i > 3:
            role = "出镜"
        if role == "店员" and ("正品" in text or "尚品" in text) and len(text) > 30:
            phase = "host"

        out.append(Seg(s.start, s.end, text, s.source, speaker=role, conf=s.conf))
        if open_turns >= 6:
            phase = "host"
    return out


def collapse_for_script(segs: list[Seg]) -> list[tuple[str, str]]:
    """Collapse consecutive same-role lines into script turns."""
    turns: list[tuple[str, str]] = []
    for s in segs:
        role = s.speaker or "出镜"
        text = s.text
        if turns and turns[-1][0] == role:
            turns[-1] = (role, turns[-1][1] + text)
        else:
            turns.append((role, text))
    # light punctuation spacing for readability
    cleaned: list[tuple[str, str]] = []
    for role, text in turns:
        t = text
        t = re.sub(r"([。！？])+", r"\1", t)
        cleaned.append((role, t))
    return cleaned


def render_script(video_id: str, turns: list[tuple[str, str]], url: str | None = None) -> str:
    head = f"对标：{url}" if url else f"对标：{video_id}"
    lines = [head, ""]
    for role, text in turns:
        lines.append(f"{role}：{text}")
        # blank line before long 出镜 blocks already handled by role switch
    # insert blank line between opening dialogue and host monologue
    out_lines: list[str] = [lines[0], ""]
    seen_host = False
    for role, text in turns:
        if role == "出镜" and not seen_host:
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            seen_host = True
        out_lines.append(f"{role}：{text}")
    return "\n".join(out_lines).rstrip() + "\n"


def merge_paths(
    asr_path: Path,
    subs_path: Path,
    out_dir: Path,
    video_id: str = "",
    url: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    asr = load_asr_segments(asr_path) if asr_path.exists() else []
    # apply domain corrections early
    asr = [
        Seg(s.start, s.end, apply_asr_corrections(s.text), s.source, s.speaker, s.conf)
        for s in asr
    ]
    subs: list[Seg] = []
    quality: dict = {"mode": "none"}
    if subs_path.exists():
        subs, quality = load_sub_payload(subs_path)
    mode = str(quality.get("mode") or "none")
    merged, diffs = merge_segments(asr, subs, subs_mode=mode)
    role_segs = assign_roles(merged)
    turns = collapse_for_script(role_segs)
    script = render_script(video_id or out_dir.name, turns, url=url)

    merge_payload = {
        "video_id": video_id,
        "asr_path": str(asr_path),
        "subs_path": str(subs_path),
        "subs_mode": mode,
        "subs_quality": quality,
        "segments": [asdict(s) for s in role_segs],
        "turns": [{"role": r, "text": t} for r, t in turns],
        "diff_count": len(diffs),
    }
    (out_dir / "merge.json").write_text(
        json.dumps(merge_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "diff.json").write_text(
        json.dumps({"count": len(diffs), "items": diffs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "script_original.md").write_text(script, encoding="utf-8")
    return merge_payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge ASR + burned-in subs")
    ap.add_argument("--asr", type=Path, required=True)
    ap.add_argument("--subs", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--video-id", type=str, default="")
    ap.add_argument("--url", type=str, default=None)
    args = ap.parse_args()
    payload = merge_paths(args.asr, args.subs, args.out_dir, args.video_id, args.url)
    print(f"merged turns={len(payload['turns'])} diffs={payload['diff_count']}")
    print(args.out_dir / "script_original.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
