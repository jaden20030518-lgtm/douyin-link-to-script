#!/usr/bin/env python3
"""Shared domain lexicon for 本地生活 / 服饰团购 script extraction."""

from __future__ import annotations

import re

# FunASR hotword string (space-separated). Higher weight via repetition is not
# supported uniformly; keep a focused list.
DEFAULT_HOTWORDS = (
    "沐沄尚品 快鱼 绿谷店 361度 代金券 叠加 团购 "
    "9块9 九点九 代20 任选 核销 直销 直供 "
    "男装 女装 童装 羽绒服 卫衣"
)

BRAND_LEXICON = [
    "沐沄尚品",
    "快鱼",
    "绿谷店",
    "361度",
    "361",
]

# Common Paraformer confusions on this corpus → preferred form
ASR_CORRECTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"暮云上品|目云商品|沐运尚品|沐运简品|肽沄简品|木运尚品|沐云尚品|沐云上品"), "沐沄尚品"),
    (re.compile(r"柴油代金券|叠用代金券"), "叠加代金券"),
    (re.compile(r"柴油"), "叠加"),
    (re.compile(r"闭眼充"), "闭眼冲"),
    (re.compile(r"工厂直购"), "工厂直销"),
    (re.compile(r"两千瓶"), "2000平"),
    (re.compile(r"两千平的"), "2000平的"),
    (re.compile(r"进双挂"), "进店逛"),
    (re.compile(r"赚下价"), "赚差价"),
    (re.compile(r"只装女装"), "男装女装"),
    (re.compile(r"惊艳到位"), "惊艳到我"),
    (re.compile(r"休闲产饮"), "休闲茶饮"),
    (re.compile(r"九点九抵二十|9\.9抵二十"), "9块9代20"),
    (re.compile(r"九块九到二十"), "9块9代20"),
    (re.compile(r"不用忘了"), "不用问"),
    (re.compile(r"先存二十张|先囤二十米|先国20张|先囤20米"), "先囤20张"),
    (re.compile(r"摸到到"), "摸得到"),
    (re.compile(r"穿心新"), "穿新衣"),
    (re.compile(r"又有有面|有有面|又又有面"), "又有面"),
    (re.compile(r"还解锁你的"), "来解锁你的"),
    (re.compile(r"赶紧带上去。对，一起来吧"), "赶紧带上你闺蜜一起来吧"),
    (re.compile(r"垂直感十足"), "直筒感十足"),
    # spoken digits common in short ads
    (re.compile(r"幺九九|么九九|一九九"), "199"),
    (re.compile(r"五百块"), "500块"),
    (re.compile(r"带五百抢|代五百抢"), "代500抢"),
    (re.compile(r"(?<![代带])五百抢"), "500抢"),
    (re.compile(r"这视频我要讲啊"), "在视频左下角啊"),
    (re.compile(r"那就有哦哦，好的。?$"), ""),
]

# OCR garbage: symbol-heavy or meaningless crumbs
OCR_GARBAGE = re.compile(
    r"^[\s·•。．\.\,，、\-—_~～<>＜＞\*＊]+$|"
    r"^[令今今·•。]{1,6}$|"
    r"^(广告|关注|已关注)$"
)

HUZI_KEYWORDS = {
    "超划算",
    "每周都有上新",
    "面周都有上新",
    "工厂直供",
    "自家工厂直供",
    "都是自家工厂直供",
}


def apply_asr_corrections(text: str) -> str:
    out = text
    for pat, repl in ASR_CORRECTIONS:
        out = pat.sub(repl, out)
    return out


def is_ocr_garbage(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 2:
        return True
    if OCR_GARBAGE.search(t):
        return True
    han = len(re.findall(r"[\u4e00-\u9fff]", t))
    if han == 0:
        return True
    # mostly punctuation / bullets
    if han / max(len(t), 1) < 0.4 and t not in HUZI_KEYWORDS:
        return True
    return False
