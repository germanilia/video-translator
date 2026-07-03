"""Gender helpers for Hebrew dubbing.

Many second-person forms are identical in consonants for both genders and only
niqqud differentiates them (lecha/lakh, otcha/otakh, yatsata/yatsat). Dicta's
diacritizer picks the statistically common masculine reading, so lines addressed
to a woman flip gender mid-sentence in the TTS. feminize_second_person() runs
AFTER diacritization on lines whose addressee is female.

No heavy imports - safe for both venvs.
"""

import json
import re
from pathlib import Path

SCENE_GAP_SEC = 8.0

# plural possessive/prepositional suffix: -eicha -> -ayich  (עָלֶיךָ -> עָלַיִךְ)
_PLURAL_SUFFIX = re.compile("ֶ?יךָ")
# singular suffix kaf: -cha -> -akh  (שֶׁלְּךָ -> שֶׁלָּךְ handled approximately: -> שֶׁלְּךְ)
_KAF_SUFFIX = re.compile("[ְּ]*ךָ")
# past-tense 2sg masculine: word-final tav+qamats (יָצָאתָ -> יָצָאתְ)
_PAST_TAV = re.compile("(תָּ|תָ)(?=[\\s,.!?:;\"'\\-]|$)")


def feminize_second_person(vowelized: str) -> str:
    out = _PLURAL_SUFFIX.sub("ַיִךְ", vowelized)
    out = _KAF_SUFFIX.sub("ָךְ", out)
    out = _PAST_TAV.sub("תְּ", out)
    return out


def addressee_gender(segs: list[dict], i: int, genders: dict[str, str]) -> str:
    """Gender of who line i is spoken TO: nearest substantial other-speaker line."""
    me = segs[i]["speaker"]
    for j in range(i - 1, -1, -1):
        if segs[j + 1]["start"] - segs[j]["end"] > SCENE_GAP_SEC:
            break
        if segs[j]["speaker"] != me and len(segs[j]["text"].split()) > 1:
            return genders.get(segs[j]["speaker"], "unknown")
    for j in range(i + 1, len(segs)):
        if segs[j]["start"] - segs[j - 1]["end"] > SCENE_GAP_SEC:
            break
        if segs[j]["speaker"] != me and len(segs[j]["text"].split()) > 1:
            return genders.get(segs[j]["speaker"], "unknown")
    return "unknown"


def addressee_map(work: Path) -> dict[int, str]:
    """segment id -> addressee gender, computed from segments.json + speakers.json."""
    segs = sorted(json.loads((work / "segments.json").read_text()), key=lambda s: s["start"])
    genders = {}
    sp = work / "speakers.json"
    if sp.exists():
        genders = {k: v.get("gender", "unknown") for k, v in json.loads(sp.read_text()).items()}
    return {s["id"]: addressee_gender(segs, i, genders) for i, s in enumerate(segs)}

def speaker_overrides(work: Path, segs: list[dict]) -> None:
    """segments.json is the single source of truth for speaker labels; join by id
    so attribution repairs there propagate without editing translations.json."""
    import json as _json
    seg_path = work / "segments.json"
    if not seg_path.exists():
        return
    spk = {s["id"]: s["speaker"] for s in _json.loads(seg_path.read_text())}
    for s in segs:
        if s["id"] in spk:
            s["speaker"] = spk[s["id"]]
