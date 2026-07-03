"""Translate transcribed segments to Hebrew locally with NLLB-200 (no API).

Reads:  work/segments.json
Writes: work/translations.json  [{id, start, end, speaker, text_he}]
"""

import json
import os
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / os.environ.get("VT_WORK", "work")
MODEL = "facebook/nllb-200-distilled-600M"
BATCH = 16


def main() -> None:
    segs = json.loads((WORK / "segments.json").read_text())
    tok = AutoTokenizer.from_pretrained(MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    bos = tok.convert_tokens_to_ids("heb_Hebr")

    out = []
    for i in range(0, len(segs), BATCH):
        batch = segs[i : i + BATCH]
        inputs = tok([s["text"] for s in batch], return_tensors="pt", padding=True)
        gen = model.generate(**inputs, forced_bos_token_id=bos, max_new_tokens=128)
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for seg, he in zip(batch, texts):
            out.append(
                {
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": seg["speaker"],
                    "text_he": he.strip(),
                }
            )
        print(f"translated {min(i + BATCH, len(segs))}/{len(segs)}")

    (WORK / "translations.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {WORK / 'translations.json'}")


if __name__ == "__main__":
    main()
