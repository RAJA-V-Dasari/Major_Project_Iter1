"""
Read the handwriting. Real recognition, not simulation.

    03_router/output/routed_regions.json  (what the router sent to `ocr`)
    04_ocr/input/                      (the line crops)
        -> 04_ocr/output/  ocr.json, lines.csv, transcripts/

This is the engine simulate.py was a placeholder for. It emits exactly
the same records - schema.LineResult - but with `simulated: false` and
`engine` set to the real model, so anything already built against the
simulated output keeps working unchanged. That was the whole point of
putting the contract in schema.py.

TAKES ONLY WHAT WAS ROUTED TO IT
--------------------------------
The input is Module 2's routed output, and this stage reads only the
regions whose `processor` is `ocr`. Equations belong to math_ocr,
figures to diagram_parser, and a line recogniser asked to read either
returns confident nonsense rather than an error - which is worse than
returning nothing. Regions routed elsewhere are counted in the payload
(`routed_elsewhere`) rather than silently dropped, so the numbers
reconcile against the routed file.

WHERE IDENTITY COMES FROM
-------------------------
Module 2's Region carries a `metadata` dict but never fills it - it
builds regions from six fields and metadata comes out `{}`. The one
thing that does survive is `crop_path`, and 02_segment named those
files exactly `<page_id>_b<NN>_l<NN>.png` (see crop_lines.crop_name).
So identity is parsed back out of the path, which keeps this stage
dependent only on its own input and produces the same `line_uid` that
simulate.py did - existing joins keep working.

CONFIDENCE IS MEASURED, NOT INVENTED
------------------------------------
TrOCR has no confidence head, so it is computed from the generation
itself: the mean log-probability of the tokens actually chosen,
exponentiated back into 0-1. That is a real quantity - a line the
model was unsure about scores lower - but it is a fluency measure, not
a calibrated probability of correctness. Use it to rank and to
threshold, not as "this line is 87% right".

WHAT THIS DOES NOT DO
---------------------
The text is the engine's raw output, deliberately unedited. TrOCR was
trained on IAM, so it spaces punctuation (" ." and " ,") and will
happily render a circled question number as "O.O". Cleaning that up
here would bake one engine's quirks into the contract and make later
engines look different for no reason; post-processing belongs in its
own stage, downstream, where it can be measured.

Run:
    python recognise.py                  # every ocr-routed line
    python recognise.py --limit 20       # a sample, for a quick look
    python recognise.py --batch 16
"""

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import schema
from schema import LineResult, PageResult


SRC_DIR = Path(__file__).resolve().parent
STAGE_DIR = SRC_DIR.parent
MODULES_DIR = STAGE_DIR.parent

CROP_DIR = STAGE_DIR / "input"

OUT_DIR = STAGE_DIR / "output"
# The REAL router's output (03_router), not a simulation of it.
ROUTED_PATH = MODULES_DIR / "03_router" / "output" / "routed_regions.json"
TRANSCRIPT_DIR = OUT_DIR / "transcripts"

SEGMENTATION_PATH = MODULES_DIR / "02_segment" / "output" / "segmentation.json"

DEFAULT_MODEL = "microsoft/trocr-base-handwritten"

# Only these reach a line recogniser. Module 2 decides this, not us -
# the string is its ROUTING_TABLE's value for the text labels.
OCR_PROCESSOR = "ocr"

# Lines are one row of handwriting; 64 tokens is far more than any of
# them need and caps a runaway decode.
MAX_NEW_TOKENS = 64

# 02_segment/src/crop_lines.py:crop_name() - the naming contract this
# stage parses identity back out of.
CROP_NAME_RE = re.compile(
    r"^s(?P<student>\d+)_c(?P<cie>\d+)_p(?P<page>\d+)"
    r"_b(?P<block>\d+)_l(?P<line>\d+)\.png$"
)


def parse_identity(crop_path):
    """
    "student_01/cie_1/s01_c1_p02_b01_l00.png" -> the ids inside it.

    Raises rather than guessing: a crop this cannot parse means the
    naming contract changed, and silently inventing a line_uid would
    corrupt every downstream join.
    """

    name = Path(crop_path).name

    match = CROP_NAME_RE.match(name)

    if not match:
        raise ValueError(
            f"crop path does not match 02_segment's naming contract: "
            f"{crop_path!r}"
        )

    parts = {k: int(v) for k, v in match.groupdict().items()}

    page_id = (f"s{parts['student']:02d}_c{parts['cie']}"
               f"_p{parts['page']:02d}")

    return {
        "page_id": page_id,
        "student": parts["student"],
        "cie": parts["cie"],
        "page": parts["page"],
        "block_id": parts["block"],
        "line_id": parts["line"],
        "line_uid": (f"{page_id}_b{parts['block']:02d}"
                     f"_l{parts['line']:02d}"),
    }


def load_engine(model_id, threads):
    """
    TrOCR: image processor + tokenizer + model.

    The tokenizer is assembled by hand from vocab.json/merges.txt
    rather than loaded with from_pretrained. transformers 5.x will only
    build a *fast* tokenizer, the TrOCR repos ship no tokenizer.json,
    and its slow->fast converter refuses to run without sentencepiece
    or tiktoken installed. Building the ByteLevel-BPE directly with the
    `tokenizers` library sidesteps that entirely and adds no
    dependency - RoBERTa's vocabulary is exactly what those two files
    already describe.
    """

    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
    from transformers import (RobertaTokenizerFast, AutoImageProcessor,
                              VisionEncoderDecoderModel)
    from huggingface_hub import snapshot_download

    torch.set_num_threads(threads)

    path = snapshot_download(model_id)

    backend = Tokenizer(models.BPE.from_file(
        f"{path}/vocab.json", f"{path}/merges.txt", unk_token="<unk>"
    ))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.RobertaProcessing(
        sep=("</s>", 2), cls=("<s>", 0), add_prefix_space=False
    )

    tokenizer = RobertaTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>", eos_token="</s>", unk_token="<unk>",
        sep_token="</s>", cls_token="<s>", pad_token="<pad>",
        mask_token="<mask>",
    )

    image_processor = AutoImageProcessor.from_pretrained(path)

    model = VisionEncoderDecoderModel.from_pretrained(path).eval()

    return torch, image_processor, tokenizer, model


def recognise_batch(torch, image_processor, tokenizer, model, images):
    """
    One batch of crops -> [(text, confidence)].

    Confidence is exp(mean log-prob) over the tokens the model actually
    emitted, padding excluded. Greedy decoding, so the transition
    scores are the chosen tokens' log-probabilities directly.
    """

    pixels = image_processor(images=images, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated = model.generate(
            pixels,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

    scores = model.compute_transition_scores(
        generated.sequences, generated.scores, normalize_logits=True
    )

    texts = tokenizer.batch_decode(
        generated.sequences, skip_special_tokens=True
    )

    # sequences carries the decoder start token; scores line up with
    # everything after it
    emitted = generated.sequences[:, 1:]
    real = emitted != tokenizer.pad_token_id

    results = []

    for index, text in enumerate(texts):

        mask = real[index]

        if mask.any():
            mean_logprob = scores[index][mask].mean().item()
            confidence = float(torch.exp(torch.tensor(mean_logprob)))
        else:
            confidence = 0.0

        results.append((
            text.strip(),
            round(min(1.0, max(0.0, confidence)), 3),
        ))

    return results


def select_regions(routed):
    """
    The ocr-routed regions, in reading order, plus what went elsewhere.

    Module 2 orders per page and numbers from 1. Its numbering counts
    every region on the page, including the ones it sent to math_ocr,
    so it is kept in `extra` for traceability but cannot be used as
    this payload's reading_order - schema.validate_run requires 0..n-1
    with no gaps over the lines actually present.
    """

    selected = defaultdict(list)
    elsewhere = defaultdict(int)

    for page in routed["pages"]:

        for region in page["regions"]:

            if region.get("ignored"):
                elsewhere["ignored"] += 1
                continue

            if region.get("processor") != OCR_PROCESSOR:
                elsewhere[region.get("processor") or "unassigned"] += 1
                continue

            selected[page["page"]].append(region)

    for regions in selected.values():
        regions.sort(key=lambda r: r.get("reading_order", 0))

    return selected, dict(elsewhere)


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--routed", type=Path, default=ROUTED_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, help="first N lines only")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)

    args = parser.parse_args()

    if not args.routed.exists():
        raise SystemExit(
            f"{args.routed} not found - run 03_router/src/route.py first"
        )

    routed = json.load(open(args.routed))

    selected, elsewhere = select_regions(routed)

    total = sum(len(v) for v in selected.values())

    if not total:
        raise SystemExit(f"no regions routed to {OCR_PROCESSOR!r} in {args.routed}")

    if args.limit:
        remaining = args.limit
        trimmed = {}
        for page_number in sorted(selected):
            if remaining <= 0:
                break
            take = selected[page_number][:remaining]
            trimmed[page_number] = take
            remaining -= len(take)
        selected = trimmed
        total = sum(len(v) for v in selected.values())

    print(f"Routed file : {args.routed}")
    print(f"Lines to OCR: {total}")
    print("Not ours    : " + (", ".join(
        f"{k}={v}" for k, v in sorted(elsewhere.items())) or "none"))
    print(f"Engine      : {args.model}")
    print("Loading engine...", flush=True)

    torch, image_processor, tokenizer, model = load_engine(
        args.model, args.threads
    )

    from PIL import Image

    segmentation = json.load(open(SEGMENTATION_PATH))
    meta = {p["page_id"]: p for p in segmentation}

    pages = []
    counts = defaultdict(int)
    done = 0

    started = time.time()

    for page_number in sorted(selected):

        regions = selected[page_number]
        lines = []

        for start in range(0, len(regions), args.batch):

            chunk = regions[start:start + args.batch]

            images = []
            usable = []

            for region in chunk:

                try:
                    image = Image.open(CROP_DIR / region["crop_path"])
                    images.append(image.convert("RGB"))
                    usable.append(region)
                except Exception as exc:
                    # the crop is missing or unreadable: `failed`, not
                    # `empty` - the page is not blank, the pipeline is
                    lines.append((region, "", 0.0, "failed", str(exc)))

            if images:
                try:
                    read = recognise_batch(
                        torch, image_processor, tokenizer, model, images
                    )
                except Exception as exc:
                    read = None
                    for region in usable:
                        lines.append((region, "", 0.0, "failed", str(exc)))

                if read is not None:
                    for region, (text, confidence) in zip(usable, read):
                        status = "ok" if text else "empty"
                        lines.append((region, text, confidence, status, None))

            done += len(chunk)

            print(f"  page {page_number:02d}  {done}/{total} lines "
                  f"({time.time() - started:.0f}s)", flush=True)

        # A crop that failed to open was recorded before the rest of its
        # batch, so restore the router's order before numbering -
        # otherwise a single unreadable file would shuffle the page.
        lines.sort(key=lambda entry: entry[0].get("reading_order", 0))

        # reading_order must be 0..n-1 over the lines in THIS payload
        results = []

        for order, (region, text, confidence, status, error) in enumerate(lines):

            identity = parse_identity(region["crop_path"])

            bbox = region["bbox"]

            extra = {
                "router_region_id": region.get("id"),
                "router_label": region.get("label"),
                "router_processor": region.get("processor"),
                "router_confidence": region.get("confidence"),
                "router_reading_order": region.get("reading_order"),
            }

            if error:
                extra["error"] = error

            counts[status] += 1

            results.append(LineResult(
                line_uid=identity["line_uid"],
                page_id=identity["page_id"],
                student=identity["student"],
                cie=identity["cie"],
                page=identity["page"],
                block_id=identity["block_id"],
                line_id=identity["line_id"],
                bbox=[bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
                crop=region["crop_path"],
                text=text,
                confidence=confidence,
                status=status,
                reading_order=order,
                # Module 2 sends `tall` regions to diagram_parser, so
                # nothing that reaches this stage is one.
                tall=False,
                extra=extra,
            ))

        if not results:
            continue

        info = meta.get(results[0].page_id, {})

        pages.append(PageResult(
            page_id=results[0].page_id,
            student=results[0].student,
            cie=results[0].cie,
            page=results[0].page,
            source=info.get("source", ""),
            size=info.get("size", []),
            lines=results,
        ))

    elapsed = time.time() - started

    payload = {
        "schema_version": schema.SCHEMA_VERSION,
        "simulated": False,
        "engine": args.model,
        "generated": date.today().isoformat(),
        "source": str(args.routed),
        "pages_total": len(pages),
        "lines_total": sum(len(p.lines) for p in pages),
        # so the numbers reconcile against the routed file rather than
        # a reader having to wonder where the rest went
        "routed_elsewhere": elsewhere,
        "seconds": round(elapsed, 1),
        "pages": [p.to_dict() for p in pages],
    }

    # never ship a payload the module's own validator would reject
    checked = schema.validate_run(payload)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "ocr.json", "w") as handle:
        json.dump(payload, handle, indent=1)

    with open(OUT_DIR / "lines.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["line_uid", "page_id", "student", "cie", "page",
                         "block_id", "line_id", "reading_order",
                         "x1", "y1", "x2", "y2", "status", "confidence",
                         "router_label", "crop", "text"])
        for page in pages:
            for line in page.lines:
                writer.writerow([
                    line.line_uid, line.page_id, line.student, line.cie,
                    line.page, line.block_id, line.line_id,
                    line.reading_order, *line.bbox, line.status,
                    line.confidence, line.extra.get("router_label"),
                    line.crop, line.text,
                ])

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    booklets = defaultdict(list)

    for page in pages:
        booklets[(page.student, page.cie)].append(page)

    for (student, cie), booklet_pages in sorted(booklets.items()):

        booklet_pages.sort(key=lambda p: p.page)

        out = [
            "=" * 68,
            f"REAL OCR OUTPUT - {args.model}",
            "",
            "This is a real student's handwriting, read by a machine.",
            "Personal data: do not commit, publish or share. Text is the",
            "engine's raw output and contains recognition errors.",
            "=" * 68,
            "",
            f"student_{student:02d}  cie_{cie}",
            "",
        ]

        for page in booklet_pages:
            out.append(f"--- page {page.page:02d} ({page.page_id}) ---")
            out.append(page.text)
            out.append("")

        (TRANSCRIPT_DIR / f"student_{student:02d}_cie_{cie}.txt").write_text(
            "\n".join(out)
        )

    confidences = [l.confidence for p in pages for l in p.lines
                   if l.status == "ok"]

    print()
    print(f"Pages       : {len(pages)}")
    print(f"Lines       : {payload['lines_total']}  (validated: {checked})")
    print("Status      : " + ", ".join(
        f"{k}={counts[k]}" for k in schema.STATUSES if counts[k]))

    if confidences:
        confidences.sort()
        mid = confidences[len(confidences) // 2]
        weak = sum(1 for c in confidences if c < 0.5)
        print(f"Confidence  : median {mid:.3f}, "
              f"{weak} line(s) below 0.50")

    print(f"Time        : {elapsed:.0f}s "
          f"({elapsed / max(total, 1):.2f}s/line)")
    print()
    print(f"Output      : {OUT_DIR}")
    print(f"Transcripts : {TRANSCRIPT_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
