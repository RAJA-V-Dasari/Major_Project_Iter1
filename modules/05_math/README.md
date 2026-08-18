# Module 05 — Math OCR

Reads the regions the router addresses to `math_ocr` and returns, for each
expression, the LaTeX it read and the box on the page it came from.

```
02_segment  ->  router  ->  05_math  ->  { LaTeX + page coordinates }
                  |
                  +------->  ocr, diagram_parser, table_parser, ...
```

**Status: exploratory.** The stage runs end to end and its geometry is
correct, but no off-the-shelf recogniser tested here reads this
handwriting well enough to use. The measurements are below, and they are
the point of this branch — they say what to build next.

---

## 1. The router hand-off is real

`03_router` addresses regions to `math_ocr`, and this stage reads them
from `input/routed/` (a symlink to `03_router/output/after_ocr/`), one
file per booklet. There is no simulated router here any more — the
earlier `src/simulate_router.py` invented a label per region to drive
the old label-based `module2_router`, and both have been removed.

**The routing happens *after* recognition, not before it.** That is the
one thing to understand about this input:

```
02_segment -> 03_router -> 04_ocr -> 03_router (reroute) -> 05_math
```

Separating equations from prose by looking at the crop was measured on
600 real regions and rejected: ink density, inter-symbol gap over line
height, component-size variation and aspect ratio are all **unimodal**,
so any threshold would be an invented boundary, and the corpus has no
labelled maths to validate one against. This is the same reason
`02_segment` emits no labels at all.

After recognition the question is easy, because `=` is a character you
can read rather than a property inferred from ink statistics. So a line
goes to the text recogniser first even when it turns out to be an
equation. That wastes one recognition per equation, which is cheap, and
buys not having to guess.

A line is re-routed when it carries an operator **and** an operand
**and** is mostly non-alphabetic, or contains a word like `mod`. The
operand condition earns its place: without it a bare `=`, `*` or `->`
scores as 100% non-alphabetic and sails through — 38% of the first
run's re-routed regions were exactly that, isolated operators and
arrows rather than expressions.

| | |
|---|---|
| **Real** | The images, the page coordinates, the routing decision, and the text it was made from. |
| **As good as its text** | The maths detection reads whatever the recogniser produced. Run against `04_ocr/src/simulate.py` the text is synthetic, so the maths found is too — the payload carries `from_simulated_ocr` to say which. |

The detection rules are marked provisional in `03_router/src/config.py`:
there is no labelled maths in this corpus to tune them against, so they
are a documented starting point to be measured once real recognition
output exists.

---

## 2. The preparation front-end

The obvious pipeline is router → recogniser. Handing a crop straight to
an image-to-LaTeX model produces nonsense, and the failures are two
specific, fixable properties of the crop.

**The crop still contains the printed rules.** `02_segment` removes the
ruling from its *mask* before segmenting but deliberately never touches
the image on disk. A formula model has exactly one thing it does with a
long horizontal line — it calls it a fraction bar:

| crop | raw reading |
|---|---|
| handwritten `= 69`, rule beneath | `\frac{\tilde\varepsilon \underbrace{\xi q}}{\ldots}` |
| `10011` inside a table box | `\frac{\lim_{\theta\to1}10\theta}{\lim_{\theta\to2}1}` |

Neither crop contains a fraction. Rules are told from handwriting by
**span and thinness**, never darkness: a printed rule crosses the whole
crop and touches both borders. An erasure is applied only where the ink
is thin, so a stroke crossing the rule keeps its glyph intact.

**A line region is not an expression.** `02_segment` finds *lines*; a
recogniser wants *one formula*. A single line here routinely holds two
unrelated workings side by side. So a crop fans out to N expressions
(usually 1) and each is read separately.

Both fixes were needed together, and each one broke something first:

- erasing a rule leaves 1–2px fragments where it wobbled. Counting those,
  median glyph height came out **3px instead of 35px**, shrinking the
  split gap tenfold and cutting `= 69` into `=` and `69`;
- those same fragments then *bridged* every gap in the column profile,
  merging two workings at opposite ends of a line back into one.

Splitting and measuring therefore run on a mask with debris dropped —
while the image handed to the recogniser is cut from the full cleaned
crop, so a decimal point is still in the pixels even though it got no
vote.

**Does it work?** Yes, measurably — and not enough:

| crop | raw | prepared |
|---|---|---|
| `2^13 = ...` (right half) | nothing resembling it | `2^{13} =` read correctly |
| `10011` in a table | spurious `\frac{\lim}{\lim}` | `\frac` gone |
| `= 69` | spurious `\frac` | still `\frac` — a stray pen mark below reads as a bar |

---

## 3. What the recognisers actually do

40 routed regions → 46 expressions, same images, both engines, CPU.

### sumen-base (image → LaTeX, 349M params)

`0 empty, 0 errors, 643s (~14s/expression), mean confidence 0.87.`

It gets short arithmetic right and degrades from there:

| on the page | read as | |
|---|---|---|
| `= 256 - 1` | `= 2 5 6 - 1` | exact |
| `= 2^{8-1} = 2^7` | `= 2^{81} = 2^7` | lost the minus |
| `Source port number = (0045)_16` | `(9) Surce port numbon = (004.5)_{16}` | close |
| `Stop-and-wait (2m), m=1` | `Slop-and-wait _ (an) _ n=1` | close |
| `= 255 packets` | `= \underbrace{aSS}_{packubs}` | partial |
| `= 69` | `\frac{\varepsilon\delta 9}{\varepsilon}` | wrong |

**Its confidence does not separate right from wrong.** This is the most
important finding here and it defeats the obvious triage strategy:

```
= 256 - 1   read exactly right                        0.96
= 69        read as \frac{\varepsilon\delta 9}{...}   0.94
```

A formula model asked to read this handwriting does not fail loudly. It
fabricates fluent, well-formed LaTeX and is confident about it. Anything
built on this stage needs a different signal; today the honest one is a
human looking at `preview/`.

### trocr-base-handwritten (image → text) — the control

Included because it is not obvious a *formula* model is right for this
corpus. These are computer-networks scripts: the maths is arithmetic,
powers and base conversion — `header len = 6x4 = 24 bytes`, `(0038)_16`
— with almost no 2-D structure, and very difficult handwriting. A formula
model buys layout and pays in handwriting; a handwriting model buys the
pen and pays by flattening every superscript.

`0 empty, 0 errors, 122s (~2.7s/expression), mean confidence 0.59.`

It is also **5× faster** than sumen on identical images — 122s against
643s for the same 46 expressions — which matters on a CPU-only machine.

**The control wins, and it is not close.** Both engines on the same 19
prepared expressions of `student_01/cie_1`, ground truth read off the
review sheet by eye:

| on the page | sumen | trocr |
|---|---|---|
| `length of UDP header (fixed) = 8 bytes` | `\fbox{\begin{array}{ccc}{\dot Q}&{unglh}...` | `( D length of UDP header ( fixed ) - 8 bytes` |
| `length of the data = 88-8 = 80 bytes` | `\fbox{\begin{array}{lll}{~```~lingth~9~}...` | `" length of the data - 88-8 - 80 bytes` |
| `Total length of the user datagram = (0058)_16` | `\underbrace{\overbrace{O}^{\prime}}_{...}` | `O Total length of the user datagram - (00688 )16` |
| `= 128 packets` | `= \frac{128}{z} \cdot padits` | `" 128 packets` |
| `= 88 bytes` | `= \underbrace{88}_{\_}{b}y\mathrm{ts}` | `" 88 bytes .` |
| `before wraparound occurs = 2^{m-1}` | `\boldmath~\displaystyle~\theta~ = ...` | `before 1wraparound occurs - 2m-` |
| **`= 256 - 1`** | **`= 2 5 6 - 1`** | `-256-` |
| **`= 2^{8-1} = 2^7`** | **`= 2^{81} = 2^7`** | `-284 - 27` |

Roughly 10 of 19 clearly to TrOCR, 3 to sumen, the rest bad both ways.
The split is systematic, not noise:

- **Anything containing words** — which is most of this corpus's
  "mathematics" — sumen cannot read at all. It does not degrade
  gracefully; it emits `\fbox`, `\underbrace`, `\lim` and `\frac`
  structures that are not on the page.
- **Purely symbolic lines** — `= 256 - 1`, `= 2^{8-1} = 2^7` — sumen
  gets right and TrOCR mangles, flattening every superscript exactly as
  expected (`2^{m-1}` → `2m-`).

**TrOCR's confidence also separates, where sumen's does not**: 0.33 on
the garbage `52- lb.`, 0.89 on both near-perfect readings. That makes it
usable for triage today.

The conclusion for this project: the maths in these scripts is mostly
*prose with numbers in it*, so a handwriting recogniser is the better
base model, and the formula model earns its place only on the genuinely
two-dimensional minority. Both are still too weak to ship untouched.

---

## 4. Running

```bash
cd modules/05_math/src

cd ../../03_router/src && python reroute.py   # real router, post-OCR
python prepare.py --preview                       # check the front-end alone

python math_ocr.py --engine none                  # plumbing, no model, seconds
python math_ocr.py --engine sumen --limit 40 --preview
python math_ocr.py --engine trocr --limit 40 --preview
```

`--engine none` runs routing, preparation, splitting, geometry and output
without downloading a model. It separates the two things that can go
wrong: if the output looks wrong with `none`, the fault is in this
module; if only with a real engine, the fault is the recogniser.

Dependencies: `requirements.txt`. Model weights are fetched on first run
and cached in `~/.cache/huggingface` (1.4GB sumen, 1.3GB TrOCR).

---

## 5. Output

`output/math_ocr_<engine>.json` and `.csv`, plus `preview/` sheets.

Every routed region is kept, **including** the ones that came back empty
or unreadable — a recogniser that silently drops what it cannot read
looks far better than it is, and the count has to reconcile against the
routing.

Each reading carries its box **in page coordinates**, not crop
coordinates:

```
page_x = region.bbox.x1 - pad + expression.offset_x
```

where `pad` is `crop_lines.py`'s margin (0.10 × the page's rule pitch),
which that stage guarantees on all four sides. Verified by cutting the
original pages at the reported boxes: every cut lands exactly on its
expression. That is what makes a reading placeable back on the page — for
overlay, for document reconstruction, or for a marker checking against
the script.

`input/`, `output/` and `preview/` are gitignored. They hold real student
work.

---

## 6. What to do next

1. **Fine-tune TrOCR on this handwriting, not a formula model.** That is
   what §3 argues: the corpus is prose-with-numbers, TrOCR already reads
   whole lines of it nearly correctly, and its one systematic failure —
   flattening superscripts — is precisely what fine-tuning on
   `2^{m-1}`-style targets would fix. The bottleneck is the model, not
   the plumbing. Prepared expressions plus hand-transcribed targets are
   the dataset, and this stage already emits the images.
2. **Sharpen the maths detection.** It now runs on recognised text
   (`03_router/src/rules.py`), but its thresholds are provisional and
   untested against real OCR output. Labelling a
   few hundred line crops `equation` / `paragraph` / `diagram` / `table`
   would beat these heuristics and unblock the other processors too.
3. **Find a signal that separates a good reading from a fabricated one.**
   Sumen's token confidence does not; TrOCR's does, which is a further
   argument for it. Round-tripping the reading back to an image and
   comparing against the crop would be stronger than either.
4. **Route tables and diagrams away first.** Most surviving false
   positives are boxed table rows, and no formula model will ever read
   them.
