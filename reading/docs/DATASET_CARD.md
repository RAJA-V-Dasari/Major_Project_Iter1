---
license: other
pretty_name: Handwritten Answer Scripts (Cleaned)
tags:
  - document-layout-analysis
  - handwriting
  - ocr
---

# Handwritten Answer Scripts — cleaned

Scanned CIE answer booklets: rotation-corrected, cropped to a uniform
physical page size, and tone-normalised. An intermediate stage of an
in-progress pipeline — de-ruling is not applied yet, so the printed
ruled lines are still present (deliberately: the de-ruling stage needs
them).

**Private dataset. Contains personal data** — student names, USNs,
signatures and marks appear on the cover sheets. Do not redistribute
or make public.

## Structure

```
student_<NN>/
    cie_<M>/
        page_01.png
        page_02.png
        ...
```

Same student/CIE/page numbering as the raw dataset
(`prss-majorproject-37/Handwritten-AnswerScripts-MajorProject`), minus
one page (`student_19/cie_2/page_14`) - a genuine ~37%-scale scan
outlier, excluded rather than force-fit to the uniform size.

All pages are 1598x2177, 8-bit greyscale.

## Processing so far

1. **Deskew** — page rotation corrected using the printed rule-line
   angle (Hough transform).
2. **Crop** — every page cropped to a fixed 1598x2177, anchored per
   page off the detected physical paper edge. A small safety margin
   reduces (but does not fully remove, on heavily-bound pages) the
   booklet's binding seam, chosen to never cut real handwritten
   content over minimizing seam residue.
3. **Flatten + tone** — each pixel divided by a local background
   estimate, then a linear stretch putting ink at black and paper at
   white. Paper now sits at 255 on every page.

   This also removes most bleed-through (show-through from the reverse
   side), which was the dominant defect. Note that it is *not* removed
   by brightness: light real strokes and bleed-through overlap in
   intensity, and page histograms have no valley between ink and
   paper. What separates them is sharpness — bleed-through has
   diffused through the paper and sits close to its local surroundings,
   so dividing by a local background estimate pushes it to near-white
   while sharp strokes survive.

   Kept greyscale rather than binarised: stroke weight carries
   information, and a later stage can always threshold this, but
   cannot recover what a threshold discarded.

## Known gaps

- `student_18` … `student_61` sat only some CIEs; missing folders are
  genuine (the exam was not written), not lost data.
- `student_19/cie_2/page_14` is absent (see above).
- The binding seam still shows on heavily-bound pages, and faint
  bleed-through residue remains on a few.
