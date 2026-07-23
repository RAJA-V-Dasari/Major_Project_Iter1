# Module 2 - Region Router

## Purpose

Receives segmented regions from Module 1.

Validates them.

Determines reading order.

Assigns each region to the correct downstream processor.

Outputs a structured JSON for OCR, Math OCR, Diagram Parsing, etc.

---

## Pipeline

```
Segmentation JSON
        ↓
Validation
        ↓
Reading Order
        ↓
Processor Assignment
        ↓
Routed JSON
```

---

## Run

```bash
pip install -r requirements.txt

python main.py
```

---

## Output

```
outputs/routed_regions.json
```