import json
from pathlib import Path

from doclayout_yolo import YOLOv10


class DocumentSegmenter:
    def __init__(
        self,
        model_path="models/doclayout_yolo_docstructbench_imgsz1024.pt",
        input_dir="input/pages",
        output_dir="output",
    ):
        self.model = YOLOv10(model_path)

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.json_path = self.output_dir / "regions.json"

    def detect_page(self, image_path):
        results = self.model.predict(
            str(image_path),
            imgsz=1024,
            conf=0.20,
            device="cpu",      # change to cuda:0 if using GPU
            verbose=False
        )

        result = results[0]

        regions = []

        region_id = 1

        if result.boxes is None:
            return regions

        for box in result.boxes:

            cls = int(box.cls.item())
            label = result.names[cls]

            confidence = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            regions.append(
                {
                    "id": region_id,
                    "label": label,
                    "confidence": round(confidence, 4),
                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                }
            )

            region_id += 1

        return regions

    def run(self):

        document = {
            "pages": []
        }

        images = sorted(self.input_dir.glob("*.png"))

        if not images:
            print("No images found.")
            return

        for page_number, image in enumerate(images, start=1):

            print(f"Processing {image.name}")

            page = {
                "page": page_number,
                "image": image.name,
                "regions": self.detect_page(image),
            }

            document["pages"].append(page)

        with open(self.json_path, "w") as f:
            json.dump(document, f, indent=4)

        print()
        print(f"Saved: {self.json_path}")


if __name__ == "__main__":

    segmenter = DocumentSegmenter()

    segmenter.run()