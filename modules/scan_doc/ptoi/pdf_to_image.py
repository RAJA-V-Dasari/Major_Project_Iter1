from pathlib import Path
import fitz


class PDFToImageConverter:
    def __init__(self, dpi=300):
        self.dpi = dpi

    def convert(self, input_pdf, output_folder):
        input_pdf = Path(input_pdf)
        output_folder = Path(output_folder)

        if not input_pdf.exists():
            raise FileNotFoundError(f"{input_pdf} not found.")

        output_folder.mkdir(parents=True, exist_ok=True)

        document = fitz.open(input_pdf)

        generated_files = []

        for page_number, page in enumerate(document, start=1):
            pix = page.get_pixmap(dpi=self.dpi)

            output_file = output_folder / f"page_{page_number}.png"

            pix.save(output_file)

            generated_files.append(output_file)

        document.close()

        return generated_files