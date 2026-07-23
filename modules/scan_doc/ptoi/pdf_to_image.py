from pathlib import Path
import fitz  # PyMuPDF


# ==========================
# Configuration
# ==========================

INPUT_PDF = Path("input/sample.pdf")
OUTPUT_DIR = Path("output/pages")
DPI = 300


# ==========================
# PDF → Image Conversion
# ==========================

def pdf_to_images(pdf_path, output_dir, dpi=300):
    """
    Convert all pages of a PDF into PNG images.

    Args:
        pdf_path (Path): Path to input PDF
        output_dir (Path): Folder to save page images
        dpi (int): Output image resolution

    Returns:
        list[Path]: List of generated image paths
    """

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    document = fitz.open(pdf_path)

    generated_pages = []

    print(f"\nOpened PDF : {pdf_path.name}")
    print(f"Total Pages: {len(document)}\n")

    for page_number in range(len(document)):

        page = document.load_page(page_number)

        pix = page.get_pixmap(dpi=dpi)

        image_path = output_dir / f"page_{page_number + 1}.png"

        pix.save(image_path)

        generated_pages.append(image_path)

        print(f"✓ Saved {image_path.name}")

    document.close()

    return generated_pages


# ==========================
# Main
# ==========================

if __name__ == "__main__":

    print("=" * 50)
    print(" PDF TO IMAGE CONVERTER ")
    print("=" * 50)

    pages = pdf_to_images(
        INPUT_PDF,
        OUTPUT_DIR,
        DPI
    )

    print("\nConversion Complete!")
    print(f"Generated {len(pages)} page(s).")