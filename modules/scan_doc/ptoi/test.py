from pathlib import Path

from pdf_to_image import PDFToImageConverter


INPUT_PDF = Path("input/sample.pdf")
OUTPUT_FOLDER = Path("output/pages")


def main():

    converter = PDFToImageConverter(dpi=300)

    pages = converter.convert(
        INPUT_PDF,
        OUTPUT_FOLDER
    )

    print("=" * 40)
    print("Conversion Successful")
    print("=" * 40)

    print(f"Pages Generated : {len(pages)}")

    for page in pages:
        print(page)


if __name__ == "__main__":
    main()