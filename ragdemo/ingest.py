from io import BytesIO

from markitdown import MarkItDown, StreamInfo
from pypdf import PdfReader, PdfWriter

from ragdemo.config import Settings
from ragdemo.errors import DemoError
from ragdemo.models import Page


def read_text(text: str, settings: Settings) -> list[Page]:
    if not text.strip():
        raise DemoError("Paste some reference text before indexing.")
    if len(text) > settings.max_chars:
        raise DemoError(f"Reference text must contain at most {settings.max_chars:,} characters.")
    return [Page(text)]


def read_pdf(data: bytes, settings: Settings) -> list[Page]:
    if len(data) > settings.max_pdf_bytes:
        raise DemoError("Choose a PDF no larger than 10 MB.")
    if not data.startswith(b"%PDF-"):
        raise DemoError("The uploaded file is not a valid PDF.")
    try:
        reader = PdfReader(BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise DemoError("Encrypted PDFs are unsupported. Upload an unencrypted text PDF.")
        if not 0 < len(reader.pages) <= settings.max_pdf_pages:
            raise DemoError("Choose a PDF containing between 1 and 100 pages.")
        outlines: dict[int, list[str]] = {}

        def collect(items: list) -> None:
            for item in items:
                if isinstance(item, list):
                    collect(item)
                else:
                    index = reader.get_destination_page_number(item)
                    if index is not None and index >= 0:
                        outlines.setdefault(index, []).append(str(item.title))

        collect(reader.outline)
        converter = MarkItDown(enable_plugins=False)
        pages = []
        total = 0
        for index, page in enumerate(reader.pages):
            extracted = page.extract_text() or ""
            images = list(page.images)
            # A blank page is fine; an image-bearing page without meaningful text is not.
            # Short OCR/text overlays on a large scanned image are conservatively rejected.
            if images and len("".join(extracted.split())) < 100:
                raise DemoError(
                    f"Page {index + 1} appears scanned or image-only. Upload a fully text-based PDF; OCR is not included."
                )
            if not extracted.strip():
                continue
            output = BytesIO()
            writer = PdfWriter()
            writer.add_page(page)
            writer.write(output)
            output.seek(0)
            converted = converter.convert_stream(
                output, stream_info=StreamInfo(extension=".pdf", mimetype="application/pdf")
            ).text_content
            if not converted.strip():
                raise DemoError(
                    f"Text extraction failed on page {index + 1}. Try a simpler text PDF."
                )
            total += len(converted)
            if total > settings.max_chars:
                raise DemoError(f"Extracted PDF text exceeds {settings.max_chars:,} characters.")
            pages.append(Page(converted, index + 1, tuple(outlines.get(index, []))))
        if not pages:
            raise DemoError("No selectable text was found. Upload a text PDF; OCR is not included.")
        return pages
    except DemoError:
        raise
    except Exception:
        raise DemoError(
            "The PDF could not be read safely. Try an unencrypted, valid text PDF."
        ) from None
