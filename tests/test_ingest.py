from io import BytesIO

import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from ragdemo.config import Settings
from ragdemo.errors import DemoError
from ragdemo.ingest import read_pdf, read_text


def make_pdf(*, scan=False, mixed=False):
    output = BytesIO()
    canvas = Canvas(output)
    canvas.bookmarkPage("refunds")
    canvas.addOutlineEntry("Refunds", "refunds", level=0)
    canvas.drawString(72, 750, "Refund requests are accepted within 30 days.")
    if scan or mixed:
        if mixed:
            canvas.showPage()
        else:
            # The separate image-only page is still detected even after a valid text page.
            canvas.showPage()
        image = Image.new("RGB", (100, 100), "white")
        canvas.drawImage(ImageReader(image), 72, 400, width=300, height=300)
        if mixed:
            canvas.drawString(72, 750, "Scanned page")
    canvas.save()
    return output.getvalue()


def test_pdf_conversion_and_outline():
    pages = read_pdf(make_pdf(), Settings(api_key=""))
    assert len(pages) == 1
    assert "30 days" in pages[0].text
    assert pages[0].number == 1
    assert pages[0].outline == ("Refunds",)


@pytest.mark.parametrize("kwargs", [{"scan": True}, {"mixed": True}])
def test_scanned_or_mixed_pdf_rejected(kwargs):
    with pytest.raises(DemoError, match="scanned|image-only"):
        read_pdf(make_pdf(**kwargs), Settings(api_key=""))


def test_encrypted_pdf_rejected():
    writer = PdfWriter()
    writer.append(PdfReader(BytesIO(make_pdf())))
    writer.encrypt("not-a-real-secret")
    output = BytesIO()
    writer.write(output)
    with pytest.raises(DemoError, match="Encrypted"):
        read_pdf(output.getvalue(), Settings(api_key=""))


def test_bad_size_page_count_and_malformed_inputs():
    settings = Settings(api_key="")
    with pytest.raises(DemoError, match="valid PDF"):
        read_pdf(b"not a pdf", settings)
    with pytest.raises(DemoError, match="could not be read"):
        read_pdf(b"%PDF-broken", settings)
    with pytest.raises(DemoError, match="10 MB"):
        read_pdf(b"%PDF-" + b"a" * settings.max_pdf_bytes, settings)
    writer = PdfWriter()
    for _ in range(101):
        writer.add_blank_page(width=100, height=100)
    output = BytesIO()
    writer.write(output)
    with pytest.raises(DemoError, match="100 pages"):
        read_pdf(output.getvalue(), settings)
    with pytest.raises(DemoError):
        read_text(" ", settings)
    with pytest.raises(DemoError, match="characters"):
        read_text("a" * (settings.max_chars + 1), settings)
