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


def test_pdf_font_headings_preserve_company_section_separate_from_instructions(budget):
    from ragdemo.chunking import make_chunks

    output = BytesIO()
    canvas = Canvas(output)
    canvas.setFont("Helvetica", 21)
    canvas.drawString(72, 750, "System design challenge")
    canvas.setFont("Helvetica", 12)
    canvas.drawString(72, 710, "How the call works")
    canvas.setFont("Helvetica", 10)
    canvas.drawString(
        72, 685, "Have a diagram ready. This paragraph describes the interview process."
    )
    canvas.setFont("Helvetica", 12)
    canvas.drawString(72, 650, "The scenario")
    canvas.setFont("Helvetica", 10)
    canvas.drawString(72, 625, "VeloNord is a fictional e-bike company with about 800 employees.")
    canvas.drawString(72, 605, "The company sells through bike dealers and uses Microsoft 365.")
    canvas.setFont("Helvetica", 12)
    canvas.drawString(72, 565, "Your task")
    canvas.setFont("Helvetica", 10)
    canvas.drawString(72, 540, "Design a system that helps people find important changes early.")
    canvas.save()
    pages = read_pdf(output.getvalue(), Settings(api_key=""))
    assert "###" not in pages[0].text  # Only two detected heading sizes in this fixture.
    assert "## The scenario" in pages[0].text
    chunks, _ = make_chunks(pages, budget)
    definition = next(chunk for chunk in chunks if "800 employees" in chunk.text)
    assert definition.section.endswith("The scenario")
    assert "Have a diagram" not in definition.text
    assert "Design a system" not in definition.text


def test_missing_pdf_layout_metadata_falls_back_to_complete_text(monkeypatch):
    def broken_layout(*args, **kwargs):
        raise ValueError("Missing font metadata")

    monkeypatch.setattr("ragdemo.ingest.pdfplumber.open", broken_layout)
    pages = read_pdf(make_pdf(), Settings(api_key=""))
    assert "30 days" in pages[0].text


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
