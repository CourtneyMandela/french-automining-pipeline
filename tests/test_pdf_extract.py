from fpdf import FPDF

from french_mining.lingq.pdf_extract import extract_text, extract_text_from_folder


def _write_pdf(path, text: str) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(str(path))


def test_extract_text_reads_pdf_content(tmp_path):
    pdf_path = tmp_path / "reading.pdf"
    _write_pdf(pdf_path, "Le chat noir dort sur le canape.")
    text = extract_text(pdf_path)
    assert "chat noir" in text


def test_extract_text_from_folder_keys_by_filename(tmp_path):
    _write_pdf(tmp_path / "a.pdf", "Premiere lecture de la semaine.")
    _write_pdf(tmp_path / "b.pdf", "Deuxieme lecture de la semaine.")
    (tmp_path / "not_a_pdf.txt").write_text("ignore me")

    result = extract_text_from_folder(tmp_path)

    assert set(result.keys()) == {"a.pdf", "b.pdf"}
    assert "Premiere" in result["a.pdf"]
    assert "Deuxieme" in result["b.pdf"]
