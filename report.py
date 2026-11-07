"""
PDF downloads for the Cancer Check app (needs: pip install fpdf2).

patient_pdf()  one-page summary for a single patient
batch_pdf()    summary of an uploaded CSV of many patients
Both return the PDF as bytes, ready for st.download_button.
"""

from datetime import datetime
from fpdf import FPDF

TEAL = (15, 118, 110)
INK = (16, 41, 44)
MUTED = (77, 100, 103)
LEVEL_COLOURS = {
    "high": ((253, 236, 234), (145, 32, 24)),
    "mid": ((254, 243, 231), (147, 55, 13)),
    "low": ((232, 245, 238), (5, 96, 58)),
}
DISCLAIMER = ("Confidence shows how sure the models are. It is not the real-world chance of having cancer. "
              "This summary comes from a student research tool (Final Year Project, Universiti Teknologi "
              "PETRONAS). It cannot diagnose cancer. Always follow advice from a qualified doctor.")


def _clean(text):
    """The built-in PDF fonts only support Latin-1, so swap other characters."""
    swaps = {"−": "-", "–": "-", "—": "-", "±": "+/-", "’": "'",
             "‘": "'", "“": '"', "”": '"', "·": "-", "…": "..."}
    for a, b in swaps.items():
        text = str(text).replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


class _Doc(FPDF):
    def __init__(self, title):
        super().__init__(format="A4")
        self.doc_title = title
        self.set_auto_page_break(True, margin=18)
        self.set_margins(18, 16, 18)

    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*TEAL)
        self.cell(0, 8, "Cancer Check", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*MUTED)
        self.cell(0, 5, _clean(f"Created {datetime.now():%d %B %Y, %H:%M}"), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*TEAL)
        self.set_line_width(0.6)
        self.line(18, self.get_y() + 2, 192, self.get_y() + 2)
        self.ln(7)

    def footer(self):
        self.set_y(-16)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 5, _clean(f"Page {self.page_no()}"), align="R")

    def h1(self, text):
        self.set_font("Helvetica", "B", 17)
        self.set_text_color(*INK)
        self.multi_cell(0, 8, _clean(text), new_x="LMARGIN", new_y="NEXT")

    def h2(self, text):
        self.ln(3)
        self.set_font("Helvetica", "B", 11.5)
        self.set_text_color(*INK)
        self.cell(0, 7, _clean(text), new_x="LMARGIN", new_y="NEXT")

    def para(self, text, size=10, colour=INK):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*colour)
        self.multi_cell(0, 5.2, _clean(text), new_x="LMARGIN", new_y="NEXT")

    def verdict(self, level, big, title, text):
        fill, ink = LEVEL_COLOURS.get(level, LEVEL_COLOURS["mid"])
        y = self.get_y() + 2
        self.set_fill_color(*fill)
        self.rect(18, y, 174, 24, style="F")
        self.set_xy(22, y + 4)
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*ink)
        self.cell(34, 10, _clean(big))
        self.set_xy(58, y + 3)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 7, _clean(title), new_x="LMARGIN", new_y="NEXT")
        self.set_x(58)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(130, 5, _clean(text))
        self.set_y(y + 28)

    def table(self, headers, rows, widths):
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(*MUTED)
        for h, w in zip(headers, widths):
            self.cell(w, 7, _clean(h), border="B")
        self.ln()
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*INK)
        for row in rows:
            for cell, w in zip(row, widths):
                self.cell(w, 6.5, _clean(cell)[:48], border="B")
            self.ln()


def patient_pdf(cancer_name, patient_line, level, verdict_title, verdict_text, avg_conf,
                model_rows, reasons, answers, steps):
    pdf = _Doc("summary")
    pdf.add_page()
    pdf.h1(f"{cancer_name} check: summary")
    pdf.para(patient_line, size=10, colour=MUTED)
    pdf.verdict(level, f"{avg_conf:.0%}", verdict_title, verdict_text)

    pdf.h2("Each model")
    pdf.table(["Model", "Answer", "Confidence"], model_rows, [70, 70, 34])

    if reasons:
        pdf.h2("Main reasons (compared with a typical patient)")
        for line in reasons:
            pdf.para(f"- {line}", size=10)

    pdf.h2("Answers given")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*INK)
    col_w = 58
    for i, (k, v) in enumerate(answers):
        pdf.cell(col_w, 5.6, _clean(f"{k}: {v}")[:40])
        if i % 3 == 2:
            pdf.ln()
    pdf.ln(6)

    pdf.h2("What to do next")
    for i, line in enumerate(steps, 1):
        pdf.para(f"{i}. {line}", size=10)

    pdf.ln(4)
    pdf.para(DISCLAIMER, size=8.5, colour=MUTED)
    return bytes(pdf.output())


def batch_pdf(cancer_name, file_name, counts, rows, total, accuracy_line=None):
    pdf = _Doc("batch")
    pdf.add_page()
    pdf.h1(f"{cancer_name}: results for {total} patients")
    pdf.para(f"File: {file_name}", size=10, colour=MUTED)
    pdf.ln(2)
    pdf.para(f"High concern: {counts['high']}    Some concern: {counts['mid']}    Low concern: {counts['low']}", size=11)
    if accuracy_line:
        pdf.para(accuracy_line, size=10, colour=MUTED)
    pdf.h2("Patients, sorted by concern")
    pdf.table(["Patient", "Models saying cancer", "Avg confidence", "Concern"], rows, [44, 46, 36, 48])
    if len(rows) < total:
        pdf.para(f"Showing {len(rows)} of {total} patients. The CSV download has all of them.", size=9, colour=MUTED)
    pdf.ln(4)
    pdf.para(DISCLAIMER, size=8.5, colour=MUTED)
    return bytes(pdf.output())
