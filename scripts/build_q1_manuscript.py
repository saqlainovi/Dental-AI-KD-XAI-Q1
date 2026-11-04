"""
Q1 Journal Manuscript Builder - Complete Publication Suite.
Generates:
1. Dental_AI_Q1_Journal_Manuscript_V15.md
2. Dental_AI_Q1_Journal_Manuscript_V15.docx
"""
import os, sys, docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

WORKSPACE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
OUTPUTS = os.path.join(WORKSPACE, "outputs")
PAPER_FIGS = os.path.join(OUTPUTS, "paper_figures")

doc = docx.Document()

# Page Margins: 1 inch standard
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

# Set base styles
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(11)
style.font.color.rgb = RGBColor(30, 30, 30)
style.paragraph_format.line_spacing = 1.15
style.paragraph_format.space_after = Pt(4)

md_lines = []

def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.bold = True
    run.font.size = Pt(18)
    run.font.color.rgb = RGBColor(26, 54, 93) # Navy
    p.paragraph_format.space_after = Pt(12)
    md_lines.append(f"# {text}\n")

def add_authors(authors, affil):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p.add_run(authors + "\n")
    r1.font.size = Pt(11)
    r1.font.bold = True
    r2 = p.add_run(affil)
    r2.font.size = Pt(9.5)
    r2.font.italic = True
    p.paragraph_format.space_after = Pt(16)
    md_lines.append(f"**{authors}**  \n*{affil}*\n\n---\n")

def add_sec_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    prefix = "#" * (level + 1)
    md_lines.append(f"\n{prefix} {text}\n")

def add_body(text):
    p = doc.add_paragraph(text)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    md_lines.append(f"{text}\n")

def set_cell_background(cell, fill_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_color)
    tcPr.append(shd)

def add_styled_table(headers, rows, caption=""):
    if caption:
        p_cap = doc.add_paragraph()
        run_cap = p_cap.add_run(caption)
        run_cap.font.bold = True
        run_cap.font.size = Pt(10)
        p_cap.paragraph_format.space_after = Pt(3)
        md_lines.append(f"\n**{caption}**\n")

    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'
    
    # Format header
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], '1A365D')
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.font.size = Pt(9.5)
            
    # Format rows
    for r_idx, row in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = 'F7FAFC' if r_idx % 2 == 1 else 'FFFFFF'
        for c_idx, val in enumerate(row):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(40, 40, 40)
    
    doc.add_paragraph() # Spacing
    
    # Markdown Table
    md_header = "| " + " | ".join(headers) + " |"
    md_sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    md_rows = ["| " + " | ".join([str(v) for v in r]) + " |" for r in rows]
    md_lines.append("\n" + "\n".join([md_header, md_sep] + md_rows) + "\n")

def add_figure(img_path, caption):
    if os.path.exists(img_path):
        doc.add_picture(img_path, width=Inches(6.2))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = p_cap.add_run(caption)
        run_cap.font.italic = True
        run_cap.font.size = Pt(9.5)
        p_cap.paragraph_format.space_after = Pt(8)
    md_lines.append(f"\n![{caption}]({img_path})\n*{caption}*\n")

print("Part 1 Core Definitions loaded.")
