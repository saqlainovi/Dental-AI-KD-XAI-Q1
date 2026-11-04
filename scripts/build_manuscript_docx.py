from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


PROJECT_ROOT = Path(r"H:\OneDrive\WORK\RECHARCH TEAM\DENTAL")
PACKAGE_ROOT = Path(r"E:\dental_ai_results\paper_package_v1")
OUTPUT_ROOT = Path(r"E:\dental_ai_results\manuscript_docx_v1")
OUTPUT_DOCX = OUTPUT_ROOT / "Dental_AI_Manuscript_Draft_v1.docx"

FIGURE_ROOT = PACKAGE_ROOT / "figures"
TABLE_ROOT = PACKAGE_ROOT / "tables"
MANUSCRIPT_ROOT = PACKAGE_ROOT / "manuscript"

BLUE = RGBColor(46, 116, 181)
DARK_BLUE = RGBColor(31, 77, 120)
INK = RGBColor(11, 37, 69)
MUTED = RGBColor(90, 97, 110)
LIGHT_GRAY_HEX = "F2F4F7"
CALL_OUT_HEX = "F4F6F9"
BORDER_HEX = "D9DEE7"

DISPLAY_LABELS = {
    "rows_or_images": "rows / images",
    "raw_rows": "raw",
    "clean_rows": "clean",
    "raw_tooth_fraction_mean": "raw tooth frac",
    "clean_tooth_fraction_mean": "clean tooth frac",
    "mean_confidence": "confidence",
    "fraction_reduction": "fraction drop",
    "params_million": "params (M)",
    "checkpoint_mb": "ckpt (MB)",
    "dentex_val_dice": "DENTEX Dice",
    "dentex_val_iou": "DENTEX IoU",
    "images_per_second": "img/s",
    "best_val_accuracy": "best val acc",
    "mean_dice": "mean Dice",
    "mean_iou": "mean IoU",
    "kept_images": "kept images",
    "failure_rate_dice_lt_0_7": "fail rate Dice < 0.7",
}


def set_run_font(run, size=None, bold=None, italic=None, color=None, name="Calibri"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def paragraph_spacing(paragraph, before=0, after=6, line=1.10):
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line


def set_cell_shading(cell, fill_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill_hex)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin_name, value in {
        "top": top,
        "start": start,
        "bottom": bottom,
        "end": end,
    }.items():
        node = tc_mar.find(qn(f"w:{margin_name}"))
        if node is None:
            node = OxmlElement(f"w:{margin_name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_width(table, width_dxa=9360, indent_dxa=120):
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(width_dxa))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")


def set_col_widths(table, widths_inches):
    for row in table.rows:
        for idx, width in enumerate(widths_inches):
            if idx >= len(row.cells):
                continue
            cell = row.cells[idx]
            cell.width = Inches(width)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(width * 1440)))
            tc_w.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def add_page_number(paragraph):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)


def add_horizontal_rule(paragraph, color="2E74B5", size="8", space="1"):
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), space)
    bottom.set(qn("w:color"), color)
    p_bdr.append(bottom)
    p_pr.append(p_bdr)


def add_paragraph(doc, text="", style=None, before=0, after=6, line=1.10):
    paragraph = doc.add_paragraph(style=style)
    if text:
        paragraph.add_run(text)
    paragraph_spacing(paragraph, before=before, after=after, line=line)
    return paragraph


def add_heading(doc, text, level):
    paragraph = doc.add_heading(text, level=level)
    paragraph.paragraph_format.keep_with_next = True
    return paragraph


def add_caption(doc, text):
    paragraph = doc.add_paragraph(style="Caption")
    paragraph.paragraph_format.keep_with_next = False
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.paragraph_format.line_spacing = 1.10
    run = paragraph.add_run(text)
    set_run_font(run, size=9.5, italic=True, color=MUTED)
    return paragraph


def add_callout(doc, title, body):
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    set_table_width(table, width_dxa=9360, indent_dxa=120)
    cell = table.cell(0, 0)
    set_cell_margins(cell)
    set_cell_shading(cell, CALL_OUT_HEX)
    paragraph = cell.paragraphs[0]
    paragraph_spacing(paragraph, before=0, after=2, line=1.10)
    title_run = paragraph.add_run(title)
    set_run_font(title_run, size=10.5, bold=True, color=INK)
    paragraph.add_run(" ")
    body_run = paragraph.add_run(body)
    set_run_font(body_run, size=10.5, color=INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_numbered_takeaways(doc, items):
    for item in items:
        paragraph = doc.add_paragraph(style="List Number")
        paragraph_spacing(paragraph, before=0, after=4, line=1.167)
        run = paragraph.add_run(item)
        set_run_font(run, size=10.5)


def add_bullet_items(doc, items):
    for item in items:
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph_spacing(paragraph, before=0, after=4, line=1.167)
        run = paragraph.add_run(item)
        set_run_font(run, size=10.5)


def read_csv_table(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        return list(reader)


def format_cell(value):
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    if abs(number) >= 100:
        return str(int(number)) if number.is_integer() else f"{number:.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}".rstrip("0").rstrip(".")
    if abs(number) >= 1:
        return f"{number:.3f}".rstrip("0").rstrip(".")
    return f"{number:.4f}".rstrip("0").rstrip(".")


def format_header(value):
    text = str(value)
    return DISPLAY_LABELS.get(text, text.replace("_", " "))


def format_body_text(value):
    text = str(value)
    replacements = {
        "quadrant_full": "quadrant",
        "enumeration_full": "enumeration",
        "disease_full": "disease",
        "validation_full": "validation",
        "best_val_accuracy": "best val acc",
        "mean_dice": "mean Dice",
    }
    return replacements.get(text, text)


def add_csv_table(doc, csv_path, caption, widths):
    rows = read_csv_table(csv_path)
    if not rows:
        return
    add_caption(doc, caption)
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    set_table_width(table, width_dxa=9360, indent_dxa=120)
    set_col_widths(table, widths)
    set_repeat_table_header(table.rows[0])

    for row_idx, row in enumerate(rows):
        for col_idx, value in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if row_idx == 0:
                set_cell_shading(cell, LIGHT_GRAY_HEX)
            paragraph = cell.paragraphs[0]
            paragraph_spacing(paragraph, before=0, after=0, line=1.05)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx > 1 else WD_ALIGN_PARAGRAPH.LEFT
            display_value = format_header(value) if row_idx == 0 else format_body_text(format_cell(value))
            run = paragraph.add_run(display_value)
            set_run_font(run, size=8.5, bold=(row_idx == 0), color=INK if row_idx == 0 else None)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_figure(doc, image_path, caption, width=6.35, page_break_before=False):
    if page_break_before:
        doc.add_page_break()
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.add_run().add_picture(str(image_path), width=Inches(width))
    add_caption(doc, caption)


def read_markdown_body(path):
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#")).strip()


def configure_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    title = styles["Title"]
    title.font.name = "Calibri"
    title._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    title._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    title.font.size = Pt(22)
    title.font.bold = True
    title.font.color.rgb = INK
    title.paragraph_format.space_after = Pt(8)

    subtitle = styles["Subtitle"]
    subtitle.font.name = "Calibri"
    subtitle._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    subtitle._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    subtitle.font.size = Pt(12)
    subtitle.font.italic = True
    subtitle.font.color.rgb = MUTED
    subtitle.paragraph_format.space_after = Pt(12)

    for style_name, size, color, before, after in [
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ]:
        style = styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    caption = styles["Caption"]
    caption.font.name = "Calibri"
    caption._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    caption._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    caption.font.size = Pt(9.5)
    caption.font.italic = True
    caption.font.color.rgb = MUTED


def configure_sections(doc):
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.right_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    header = section.header.paragraphs[0]
    header.text = ""
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = header.add_run("Dental AI manuscript draft")
    set_run_font(run, size=9, color=MUTED)
    add_horizontal_rule(header, color="D9DEE7", size="4", space="1")

    footer = section.footer.paragraphs[0]
    footer.text = ""
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = footer.add_run("Page ")
    set_run_font(run, size=9, color=MUTED)
    add_page_number(footer)


def add_title_page(doc):
    p = doc.add_paragraph()
    paragraph_spacing(p, before=8, after=2, line=1.10)
    kicker = p.add_run("Research Manuscript Draft")
    set_run_font(kicker, size=11, bold=True, color=BLUE)

    title = doc.add_paragraph(style="Title")
    title.paragraph_format.space_before = Pt(8)
    title.paragraph_format.space_after = Pt(4)
    title.add_run(
        "Explainable Teacher-Student Segmentation for Panoramic Dental X-ray Analysis"
    )

    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle.add_run(
        "Pseudo-mask supervision, uncertainty-aware reporting, and external TUFTS validation"
    )

    metadata = [
        ("Project", "DENTAL AI research package"),
        ("Prepared from", str(PACKAGE_ROOT)),
        ("Generated", date.today().isoformat()),
        ("Primary outputs", "DENTEX pseudo-mask validation, TUFTS external validation, XAI and uncertainty figures"),
        ("Status", "Draft for supervisor/journal preparation; not yet final submission formatting"),
    ]
    table = doc.add_table(rows=len(metadata), cols=2)
    table.style = "Table Grid"
    set_table_width(table, width_dxa=9360, indent_dxa=120)
    set_col_widths(table, [1.55, 4.95])
    for row_idx, (label, value) in enumerate(metadata):
        label_cell = table.cell(row_idx, 0)
        value_cell = table.cell(row_idx, 1)
        for cell in [label_cell, value_cell]:
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_shading(label_cell, LIGHT_GRAY_HEX)
        label_run = label_cell.paragraphs[0].add_run(label)
        set_run_font(label_run, size=9.5, bold=True, color=INK)
        value_run = value_cell.paragraphs[0].add_run(value)
        set_run_font(value_run, size=9.5, color=INK)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    add_horizontal_rule(p, color="2E74B5", size="8", space="1")
    add_callout(
        doc,
        "Core message.",
        "The strongest current claim is not disease classification; it is an explainable, lightweight X-ray segmentation workflow that transfers TUFTS tooth-mask supervision into DENTEX pseudo-mask learning and validates externally on TUFTS.",
    )
    doc.add_page_break()


def add_abstract(doc):
    add_heading(doc, "Abstract", 1)
    abstract = read_markdown_body(MANUSCRIPT_ROOT / "abstract_draft.md")
    add_paragraph(doc, abstract)
    add_paragraph(
        doc,
        "Keywords: panoramic radiograph; dental X-ray; teacher-student learning; pseudo masks; explainable AI; uncertainty; external validation.",
        style=None,
        after=10,
    )


def add_introduction(doc):
    add_heading(doc, "1. Introduction", 1)
    add_paragraph(
        doc,
        "Panoramic dental radiographs are central to routine screening, treatment planning, and longitudinal dental assessment. Recent dental AI benchmarks have accelerated automatic tooth detection, enumeration, and abnormality recognition, but most systems still return model outputs that are difficult for clinicians to interrogate. For clinical translation, a useful model must show not only what it predicts, but also where the supporting image evidence lies and when the prediction is uncertain.",
    )
    add_paragraph(
        doc,
        "The DENTEX MICCAI 2023 benchmark provides tooth-level bounding boxes, enumeration, and diagnostic labels for panoramic X-rays, while the TUFTS Dental Database provides an external panoramic source with expert tooth masks. These datasets create an opportunity to study a pragmatic supervision pathway: use reliable X-ray segmentation masks from one source to create pseudo-mask supervision for another source where dense pixel masks are limited.",
    )
    add_paragraph(
        doc,
        "This work therefore reframes the project around a focused and currently evidence-backed contribution: a teacher-student segmentation pipeline with pseudo-mask generation, quality control, external validation, and dentist-readable visual outputs. Classification baselines are retained as comparison evidence, but the manuscript's main technical claim is the segmentation and XAI workflow.",
    )
    add_heading(doc, "1.1 Contributions", 2)
    add_numbered_takeaways(
        doc,
        [
            "A TUFTS-trained X-ray teacher model that transfers dense tooth-mask supervision into DENTEX panoramic images through cleaned pseudo masks.",
            "A lightweight student segmentation model that preserves strong DENTEX pseudo-validation performance with a 0.118M-parameter checkpoint.",
            "External TUFTS validation reported with explicit handling of empty target masks and non-empty-mask performance.",
            "A reviewer-facing XAI package including saliency maps, uncertainty maps, error overlays, failure panels, and reliability filtering.",
            "Ablation evidence showing the practical value of model compression, pseudo-mask quality control, and uncertainty-aware filtering.",
        ],
    )


def add_related_work(doc):
    add_heading(doc, "2. Related Work", 1)
    add_heading(doc, "2.1 Dental Enumeration and Diagnosis Benchmarks", 2)
    add_paragraph(
        doc,
        "DENTEX established a hierarchical panoramic X-ray benchmark covering quadrant segmentation, tooth enumeration, and abnormal tooth diagnosis. The benchmark is useful for disease-aware dental AI because it links tooth localization, FDI-style enumeration, and four diagnostic labels. The challenge report also emphasizes that top systems used specialized strategies rather than a single generic model, which supports the need for task-specific pipelines.",
    )
    add_paragraph(
        doc,
        "Challenge solution papers such as the integrated segmentation and detection approach for DENTEX demonstrate that segmentation-guided reasoning is an accepted direction for panoramic dental X-rays. In this project, the classification baselines are therefore treated as comparison evidence, while the main method focuses on segmentation, pseudo-mask transfer, and XAI output.",
    )

    add_heading(doc, "2.2 Public Dataset Combination and Annotation Quality", 2)
    add_paragraph(
        doc,
        "Recent panoramic radiograph studies have combined public datasets to reduce annotation burden and broaden task coverage. Falk et al. combined tooth segmentation and diagnosis resources and explicitly noted that public annotation quality can vary, recommending quality assurance to reduce model bias. Our pseudo-mask cleanup and reliability filtering follow the same practical concern: generated masks must be checked and gated before they become supervision.",
    )
    add_paragraph(
        doc,
        "The TUFTS Dental Database is especially valuable for this project because it provides 1,000 panoramic radiographs with expert labeling of abnormalities and teeth. This makes it better suited than RGB intraoral datasets for an X-ray teacher and for external validation.",
    )

    add_heading(doc, "2.3 Tooth Segmentation, Lightweight Models, and Distillation", 2)
    add_paragraph(
        doc,
        "Tooth segmentation is commonly positioned as a prerequisite for downstream numbering and diagnosis. Recent work has explored transformer, SAM/SAM2-based, and lightweight U-Net-style approaches. The SAM2 tooth segmentation literature motivates teacher-student transfer for dental panoramic X-rays under limited annotation, while the broader distillation literature motivates compressing a stronger model into a smaller deployable student.",
    )

    add_heading(doc, "2.4 Explainability and Uncertainty", 2)
    add_paragraph(
        doc,
        "Explainability methods such as Grad-CAM and feature-attribution approaches are widely used to make deep models more reviewable. In this manuscript, XAI is scoped to segmentation behavior: saliency, uncertainty, prediction overlays, and error maps are shown at the image level so reviewers can inspect where the model succeeds or fails.",
    )


def add_methods(doc):
    add_heading(doc, "3. Materials and Methods", 1)
    add_heading(doc, "3.1 Datasets", 2)
    add_paragraph(
        doc,
        "Three public sources were prepared for the project. DENTEX is the main in-domain panoramic X-ray benchmark and provides task-specific labels for quadrant, enumeration, disease, and validation splits. DentalAI supplies RGB intraoral segmentation annotations and was retained as a reference source, but it was not used as the final X-ray teacher because the modality gap caused weak transfer. TUFTS provides panoramic X-rays with tooth masks and was used both for teacher training and for external validation.",
    )
    add_csv_table(
        doc,
        TABLE_ROOT / "Table1_dataset_summary.csv",
        "Table 1. Dataset roles and available image/annotation counts used in the final package.",
        [1.15, 2.35, 1.2, 1.8],
    )

    add_heading(doc, "3.2 Preprocessing and Pseudo-mask Generation", 2)
    add_paragraph(
        doc,
        read_markdown_body(MANUSCRIPT_ROOT / "methods_summary_draft.md"),
    )
    add_paragraph(
        doc,
        "DENTEX pseudo masks were post-processed to reduce border artifacts and implausible tooth-area predictions. The retained masks form the supervision set for the student model and the validation target for DENTEX pseudo-validation.",
    )
    add_csv_table(
        doc,
        TABLE_ROOT / "Table4_pseudo_mask_cleanup.csv",
        "Table 2. Pseudo-mask cleanup statistics across DENTEX task folders.",
        [1.2, 0.8, 0.8, 0.7, 0.7, 1.0, 1.0, 0.95, 1.0],
    )
    add_figure(
        doc,
        FIGURE_ROOT / "Fig8_pseudo_mask_cleanup.png",
        "Figure 1. Raw versus cleaned pseudo-mask tooth-area statistics.",
        width=5.7,
    )

    add_heading(doc, "3.3 Teacher and Student Models", 2)
    add_paragraph(
        doc,
        "The final training path uses a compact U-Net family rather than a heavy transformer-only stack because the current objective is deployable segmentation with explicit uncertainty reporting. The TUFTS teacher uses a wider Tiny U-Net configuration, while the student uses a narrower configuration trained on cleaned DENTEX pseudo masks. Training used CUDA automatic mixed precision and checkpoints were written to the E drive to avoid OneDrive write failures.",
    )
    add_csv_table(
        doc,
        TABLE_ROOT / "Table3_teacher_student_deployment.csv",
        "Table 3. Teacher-student deployment comparison.",
        [1.7, 0.9, 0.9, 1.0, 1.0, 1.0],
    )

    add_heading(doc, "3.4 Explainability and Uncertainty", 2)
    add_paragraph(
        doc,
        "Explainability outputs were generated from the trained student model. Each panel combines the image, target or reference mask where available, predicted mask, error map, gradient saliency, and softmax-entropy uncertainty. These outputs are designed to make the segmentation behavior reviewable rather than treating Dice and IoU as the only evidence.",
    )
    add_heading(doc, "3.5 Evaluation Protocol", 2)
    add_paragraph(
        doc,
        "Segmentation quality is reported with Dice coefficient and IoU. Pixel-level ROC AUC is reported as an auxiliary threshold-sweep measure. DENTEX evaluation uses the cleaned pseudo-validation masks, while TUFTS is treated as external validation. Because 32 TUFTS masks are empty, external validation is reported both on all rows and on the 968 non-empty targets.",
    )


def add_results(doc):
    add_heading(doc, "4. Results", 1)
    add_paragraph(doc, read_markdown_body(MANUSCRIPT_ROOT / "results_summary_draft.md"))
    add_csv_table(
        doc,
        TABLE_ROOT / "Table2_main_results.csv",
        "Table 4. Main segmentation results.",
        [1.65, 1.75, 0.75, 0.75, 1.6],
    )
    add_figure(
        doc,
        FIGURE_ROOT / "Fig1_method_pipeline.png",
        "Figure 2. Overview of the proposed Dental AI teacher-student XAI pipeline.",
        width=6.2,
    )
    add_figure(
        doc,
        FIGURE_ROOT / "Fig2_training_curves.png",
        "Figure 3. Student training loss and pseudo-mask validation Dice/IoU across epochs.",
        width=6.2,
    )
    add_figure(
        doc,
        FIGURE_ROOT / "Fig3_confusion_and_roc.png",
        "Figure 4. Segmentation confusion matrices and pixel-level ROC curves for DENTEX validation and TUFTS external testing.",
        width=6.2,
    )

    add_heading(doc, "4.1 Baseline Comparisons", 2)
    add_paragraph(
        doc,
        "Crop-level Swin-T classifier baselines were trained for quadrant, enumeration, and disease tasks. These baselines are useful for comparison, but they do not provide the segmentation masks, saliency overlays, uncertainty maps, or deployable mask output required by the final research objective.",
    )
    add_csv_table(
        doc,
        TABLE_ROOT / "Table6_baseline_vs_proposed.csv",
        "Table 5. Baseline classifier results compared with the proposed teacher-student segmentation workflow.",
        [1.75, 1.25, 1.05, 0.8, 1.65],
    )

    add_heading(doc, "4.2 XAI and Failure Analysis", 2)
    add_paragraph(
        doc,
        "The DENTEX and TUFTS explanation panels show model behavior across successful and difficult examples. DENTEX panels are tied to pseudo-validation targets, while TUFTS panels provide external behavior against expert tooth masks. The failure panel surfaces the weakest non-empty TUFTS examples and should guide future data curation and model refinement.",
    )
    add_figure(
        doc,
        FIGURE_ROOT / "Fig4_dentex_xai_uncertainty.png",
        "Figure 5. DENTEX validation examples with target masks, predictions, error maps, gradient saliency, and uncertainty maps.",
        width=6.35,
    )
    add_figure(
        doc,
        FIGURE_ROOT / "Fig5_tufts_xai_uncertainty.png",
        "Figure 6. TUFTS external examples with prediction, saliency, uncertainty, and error visualization.",
        width=6.35,
    )
    add_figure(
        doc,
        FIGURE_ROOT / "Fig6_tufts_failure_cases.png",
        "Figure 7. Lowest-Dice non-empty TUFTS failure cases.",
        width=6.35,
    )

    add_heading(doc, "4.3 Reliability Filtering and Ablation Evidence", 2)
    add_paragraph(
        doc,
        "Reliability filtering uses the model's own uncertainty summary to retain higher-confidence cases. At 85 percent retained coverage, the TUFTS non-empty mean Dice increased from 0.8758 to 0.8888 and the Dice < 0.7 failure rate decreased from 0.0248 to 0.0146.",
    )
    add_csv_table(
        doc,
        TABLE_ROOT / "Table5_reliability_filtering.csv",
        "Table 6. Reliability filtering on TUFTS non-empty masks.",
        [0.8, 0.95, 1.0, 1.0, 1.75],
    )
    add_figure(
        doc,
        FIGURE_ROOT / "Fig7_ablation_summary.png",
        "Figure 8. Teacher-student deployment comparison and reliability filtering curve.",
        width=6.2,
    )
    add_figure(
        doc,
        FIGURE_ROOT / "Fig9_augmentation_examples.png",
        "Figure 9. Image augmentation examples used during student training.",
        width=6.2,
    )


def add_discussion(doc):
    add_heading(doc, "5. Discussion", 1)
    add_paragraph(
        doc,
        "The results support a practical direction for the paper: the strongest contribution is a deployable segmentation and XAI workflow, not a broad claim of solved multi-pathology diagnosis. The classifier baselines show that crop-level disease recognition remains comparatively weak, especially for disease classes, while the segmentation pathway produces stronger, more interpretable evidence.",
    )
    add_paragraph(
        doc,
        "The TUFTS-trained teacher and DENTEX student arrangement also avoids the modality mismatch observed with DentalAI RGB intraoral images. This is an important methodological lesson: X-ray-to-X-ray transfer was more reliable for panoramic segmentation than direct transfer from RGB intraoral masks to DENTEX panoramic X-rays.",
    )
    add_paragraph(
        doc,
        "The external TUFTS validation result is encouraging because the student was trained on DENTEX pseudo masks but retained strong performance on non-empty TUFTS masks. However, this should be framed as external segmentation generalization, not final clinical diagnostic validation.",
    )

    add_heading(doc, "5.1 Novelty Claims Supported by Current Evidence", 2)
    add_csv_table(
        doc,
        TABLE_ROOT / "Table7_novelty_claims.csv",
        "Table 7. Novelty claims and current evidence.",
        [2.05, 4.45],
    )

    add_heading(doc, "5.2 Limitations", 2)
    add_paragraph(doc, read_markdown_body(MANUSCRIPT_ROOT / "limitations_draft.md"))

    add_heading(doc, "5.3 Next Work Before Submission", 2)
    add_bullet_items(
        doc,
        [
            "Add a stronger journal-formatted related-work section with verified DOI metadata.",
            "Optionally retrain strict ablations without pseudo-mask cleanup and without uncertainty filtering if the target journal requests causal ablation evidence.",
            "Improve disease-level classification or frame classification as a secondary baseline rather than a primary contribution.",
            "Add calibration metrics such as expected calibration error if uncertainty becomes a central claim.",
            "Convert this draft to the exact template of the selected Q1 target journal once the supervisor approves the evidence framing.",
        ],
    )


def add_conclusion_and_refs(doc):
    add_heading(doc, "6. Conclusion", 1)
    add_paragraph(
        doc,
        "This project now has a coherent evidence-backed manuscript direction. A TUFTS X-ray teacher generated cleaned DENTEX pseudo masks, a lightweight DENTEX student preserved strong pseudo-validation segmentation performance, and external TUFTS validation remained robust on non-empty masks. The final figure package provides XAI, uncertainty, failure-case, reliability, and ablation views that make the system reviewer-readable and clinically interpretable at the prototype level.",
    )
    add_heading(doc, "References", 1)
    references = [
        'I. E. Hamamci et al., "DENTEX: Dental Enumeration and Tooth Pathosis Detection Benchmark for Panoramic X-ray," arXiv:2305.19112, 2023, doi: 10.48550/arXiv.2305.19112.',
        'DENTEX Challenge Organizers, "Dental Enumeration and Diagnosis on Panoramic X-rays Challenge," Zenodo, 2023, doi: 10.5281/zenodo.7812323.',
        'L. He, Y. Liu, and L. Wang, "Integrated Segmentation and Detection Models for DENTEX Challenge 2023," arXiv:2308.14161, 2023, doi: 10.48550/arXiv.2308.14161.',
        'S. Falk et al., "Combining Public Datasets for Automated Tooth Assessment in Panoramic Radiographs," BMC Oral Health, 2024, doi: 10.1186/s12903-024-04129-5.',
        'K. Panetta et al., "The Tufts Dental Database," Tufts University, accessed 2026-05-18, https://tdd.ece.tufts.edu/.',
        'Z. Li, W. Tang, S. Gao, Y. Wang, and S. Wang, "Adapting SAM2 Model from Natural Images for Tooth Segmentation in Dental Panoramic X-Ray Images," Entropy, vol. 26, no. 12, 1059, 2024, doi: 10.3390/e26121059.',
        'Z. Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," ICCV, 2021, doi: 10.1109/ICCV48922.2021.00986.',
        'A. Kirillov et al., "Segment Anything," ICCV, 2023.',
        'R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," ICCV, 2017, doi: 10.1109/ICCV.2017.74.',
        'S. M. Lundberg and S. I. Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS, 2017.',
        'T. L. Koch, M. Perslev, C. Igel, and S. S. Brandt, "Lightweight Deep Learning Methods for Panoramic Dental X-ray Image Segmentation," Neural Computing and Applications, 2022.',
        'G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," arXiv:1503.02531, 2015.',
    ]
    for idx, ref in enumerate(references, start=1):
        paragraph = add_paragraph(doc, f"[{idx}] {ref}", after=3, line=1.05)
        paragraph.paragraph_format.left_indent = Inches(0.28)
        paragraph.paragraph_format.first_line_indent = Inches(-0.28)


def build_document():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    doc = Document()
    configure_sections(doc)
    configure_styles(doc)

    add_title_page(doc)
    add_abstract(doc)
    add_introduction(doc)
    add_related_work(doc)
    add_methods(doc)
    add_results(doc)
    add_discussion(doc)
    add_conclusion_and_refs(doc)

    doc.core_properties.title = "Explainable Teacher-Student Segmentation for Panoramic Dental X-ray Analysis"
    doc.core_properties.subject = "Dental AI manuscript draft"
    doc.core_properties.author = "Dental AI Research Team"
    doc.core_properties.keywords = "dental X-ray, segmentation, teacher-student learning, pseudo masks, XAI"
    doc.save(OUTPUT_DOCX)

    manifest = {
        "docx": str(OUTPUT_DOCX),
        "source_package": str(PACKAGE_ROOT),
        "preset": "standard_business_brief",
        "generated": date.today().isoformat(),
    }
    (OUTPUT_ROOT / "manuscript_docx_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return OUTPUT_DOCX


if __name__ == "__main__":
    print(build_document())
