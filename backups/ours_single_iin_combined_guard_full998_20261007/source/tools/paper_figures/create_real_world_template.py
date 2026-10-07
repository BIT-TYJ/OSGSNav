#!/usr/bin/env python3
"""Create the editable TarGSNav real-world experiment figure template."""

from pathlib import Path
from xml.sax.saxutils import escape


OUT_DIR = Path(__file__).resolve().parents[2] / "detectron2" / "docs" / "targsnav_figures"
SVG_PATH = OUT_DIR / "Fig7_real_world_template.svg"
FODP_PATH = OUT_DIR / "Fig7_real_world_template.fodp"
PPTX_PATH = OUT_DIR / "Fig7_real_world_template.pptx"

W, H = 3840, 2160

COLORS = {
    "ink": "#18222D",
    "muted": "#65727E",
    "line": "#245B8A",
    "observation": "#DFF2F5",
    "exploration": "#E7F4CF",
    "exploration_dark": "#BFDFA0",
    "localization": "#EEE8F6",
    "fine": "#FFF0C8",
    "white": "#FFFFFF",
    "row_alt": "#FAFCFD",
    "magenta": "#F04B87",
    "case_a": "#78B957",
    "case_b": "#4C9AC8",
    "case_c": "#E8A247",
    "case_d": "#9A79C2",
}


def svg_rect(x, y, w, h, fill, stroke=None, sw=4, rx=24, dash=None):
    attrs = [
        f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"',
        f'rx="{rx}"', f'fill="{fill}"',
    ]
    if stroke:
        attrs.extend([f'stroke="{stroke}"', f'stroke-width="{sw}"'])
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    return f"<rect {' '.join(attrs)}/>"


def svg_text(x, y, text, size=42, weight="normal", anchor="middle", fill=None):
    fill = fill or COLORS["ink"]
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Comic Sans MS, DejaVu Sans, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">'
        f'{escape(text)}</text>'
    )


def svg_multiline(x, y, lines, size=34, leading=1.18, weight="normal", fill=None):
    fill = fill or COLORS["muted"]
    texts = []
    for index, line in enumerate(lines):
        line_y = y + index * size * leading
        texts.append(svg_text(x, line_y, line, size, weight, "middle", fill))
    return "".join(texts)


def svg_image_placeholder(x, y, w, h, label, fill="#FFFFFF", compact=False):
    parts = [svg_rect(x, y, w, h, fill, COLORS["ink"], 3, 18, "13 11")]
    icon_w = min(92, w * 0.18)
    icon_h = min(64, h * 0.28)
    cx = x + w / 2
    icon_y = y + h * (0.33 if not compact else 0.28)
    parts.append(
        f'<rect x="{cx-icon_w/2}" y="{icon_y-icon_h/2}" width="{icon_w}" '
        f'height="{icon_h}" rx="7" fill="none" stroke="{COLORS["muted"]}" stroke-width="3"/>'
    )
    parts.append(
        f'<circle cx="{cx+icon_w*0.22}" cy="{icon_y-icon_h*0.18}" r="7" '
        f'fill="{COLORS["muted"]}"/>'
    )
    points = [
        (cx-icon_w*0.38, icon_y+icon_h*0.30),
        (cx-icon_w*0.12, icon_y+icon_h*0.02),
        (cx+icon_w*0.06, icon_y+icon_h*0.20),
        (cx+icon_w*0.28, icon_y-icon_h*0.02),
        (cx+icon_w*0.40, icon_y+icon_h*0.30),
    ]
    parts.append(
        '<polyline points="{}" fill="none" stroke="{}" stroke-width="3"/>'.format(
            " ".join(f"{px},{py}" for px, py in points), COLORS["muted"]
        )
    )
    label_y = y + h * (0.73 if not compact else 0.70)
    parts.append(svg_text(cx, label_y, label, 28 if compact else 32, "bold", fill=COLORS["muted"]))
    parts.append(svg_text(cx, label_y + 36, "DROP IMAGE HERE", 19 if compact else 21, fill=COLORS["muted"]))
    return "".join(parts)


def build_svg():
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        svg_text(80, 86, "Real-World Deployment", 58, "bold", "start"),
        svg_text(W-80, 84, "TarGSNav", 34, "bold", "end", COLORS["line"]),
    ]

    # Platform overview.
    parts.append(svg_rect(60, 120, W-120, 470, COLORS["observation"], COLORS["ink"], 4, 30, "16 12"))
    parts.append(svg_text(100, 176, "A  Robotic Setup", 38, "bold", "start"))
    setup_x = [110, 1045, 1980]
    setup_w = [855, 855, 1750]
    setup_titles = ["Robot Platform", "Sensor Suite", "Test Environment"]
    setup_fills = [COLORS["fine"], COLORS["observation"], COLORS["exploration"]]
    for x, width, title, fill in zip(setup_x, setup_w, setup_titles, setup_fills):
        parts.append(svg_rect(x, 205, width, 330, fill, COLORS["ink"], 3, 22))
        parts.append(svg_image_placeholder(x+24, 232, width-48, 236, title))
        parts.append(svg_text(x+width/2, 507, title, 27, "bold"))

    # Experiment cases.
    parts.append(svg_rect(60, 635, W-120, 1460, COLORS["white"], COLORS["ink"], 4, 30, "16 12"))
    parts.append(svg_text(100, 694, "B  Navigation Trials", 38, "bold", "start"))

    left = 95
    widths = [175, 510, 510, 600, 550, 510, 585]
    gaps = 18
    xs = [left]
    for width in widths[:-1]:
        xs.append(xs[-1] + width + gaps)
    headers = [
        ("Case", COLORS["white"]),
        ("Image Goal", COLORS["observation"]),
        ("Initial Observation", COLORS["observation"]),
        ("Exploration & Localization", COLORS["exploration"]),
        ("Navigation Trajectory", COLORS["localization"]),
        ("Final Observation", COLORS["fine"]),
        ("Outcome", COLORS["fine"]),
    ]
    for x, width, (label, fill) in zip(xs, widths, headers):
        parts.append(svg_rect(x, 725, width, 94, fill, COLORS["ink"], 3, 18))
        if label == "Exploration & Localization":
            parts.append(svg_multiline(x+width/2, 768, ["Exploration &", "Localization"], 26, 1.05, "bold", COLORS["ink"]))
        else:
            parts.append(svg_text(x+width/2, 785, label, 28, "bold"))

    row_top = 842
    row_h = 292
    row_gap = 18
    case_colors = [COLORS["case_a"], COLORS["case_b"], COLORS["case_c"], COLORS["case_d"]]
    for index in range(4):
        y = row_top + index * (row_h + row_gap)
        parts.append(svg_rect(82, y, W-164, row_h, COLORS["row_alt"] if index % 2 else COLORS["white"], "#D1D8DE", 2, 22))
        parts.append(svg_rect(xs[0]+15, y+22, widths[0]-30, row_h-44, case_colors[index], None, rx=28))
        parts.append(svg_text(xs[0]+widths[0]/2, y+121, f"Case {chr(65+index)}", 33, "bold", fill=COLORS["white"]))
        parts.append(svg_multiline(xs[0]+widths[0]/2, y+164, ["Target: ____", "Difficulty: ____"], 18, 1.45, fill=COLORS["white"]))
        labels = ["GOAL", "START", "ACTIVE MAPPING", "PATH", "FINAL VIEW"]
        fills = [COLORS["observation"], COLORS["observation"], COLORS["exploration"], COLORS["localization"], COLORS["fine"]]
        for col in range(1, 6):
            parts.append(svg_image_placeholder(xs[col]+13, y+20, widths[col]-26, row_h-40, labels[col-1], fills[col-1], compact=True))
        ox, ow = xs[6]+15, widths[6]-30
        parts.append(svg_rect(ox, y+20, ow, row_h-40, COLORS["fine"], COLORS["ink"], 3, 18, "11 9"))
        parts.append(svg_text(ox+28, y+72, "Success:  —", 27, "bold", "start"))
        parts.append(svg_text(ox+28, y+121, "Final distance:  —  m", 24, anchor="start"))
        parts.append(svg_text(ox+28, y+168, "Steps:  —", 24, anchor="start"))
        parts.append(svg_text(ox+28, y+215, "Time:  —  s", 24, anchor="start"))

    parts.append(svg_text(80, H-28, "Replace dashed boxes with real experiment images; keep case colors and column order.", 22, anchor="start", fill=COLORS["muted"]))
    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def fodp_shape(x, y, w, h, text, style="box", para="PBody"):
    return f'''<draw:custom-shape draw:style-name="{style}" draw:text-style-name="{para}" svg:x="{x}in" svg:y="{y}in" svg:width="{w}in" svg:height="{h}in"><text:p text:style-name="{para}">{escape(text)}</text:p><draw:enhanced-geometry draw:type="round-rect"/></draw:custom-shape>'''


def build_fodp():
    # A deliberately simple, fully editable PowerPoint counterpart of the SVG.
    styles = {
        "setup": COLORS["observation"],
        "explore": COLORS["exploration"],
        "localize": COLORS["localization"],
        "fine": COLORS["fine"],
        "white": COLORS["white"],
        "caseA": COLORS["case_a"],
        "caseB": COLORS["case_b"],
        "caseC": COLORS["case_c"],
        "caseD": COLORS["case_d"],
    }
    graphic_styles = []
    for name, color in styles.items():
        graphic_styles.append(
            f'''<style:style style:name="{name}" style:family="graphic"><style:graphic-properties draw:stroke="solid" svg:stroke-width="0.025in" svg:stroke-color="{COLORS['ink']}" draw:fill="solid" draw:fill-color="{color}" draw:textarea-horizontal-align="center" draw:textarea-vertical-align="middle" fo:padding="0.05in"/></style:style>'''
        )
    graphic_styles.append(
        f'''<style:style style:name="placeholder" style:family="graphic"><style:graphic-properties draw:stroke="dash" draw:stroke-dash="Dash" svg:stroke-width="0.025in" svg:stroke-color="{COLORS['ink']}" draw:fill="solid" draw:fill-color="#FFFFFF" draw:textarea-horizontal-align="center" draw:textarea-vertical-align="middle" fo:padding="0.04in"/></style:style>'''
    )
    graphic_styles.append(
        '''<style:style style:name="none" style:family="graphic"><style:graphic-properties draw:stroke="none" draw:fill="none" draw:textarea-horizontal-align="left" draw:textarea-vertical-align="middle"/></style:style>'''
    )

    shapes = []
    shapes.append(fodp_shape(0.35, 0.10, 12.0, 0.42, "Real-World Deployment", "none", "PTitle"))
    shapes.append(fodp_shape(0.25, 0.60, 15.50, 2.02, "", "setup"))
    shapes.append(fodp_shape(0.48, 0.73, 3.70, 0.32, "A  Robotic Setup", "none", "PHead"))
    top_boxes = [(0.48, 1.08, 3.70, "ROBOT PLATFORM\n\nDROP IMAGE HERE"), (4.35, 1.08, 3.70, "SENSOR SUITE\n\nDROP IMAGE HERE"), (8.22, 1.08, 7.28, "TEST ENVIRONMENT\n\nDROP IMAGE HERE")]
    for x, y, w, text in top_boxes:
        shapes.append(fodp_shape(x, y, w, 1.25, text, "placeholder", "PPlace"))

    shapes.append(fodp_shape(0.25, 2.82, 15.50, 6.07, "", "white"))
    shapes.append(fodp_shape(0.48, 2.91, 4.2, 0.32, "B  Navigation Trials", "none", "PHead"))
    col_x = [0.40, 1.17, 3.30, 5.43, 7.93, 10.23, 12.36]
    col_w = [0.67, 2.03, 2.03, 2.40, 2.20, 2.03, 3.08]
    headers = ["Case", "Image Goal", "Initial Observation", "Exploration &\nLocalization", "Navigation Trajectory", "Final Observation", "Outcome"]
    header_styles = ["white", "setup", "setup", "explore", "localize", "fine", "fine"]
    for x, w, label, style in zip(col_x, col_w, headers, header_styles):
        shapes.append(fodp_shape(x, 3.30, w, 0.44, label, style, "PHeader"))

    row_y = [3.88, 5.10, 6.32, 7.54]
    case_styles = ["caseA", "caseB", "caseC", "caseD"]
    stage_labels = ["GOAL\nDROP IMAGE HERE", "START\nDROP IMAGE HERE", "ACTIVE MAPPING\nDROP IMAGE HERE", "PATH\nDROP IMAGE HERE", "FINAL VIEW\nDROP IMAGE HERE"]
    for idx, y in enumerate(row_y):
        shapes.append(fodp_shape(col_x[0], y, col_w[0], 1.05, f"Case {chr(65+idx)}\nTarget: ____", case_styles[idx], "PCase"))
        for col in range(1, 6):
            shapes.append(fodp_shape(col_x[col], y, col_w[col], 1.05, stage_labels[col-1], "placeholder", "PPlaceSmall"))
        shapes.append(fodp_shape(col_x[6], y, col_w[6], 1.05, "Success: —\nFinal distance: — m\nSteps: —     Time: — s", "fine", "PMetric"))

    content = f'''<?xml version="1.0" encoding="UTF-8"?>
<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0" office:version="1.2" office:mimetype="application/vnd.oasis.opendocument.presentation">
<office:font-face-decls><style:font-face style:name="Comic Sans MS" svg:font-family="Comic Sans MS"/></office:font-face-decls>
<office:styles>
<draw:stroke-dash draw:name="Dash" draw:style="rect" draw:dots1="1" draw:dots1-length="0.10in" draw:distance="0.07in"/>
{''.join(graphic_styles)}
<style:style style:name="PTitle" style:family="paragraph"><style:paragraph-properties fo:text-align="start"/><style:text-properties style:font-name="Comic Sans MS" fo:font-size="26pt" fo:font-weight="bold" fo:color="{COLORS['ink']}"/></style:style>
<style:style style:name="PHead" style:family="paragraph"><style:paragraph-properties fo:text-align="start"/><style:text-properties style:font-name="Comic Sans MS" fo:font-size="17pt" fo:font-weight="bold" fo:color="{COLORS['ink']}"/></style:style>
<style:style style:name="PHeader" style:family="paragraph"><style:paragraph-properties fo:text-align="center"/><style:text-properties style:font-name="Comic Sans MS" fo:font-size="11pt" fo:font-weight="bold" fo:color="{COLORS['ink']}"/></style:style>
<style:style style:name="PPlace" style:family="paragraph"><style:paragraph-properties fo:text-align="center"/><style:text-properties style:font-name="Comic Sans MS" fo:font-size="13pt" fo:font-weight="bold" fo:color="{COLORS['muted']}"/></style:style>
<style:style style:name="PPlaceSmall" style:family="paragraph"><style:paragraph-properties fo:text-align="center"/><style:text-properties style:font-name="Comic Sans MS" fo:font-size="9pt" fo:font-weight="bold" fo:color="{COLORS['muted']}"/></style:style>
<style:style style:name="PCase" style:family="paragraph"><style:paragraph-properties fo:text-align="center"/><style:text-properties style:font-name="Comic Sans MS" fo:font-size="10pt" fo:font-weight="bold" fo:color="#FFFFFF"/></style:style>
<style:style style:name="PMetric" style:family="paragraph"><style:paragraph-properties fo:text-align="start"/><style:text-properties style:font-name="Comic Sans MS" fo:font-size="10pt" fo:color="{COLORS['ink']}"/></style:style>
<style:style style:name="PBody" style:family="paragraph"><style:paragraph-properties fo:text-align="center"/><style:text-properties style:font-name="Comic Sans MS" fo:font-size="11pt" fo:color="{COLORS['ink']}"/></style:style>
</office:styles>
<office:automatic-styles><style:page-layout style:name="PM1"><style:page-layout-properties fo:page-width="16in" fo:page-height="9in" style:print-orientation="landscape" presentation:display-footer="false" presentation:display-page-number="false" presentation:display-date-time="false"/></style:page-layout><style:style style:name="dp1" style:family="drawing-page"><style:drawing-page-properties draw:fill="solid" draw:fill-color="#FFFFFF"/></style:style></office:automatic-styles>
<office:master-styles><style:master-page style:name="Default" style:page-layout-name="PM1" draw:style-name="dp1"/></office:master-styles>
<office:body><office:presentation><draw:page draw:name="Slide 1" draw:style-name="dp1" draw:master-page-name="Default">{''.join(shapes)}</draw:page></office:presentation></office:body>
</office:document>'''
    FODP_PATH.write_text(content, encoding="utf-8")


def build_pptx():
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.dml import MSO_LINE_DASH_STYLE
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Inches, Pt

    def rgb(hex_color):
        value = hex_color.lstrip("#")
        return RGBColor(*(int(value[index:index + 2], 16) for index in (0, 2, 4)))

    presentation = Presentation()
    presentation.slide_width = Inches(16)
    presentation.slide_height = Inches(9)
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])

    def add_box(
        x, y, w, h, text="", fill="#FFFFFF", *,
        font_size=11, bold=False, color=None, align=PP_ALIGN.CENTER,
        line_color=None, dashed=False, radius=True, margin=0.06,
    ):
        shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
        shape = slide.shapes.add_shape(
            shape_type, Inches(x), Inches(y), Inches(w), Inches(h)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
        shape.line.color.rgb = rgb(line_color or COLORS["ink"])
        shape.line.width = Pt(1.25)
        if dashed:
            shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        text_frame = shape.text_frame
        text_frame.clear()
        text_frame.margin_left = Inches(margin)
        text_frame.margin_right = Inches(margin)
        text_frame.margin_top = Inches(margin)
        text_frame.margin_bottom = Inches(margin)
        text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        paragraph = text_frame.paragraphs[0]
        paragraph.alignment = align
        paragraph.text = text
        for run in paragraph.runs:
            run.font.name = "Comic Sans MS"
            run.font.size = Pt(font_size)
            run.font.bold = bold
            run.font.color.rgb = rgb(color or COLORS["ink"])
        return shape

    def add_text(x, y, w, h, text, font_size=16, bold=False, color=None, align=PP_ALIGN.LEFT):
        box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        frame = box.text_frame
        frame.clear()
        frame.margin_left = frame.margin_right = 0
        frame.margin_top = frame.margin_bottom = 0
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        paragraph = frame.paragraphs[0]
        paragraph.alignment = align
        paragraph.text = text
        for run in paragraph.runs:
            run.font.name = "Comic Sans MS"
            run.font.size = Pt(font_size)
            run.font.bold = bold
            run.font.color.rgb = rgb(color or COLORS["ink"])
        return box

    add_text(0.35, 0.08, 11.5, 0.42, "Real-World Deployment", 25, True)
    add_text(13.5, 0.10, 2.1, 0.35, "TarGSNav", 15, True, COLORS["line"], PP_ALIGN.RIGHT)

    add_box(0.25, 0.58, 15.50, 2.04, "", COLORS["observation"], line_color=COLORS["ink"], dashed=True)
    add_text(0.47, 0.67, 4.2, 0.34, "A  Robotic Setup", 16, True)
    top_boxes = [
        (0.48, 1.06, 3.70, "ROBOT PLATFORM\n\nDROP IMAGE HERE", COLORS["fine"]),
        (4.35, 1.06, 3.70, "SENSOR SUITE\n\nDROP IMAGE HERE", COLORS["observation"]),
        (8.22, 1.06, 7.28, "TEST ENVIRONMENT\n\nDROP IMAGE HERE", COLORS["exploration"]),
    ]
    for x, y, w, label, fill in top_boxes:
        add_box(x, y, w, 1.30, label, fill, font_size=13, bold=True, color=COLORS["muted"], dashed=True)

    add_box(0.25, 2.82, 15.50, 6.05, "", COLORS["white"], line_color=COLORS["ink"], dashed=True)
    add_text(0.47, 2.89, 4.2, 0.35, "B  Navigation Trials", 16, True)

    col_x = [0.40, 1.17, 3.30, 5.43, 7.93, 10.23, 12.36]
    col_w = [0.67, 2.03, 2.03, 2.40, 2.20, 2.03, 3.08]
    headers = [
        "Case", "Image Goal", "Initial Observation",
        "Exploration &\nLocalization", "Navigation Trajectory",
        "Final Observation", "Outcome",
    ]
    header_fills = [
        COLORS["white"], COLORS["observation"], COLORS["observation"],
        COLORS["exploration"], COLORS["localization"], COLORS["fine"], COLORS["fine"],
    ]
    for x, w, label, fill in zip(col_x, col_w, headers, header_fills):
        add_box(x, 3.29, w, 0.46, label, fill, font_size=10.5, bold=True)

    row_y = [3.88, 5.10, 6.32, 7.54]
    case_colors = [COLORS["case_a"], COLORS["case_b"], COLORS["case_c"], COLORS["case_d"]]
    stage_labels = [
        "GOAL\n\nDROP IMAGE HERE", "START\n\nDROP IMAGE HERE",
        "ACTIVE MAPPING\n\nDROP IMAGE HERE", "PATH\n\nDROP IMAGE HERE",
        "FINAL VIEW\n\nDROP IMAGE HERE",
    ]
    stage_fills = [
        COLORS["observation"], COLORS["observation"], COLORS["exploration"],
        COLORS["localization"], COLORS["fine"],
    ]
    for index, y in enumerate(row_y):
        add_box(
            col_x[0], y, col_w[0], 1.05,
            f"Case {chr(65 + index)}\nTarget: ____\nDifficulty: ____",
            case_colors[index], font_size=8.5, bold=True, color=COLORS["white"],
        )
        for col in range(1, 6):
            add_box(
                col_x[col], y, col_w[col], 1.05, stage_labels[col - 1],
                stage_fills[col - 1], font_size=8.5, bold=True,
                color=COLORS["muted"], dashed=True,
            )
        add_box(
            col_x[6], y, col_w[6], 1.05,
            "Success:  —\nFinal distance:  —  m\nSteps:  —      Time:  —  s",
            COLORS["fine"], font_size=9.5, align=PP_ALIGN.LEFT, margin=0.16,
        )

    add_text(
        0.38, 8.88, 15.0, 0.10,
        "Replace dashed boxes with real experiment images; keep case colors and column order.",
        8.5, color=COLORS["muted"],
    )
    presentation.save(str(PPTX_PATH))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_svg()
    build_fodp()
    build_pptx()
    print(SVG_PATH)
    print(FODP_PATH)
    print(PPTX_PATH)


if __name__ == "__main__":
    main()
