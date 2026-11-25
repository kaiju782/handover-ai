"""
exporters.py — 최종 인수인계서 내보내기
======================================
Markdown 문자열을 그대로 .md로, 또는 Word(.docx)로 변환한다.
표(| a | b |)는 실제 Word 표로, 제목(#)은 Word 제목 스타일로 바꾼다.
"""

from __future__ import annotations

import io
import re
from datetime import datetime

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


def markdown_bytes(md: str) -> bytes:
    return md.encode("utf-8")


def _set_korean_font(run, name: str = "맑은 고딕", size: float | None = None,
                     bold: bool = False, color: tuple[int, int, int] | None = None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size:
        run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def _inline_runs(paragraph, text: str, base_size: float = 10.5):
    """**굵게**, *기울임*, `코드` 정도만 처리한다."""
    tokens = re.split(r"(\*\*.+?\*\*|\*.+?\*|`.+?`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            _set_korean_font(paragraph.add_run(tok[2:-2]), size=base_size, bold=True)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = paragraph.add_run(tok[1:-1])
            _set_korean_font(r, size=base_size)
            r.italic = True
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            r = paragraph.add_run(tok[1:-1])
            _set_korean_font(r, name="Consolas", size=base_size - 0.5)
        else:
            _set_korean_font(paragraph.add_run(tok), size=base_size)


_TABLE_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?$")


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def docx_bytes(md: str, title: str = "업무 인수인계서") -> bytes:
    doc = Document()

    # 기본 스타일 글꼴
    style = doc.styles["Normal"]
    style.font.name = "맑은 고딕"
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "맑은 고딕")

    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # 표: 연속된 | ... | 줄
        if stripped.startswith("|") and "|" in stripped[1:]:
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = [r for r in block if not _TABLE_SEP.match(r)]
            if rows:
                header = _split_row(rows[0])
                body = [_split_row(r) for r in rows[1:]]
                cols = max(len(header), max((len(b) for b in body), default=0))
                table = doc.add_table(rows=1, cols=cols)
                table.style = "Table Grid"
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                for c in range(cols):
                    cell = table.rows[0].cells[c]
                    cell.text = ""
                    p = cell.paragraphs[0]
                    _inline_runs(p, header[c] if c < len(header) else "", base_size=9.5)
                    for run in p.runs:
                        run.bold = True
                for b in body:
                    cells = table.add_row().cells
                    for c in range(cols):
                        cells[c].text = ""
                        _inline_runs(cells[c].paragraphs[0], b[c] if c < len(b) else "", base_size=9.5)
            continue

        # 제목
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            if level == 1:
                h = doc.add_heading(level=1)
                _set_korean_font(h.add_run(text.replace("**", "")), size=18, bold=True)
                h.paragraph_format.space_after = Pt(6)
            else:
                h = doc.add_heading(level=min(level, 3))
                _set_korean_font(h.add_run(text.replace("**", "")), size={2: 14, 3: 12}.get(level, 11), bold=True)
            i += 1
            continue

        # 인용문
        if stripped.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Pt(12)
            _inline_runs(p, stripped.lstrip("> ").strip(), base_size=9)
            for r in p.runs:
                r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            i += 1
            continue

        # 구분선
        if re.fullmatch(r"-{3,}", stripped):
            doc.add_paragraph()
            i += 1
            continue

        # 목록
        m = re.match(r"^(\s*)([-*•]|\d+[.)])\s+(.*)$", line)
        if m:
            indent = len(m.group(1))
            numbered = m.group(2)[0].isdigit()
            if numbered and indent >= 2:
                # 하위 번호는 Word 자동 번호가 이어지므로, 표기 그대로 문단으로 둔다.
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Pt(18 + indent * 3)
                p.paragraph_format.first_line_indent = Pt(-12)
                _inline_runs(p, f"{m.group(2)} {m.group(3)}")
            else:
                style_name = "List Bullet" if not numbered else "List Number"
                try:
                    p = doc.add_paragraph(style=style_name)
                except KeyError:
                    p = doc.add_paragraph()
                if indent >= 2:
                    p.paragraph_format.left_indent = Pt(18 + indent * 3)
                _inline_runs(p, m.group(3))
            i += 1
            continue

        p = doc.add_paragraph()
        _inline_runs(p, stripped)
        i += 1

    # 꼬리말: 생성 시각
    doc.add_paragraph()
    tail = doc.add_paragraph()
    _set_korean_font(tail.add_run(f"생성: {datetime.now():%Y-%m-%d %H:%M} · 경영사무 AI 인수인계 도우미"),
                     size=8, color=(0x88, 0x88, 0x88))

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def safe_filename(title: str, ext: str) -> str:
    base = re.sub(r"[\\/:*?\"<>|]", "", title).strip() or "인수인계서"
    base = re.sub(r"\s+", "_", base)
    return f"{base[:40]}.{ext}"
