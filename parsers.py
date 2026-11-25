"""
parsers.py — 인수인계 도우미 자료 추출 계층
=============================================
업로드된 파일(단일 파일 / ZIP 묶음)에서 텍스트를 뽑아내고, 파일 성격을 자동 분류하고,
민감정보를 마스킹한다.

지원 형식
    .hwp   한글 5.0(바이너리, OLE 복합문서) — 순수 파이썬 파서(olefile + zlib)
    .hwpx  한글 5.0(개방형, ZIP+XML)
    .pdf .docx .xlsx .xls .csv .txt .md
    .zip   여러 파일 묶음(하위 폴더·중첩 ZIP 자동 해제)

HWP 파서는 외부 프로그램 없이 동작한다. (rhwp 등 별도 설치 불필요)
"""

from __future__ import annotations

import csv
import io
import os
import re
import struct
import unicodedata
import zipfile
import zlib
from dataclasses import dataclass, field, asdict
from typing import Iterable

# ----------------------------------------------------------------------------
# 상수
# ----------------------------------------------------------------------------

SUPPORTED_EXTS = {
    ".hwp", ".hwpx", ".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt", ".md",
}
MAX_FILE_BYTES = 25 * 1024 * 1024      # 파일 1개당 25MB
MAX_ZIP_DEPTH = 3                       # 중첩 ZIP 허용 깊이
MAX_PREVIEW_CHARS = 200_000             # 미리보기/분석에 쓰는 최대 글자 수

# HWP 5.0 레코드 태그
_TAG_PARA_TEXT = 67        # HWPTAG_PARA_TEXT = HWPTAG_BEGIN(0x10=16) + 51
_TAG_PARA_HEADER = 66


# ----------------------------------------------------------------------------
# 자료 구조
# ----------------------------------------------------------------------------

@dataclass
class ExtractedFile:
    name: str                      # 사용자에게 보이는 파일명
    ext: str
    size: int
    category: str = "미분류"        # 양식 / 완료 문서 / 규정·메모
    category_reason: str = ""
    text: str = ""
    origin: str = ""               # ZIP 안 경로 등
    pages: int | None = None
    warnings: list[str] = field(default_factory=list)
    masked_count: int = 0

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def short_name(self) -> str:
        """표에 넣기 좋게 줄인 이름. (TalkFile_ 접두어·해시 접미어 제거)"""
        return short_name(self.name)

    @property
    def ok(self) -> bool:
        return self.char_count > 0

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("text", None)
        d["char_count"] = self.char_count
        return d


# ----------------------------------------------------------------------------
# 민감정보 마스킹
# ----------------------------------------------------------------------------

# 순서가 중요하다: 카드번호 → 주민번호 → 사업자번호 → 휴대폰 → 계좌번호
_MASK_RULES: list[tuple[str, re.Pattern, callable]] = []


def _rule(pattern: str, label: str):
    def deco(fn):
        _MASK_RULES.append((label, re.compile(pattern), fn))
        return fn
    return deco


def _has_context(m: re.Match, words: tuple[str, ...]) -> bool:
    """앞 10자 안에 '계좌' 같은 말이 이미 있는지."""
    before = m.string[max(0, m.start() - 10):m.start()]
    return any(w in before for w in words)


def _keep_tail(m: re.Match, keep: int, label: str, words: tuple[str, ...] = ()) -> str:
    """마지막 keep 글자만 남기고 가운데를 *로 덮는다."""
    raw = m.group(0)
    digits = re.sub(r"\D", "", raw)
    tail = digits[-keep:] if keep else ""
    masked = f"{'*' * max(4, len(digits) - keep)}{tail}"
    if words and _has_context(m, words):
        return masked
    return f"{label} {masked}"


@_rule(r"(?<!\d)(\d{4})[- ]?(\d{4})[- ]?(\d{4})[- ]?(\d{4})(?!\d)", "카드번호")
def _mask_card(m, label):
    return _keep_tail(m, 4, label, ("카드", "card"))


@_rule(r"(?<!\d)(\d{6})[- ]([1-4]\d{6})(?!\d)", "주민번호")
def _mask_rrn(m, label):
    return f"{m.group(1)}-{m.group(2)[0]}{'*' * 6}"


@_rule(r"(?<!\d)(\d{3})[- ](\d{2})[- ](\d{5})(?!\d)", "사업자번호")
def _mask_biz(m, label):
    body = f"{m.group(1)}-**-{m.group(3)[:1]}****"
    return body if _has_context(m, ("사업자", "등록번호")) else f"{label} {body}"


@_rule(r"(?<!\d)(01[016789])[- .]?(\d{3,4})[- .]?(\d{4})(?!\d)", "휴대폰")
def _mask_mobile(m, label):
    return f"{m.group(1)}-****-{m.group(3)}"


@_rule(r"(?<!\d)(\d{2,6})[- ](\d{2,6})[- ](\d{2,8})(?!\d)", "계좌번호")
def _mask_account(m, label):
    digits = re.sub(r"\D", "", m.group(0))
    # 날짜(2026-11-21)나 짧은 번호는 건드리지 않는다.
    if len(digits) < 10:
        return m.group(0)
    body = f"{'*' * (len(digits) - 4)}{digits[-4:]}"
    return body if _has_context(m, ("계좌", "account")) else f"{label} {body}"


def mask_sensitive(text: str) -> tuple[str, int]:
    """주민등록번호·계좌번호·휴대폰·카드·사업자번호를 마스킹한다.

    반환: (마스킹된 텍스트, 마스킹 건수)
    """
    count = 0
    for label, pattern, fn in _MASK_RULES:
        def _sub(m, label=label, fn=fn):
            nonlocal count
            out = fn(m, label)
            if out != m.group(0):          # 실제로 가려진 것만 센다
                count += 1
            return out
        text = pattern.sub(_sub, text)
    return text, count


# ----------------------------------------------------------------------------
# 1) HWP 5.0 (바이너리) — 순수 파이썬
# ----------------------------------------------------------------------------

def _iter_records(data: bytes):
    """HWP 5.0 레코드 스트림을 (tag, level, body)로 순회한다."""
    i, n = 0, len(data)
    while i + 4 <= n:
        header = struct.unpack_from("<I", data, i)[0]
        i += 4
        tag = header & 0x3FF
        level = (header >> 10) & 0x3FF
        size = (header >> 20) & 0xFFF
        if size == 0xFFF:                       # 크기 확장 필드
            if i + 4 > n:
                break
            size = struct.unpack_from("<I", data, i)[0]
            i += 4
        body = data[i:i + size]
        if len(body) < size:
            break
        i += size
        yield tag, level, body


def _decode_para_text(body: bytes) -> str:
    """HWPTAG_PARA_TEXT 본문 → 문자열.

    2바이트 단위 UTF-16LE. 32 미만은 제어 문자로,
    탭(9)/줄바꿈(10)/문단끝(13)은 문자로 바꾸고 나머지 확장 컨트롤은
    8워드(14바이트)를 건너뛴다. (건너뛰는 길이는 실제 한글 문서로 검증함)
    """
    out: list[str] = []
    i, n = 0, len(body)
    while i + 1 < n:
        code = body[i] | (body[i + 1] << 8)
        i += 2
        if code >= 32:
            out.append(chr(code))
        elif code == 0:
            continue
        elif code in (10, 13):
            out.append("\n")
        elif code == 9:
            out.append("\t")
        else:
            i += 14                            # 확장 컨트롤 7워드
    return "".join(out)


def _decompress_stream(raw: bytes) -> bytes:
    """HWP 스트림 해제. 압축/비압축 모두 대응."""
    if not raw:
        return raw
    for wbits in (-15, 15, 47):
        try:
            return zlib.decompress(raw, wbits)
        except Exception:
            continue
    return raw


def _hwp_preview_text(ole) -> str:
    """PrvText 스트림(미리보기 텍스트)에서 앞부분만 얻는다."""
    try:
        raw = ole.openstream("PrvText").read()
        return raw.decode("utf-16-le", errors="ignore").replace("\r", "\n").strip()
    except Exception:
        return ""


def parse_hwp(data: bytes) -> tuple[str, dict]:
    """HWP 5.0 바이너리 → (본문 텍스트, 부가정보)."""
    try:
        import olefile
    except ImportError:                          # pragma: no cover
        raise RuntimeError("olefile 패키지가 필요합니다. (pip install olefile)")

    info: dict = {"parser": "hwp5-python", "sections": 0}
    ole = olefile.OleFileIO(io.BytesIO(data))
    try:
        if not ole.exists("FileHeader"):
            raise ValueError("한글 문서 형식이 아닙니다(FileHeader 없음).")
        header = ole.openstream("FileHeader").read(40)
        signature = header[:32].split(b"\x00")[0]
        if not signature.startswith(b"HWP Document File"):
            raise ValueError("HWP 5.0 형식이 아닙니다(서명 불일치).")
        version = ".".join(str(b) for b in reversed(header[32:36]))
        flags = struct.unpack_from("<I", header, 36)[0] if len(header) >= 40 else 0
        if flags & 0x02:
            raise ValueError("암호가 걸린 한글 문서는 열 수 없습니다.")
        info["version"] = version

        sections = sorted(e for e in ole.listdir() if e[0] == "BodyText")
        chunks: list[str] = []
        for entry in sections:
            raw = ole.openstream(entry).read()
            body_stream = _decompress_stream(raw)
            got = False
            for tag, _level, body in _iter_records(body_stream):
                if tag == _TAG_PARA_TEXT:
                    chunks.append(_decode_para_text(body))
                    got = True
            if got:
                info["sections"] += 1
        text = "\n".join(chunks)

        if len(text.strip()) < 20:
            preview = _hwp_preview_text(ole)
            if preview.strip():
                info["parser"] = "hwp5-pvtext"
                info["warning"] = "본문 스트림이 비어 있어 미리보기 텍스트(PrvText)로 대체했습니다."
                text = preview
        return text, info
    finally:
        try:
            ole.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# 2) HWPX (ZIP + XML)
# ----------------------------------------------------------------------------

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _hwpx_walk(elem, out: list[str]) -> None:
    for child in elem:
        name = _local(child.tag)
        if name == "p":
            _hwpx_walk(child, out)
            out.append("\n")
        elif name == "t":
            out.append(child.text or "")
        else:
            _hwpx_walk(child, out)


def parse_hwpx(data: bytes) -> tuple[str, dict]:
    import xml.etree.ElementTree as ET

    info: dict = {"parser": "hwpx-xml", "sections": 0}
    out: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist()
                 if re.match(r"Contents/section\d+\.xml$", n)]
        for name in sorted(names):
            try:
                root = ET.fromstring(zf.read(name))
                _hwpx_walk(root, out)
                info["sections"] += 1
            except Exception as exc:
                info.setdefault("warnings", []).append(f"{name}: {exc}")
    return "".join(out), info


# ----------------------------------------------------------------------------
# 3) PDF / DOCX / XLSX / XLS / CSV / TXT
# ----------------------------------------------------------------------------

def parse_pdf(data: bytes) -> tuple[str, dict]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for p in reader.pages[:120]:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n\n".join(pages)
    info = {"parser": "pypdf", "pages": len(reader.pages)}
    if len(text.strip()) < 20:
        info["warning"] = "PDF에서 글자가 거의 나오지 않았습니다. 스캔 이미지 PDF라면 텍스트 추출이 되지 않습니다."
    return text, info


def parse_docx(data: bytes) -> tuple[str, dict]:
    from docx import Document

    doc = Document(io.BytesIO(data))
    lines: list[str] = [p.text for p in doc.paragraphs]
    for t_index, table in enumerate(doc.tables, 1):
        lines.append(f"\n[표 {t_index}]")
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                lines.append(" | ".join(cells))
    return "\n".join(lines), {"parser": "python-docx", "tables": len(doc.tables)}


def parse_xlsx(data: bytes) -> tuple[str, dict]:
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    lines: list[str] = []
    for ws in wb.worksheets:
        lines.append(f"\n[시트: {ws.title}]")
        for r_index, row in enumerate(ws.iter_rows(values_only=True)):
            if r_index >= 3000:
                lines.append("… (행 3000개 초과, 이하 생략)")
                break
            cells = ["" if v is None else str(v) for v in row]
            if any(c.strip() for c in cells):
                lines.append(" | ".join(cells).rstrip(" |"))
    info = {"parser": "openpyxl", "sheets": [ws.title for ws in wb.worksheets]}
    wb.close()
    return "\n".join(lines), info


def parse_xls(data: bytes) -> tuple[str, dict]:
    import xlrd

    book = xlrd.open_workbook(file_contents=data)
    lines: list[str] = []
    for sheet in book.sheets():
        lines.append(f"\n[시트: {sheet.name}]")
        for r in range(min(sheet.nrows, 3000)):
            cells = [str(sheet.cell_value(r, c)) for c in range(sheet.ncols)]
            if any(c.strip() for c in cells):
                lines.append(" | ".join(cells).rstrip(" |"))
    return "\n".join(lines), {"parser": "xlrd", "sheets": book.sheet_names()}


def _decode_text(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_csv(data: bytes) -> tuple[str, dict]:
    text = _decode_text(data)
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except Exception:
        return text, {"parser": "csv"}
    lines = [" | ".join(c.strip() for c in r).rstrip(" |") for r in rows if any(c.strip() for c in r)]
    return "\n".join(lines), {"parser": "csv", "rows": len(rows)}


def parse_plain(data: bytes) -> tuple[str, dict]:
    return _decode_text(data), {"parser": "text"}


_PARSERS = {
    ".hwp": parse_hwp,
    ".hwpx": parse_hwpx,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".xlsx": parse_xlsx,
    ".xls": parse_xls,
    ".csv": parse_csv,
    ".txt": parse_plain,
    ".md": parse_plain,
}


# ----------------------------------------------------------------------------
# 4) 자동 분류
# ----------------------------------------------------------------------------

_CATEGORY_RULES = [
    ("양식", ["양식", "서식", "템플릿", "template", "form", "blank", "빈칸", "작성예시"]),
    ("규정·메모", ["규정", "지침", "매뉴얼", "manual", "가이드", "guide", "정책",
                 "규칙", "메모", "회의록", "공지", "안내문", "체크리스트"]),
    ("완료 문서", ["완료", "결과", "보고", "최종", "정산", "실적", "현황", "보고서",
                 "수행계획", "계획서", "신청서", "제출본"]),
]

_PLACEHOLDER_HINTS = ["추후 기재", "작성 예정", "○○", "ㅇㅇ", "기입", "빈칸", "해당 없음"]


def short_name(name: str, limit: int = 42) -> str:
    """'TalkFile_…-0ca1623a….hwp' 같은 업로드 이름을 사람이 읽기 좋게 줄인다."""
    stem, ext = os.path.splitext(os.path.basename(name))
    stem = unicodedata.normalize("NFC", stem)
    stem = re.sub(r"^TalkFile_", "", stem)
    stem = re.sub(r"[-_][0-9a-f]{16,}$", "", stem, flags=re.I)
    stem = re.sub(r"\s*\(\d\)$", "", stem).strip()
    if len(stem) > limit:
        stem = stem[: limit - 1] + "…"
    return f"{stem}{ext.lower()}"


def classify(name: str, text: str) -> tuple[str, str]:
    """파일명 우선, 없으면 내용 패턴으로 성격을 정한다."""
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    for category, keys in _CATEGORY_RULES:
        for key in keys:
            if key in stem:
                return category, f"파일명에 '{key}' 포함"
    sample = text[:4000]
    hits = [h for h in _PLACEHOLDER_HINTS if h in sample]
    if hits:
        return "양식", f"본문에 빈칸 표시('{hits[0]}')가 있어 서식으로 판단"
    for category, keys in _CATEGORY_RULES:
        for key in keys:
            if key in sample:
                return category, f"본문에 '{key}' 언급"
    if len(text) > 300:
        return "완료 문서", "본문이 채워져 있어 작성 완료 문서로 판단"
    return "규정·메모", "짧은 참고 자료로 판단"


# ----------------------------------------------------------------------------
# 5) 진입점
# ----------------------------------------------------------------------------

def _safe_read(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    if info.file_size > MAX_FILE_BYTES:
        raise ValueError(f"파일이 너무 큽니다({info.file_size:,} bytes)")
    with zf.open(info) as fp:
        return fp.read(MAX_FILE_BYTES)


def _make_record(name: str, ext: str, data: bytes, origin: str = "",
                 apply_mask: bool = True) -> ExtractedFile:
    name = unicodedata.normalize("NFC", name)
    rec = ExtractedFile(name=name, ext=ext, size=len(data), origin=origin)
    parser = _PARSERS.get(ext)
    if parser is None:
        rec.warnings.append("지원하지 않는 형식입니다.")
        return rec
    try:
        text, info = parser(data)
    except Exception as exc:
        rec.warnings.append(f"읽기 실패: {type(exc).__name__}: {exc}")
        return rec

    if info.get("warning"):
        rec.warnings.append(info["warning"])
    rec.warnings.extend(info.get("warnings", []) or [])
    if info.get("pages"):
        rec.pages = info["pages"]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if apply_mask:
        text, masked = mask_sensitive(text)
        rec.masked_count = masked
    rec.text = text[:MAX_PREVIEW_CHARS]
    rec.category, rec.category_reason = classify(name, rec.text)
    return rec


def extract_one(name: str, data: bytes, apply_mask: bool = True,
                origin: str = "", depth: int = 0) -> list[ExtractedFile]:
    """파일 1개(또는 ZIP)를 받아 ExtractedFile 목록으로 만든다."""
    ext = os.path.splitext(name)[1].lower()
    if ext == ".zip":
        return _extract_zip(name, data, apply_mask, depth)
    if len(data) > MAX_FILE_BYTES:
        rec = ExtractedFile(name=name, ext=ext, size=len(data), origin=origin)
        rec.warnings.append(f"파일이 너무 큽니다({len(data):,} bytes, 최대 25MB)")
        return [rec]
    return [_make_record(name, ext, data, origin, apply_mask)]


def _extract_zip(name: str, data: bytes, apply_mask: bool, depth: int) -> list[ExtractedFile]:
    results: list[ExtractedFile] = []
    if depth >= MAX_ZIP_DEPTH:
        rec = ExtractedFile(name=name, ext=".zip", size=len(data))
        rec.warnings.append("ZIP 중첩이 너무 깊습니다.")
        return [rec]
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except Exception as exc:
        rec = ExtractedFile(name=name, ext=".zip", size=len(data))
        rec.warnings.append(f"ZIP을 열 수 없습니다: {exc} (암호가 걸린 압축 파일일 수 있습니다)")
        return [rec]

    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            inner = info.filename.replace("\\", "/")
            base = os.path.basename(inner)
            if not base or base.startswith(".") or "__MACOSX" in inner:
                continue
            ext = os.path.splitext(base)[1].lower()
            if ext == ".zip":
                try:
                    results.extend(_extract_zip(base, _safe_read(zf, info), apply_mask, depth + 1))
                except Exception as exc:
                    rec = ExtractedFile(name=base, ext=".zip", size=info.file_size, origin=inner)
                    rec.warnings.append(f"중첩 ZIP 읽기 실패: {exc}")
                    results.append(rec)
                continue
            if ext not in SUPPORTED_EXTS:
                rec = ExtractedFile(name=base, ext=ext or "(확장자 없음)",
                                    size=info.file_size, origin=f"{name}/{inner}")
                rec.warnings.append("지원하지 않는 형식이라 건너뛰었습니다.")
                results.append(rec)
                continue
            try:
                results.append(_make_record(base, ext, _safe_read(zf, info),
                                            origin=f"{name}/{inner}", apply_mask=apply_mask))
            except Exception as exc:
                rec = ExtractedFile(name=base, ext=ext, size=info.file_size,
                                    origin=f"{name}/{inner}")
                rec.warnings.append(f"읽기 실패: {exc}")
                results.append(rec)
    return results


def extract_uploads(files: Iterable[tuple[str, bytes]],
                    apply_mask: bool = True) -> list[ExtractedFile]:
    """(파일명, 바이트) 목록 → ExtractedFile 목록."""
    out: list[ExtractedFile] = []
    for name, data in files:
        out.extend(extract_one(name, data, apply_mask=apply_mask))
    return out


def summary_stats(files: list[ExtractedFile]) -> dict:
    ok = [f for f in files if f.ok]
    return {
        "files": len(files),
        "readable": len(ok),
        "chars": sum(f.char_count for f in files),
        "masked": sum(f.masked_count for f in files),
        "by_category": {
            c: sum(1 for f in files if f.category == c)
            for c in ("양식", "완료 문서", "규정·메모", "미분류")
        },
    }
