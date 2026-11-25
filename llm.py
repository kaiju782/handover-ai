"""
llm.py — 인수인계서 생성 계층
=============================
두 가지 실행 모드를 같은 인터페이스로 제공한다.

1) 실제 API 모드 : OpenAI 호환 규격(base_url + api_key + model)으로 LLM을 호출한다.
                   K-water GPT도 OpenAI 호환이면 base_url만 바꿔 그대로 쓸 수 있다.
2) 데모(시뮬레이션) 모드 : API 키 없이 동작한다. 업로드한 자료에서 뽑은 실제 문장·수치·
                   표를 근거로 초안/질문/완성본을 구성한다. (하드코딩 문구가 아니라
                   문서 내용이 결과에 반영된다)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

try:                                  # 실제 API 모드에서만 쓰인다.
    from openai import OpenAI
except Exception:                     # pragma: no cover
    OpenAI = None


class LLMError(RuntimeError):
    """실제 API 호출 실패."""


# ----------------------------------------------------------------------------
# 실제 API 호출
# ----------------------------------------------------------------------------

def call_llm(*, base_url: str, api_key: str, model: str,
             messages: list[dict], temperature: float = 0.2,
             timeout: int = 120) -> str:
    """OpenAI 호환 Chat Completions 호출. 실패하면 LLMError로 올린다."""
    if not api_key:
        raise LLMError("API 키가 비어 있습니다. 사이드바에서 입력하거나 데모 모드로 바꿔 주세요.")
    if not base_url:
        raise LLMError("base_url이 비어 있습니다. 예: https://api.openai.com/v1")
    base_url = base_url.rstrip("/")
    if not model:
        raise LLMError("모델명이 비어 있습니다.")

    if OpenAI is None:
        raise LLMError("openai 패키지가 없습니다. `pip install openai` 후 다시 실행해 주세요.")

    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=1)
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise LLMError("모델이 빈 응답을 돌려주었습니다.")
        return content
    except LLMError:
        raise
    except Exception as exc:                                  # pragma: no cover
        raise LLMError(f"{type(exc).__name__}: {exc}") from exc


# ----------------------------------------------------------------------------
# 프롬프트
# ----------------------------------------------------------------------------

SYSTEM_ANALYST = (
    "당신은 기업 경영사무·사무행정 업무의 인수인계 문서를 전문으로 쓰는 실무자입니다. "
    "인수인계는 '읽는 사람이 첫날 바로 일할 수 있게' 쓰는 것이 목적입니다. "
    "자료에 없는 내용은 절대 지어내지 말고, 비어 있다고 표시한 뒤 인계자에게 물을 질문으로 바꿉니다. "
    "모든 답변은 한국어로, 실무 문서체(-합니다/-입니다)로 씁니다."
)

SYSTEM_FINAL = (
    "당신은 인수인계서 최종본을 완성하는 실무자입니다. "
    "1차 초안과 인계자 답변을 합쳐, 4개 파트(① 업무 절차, ② 숨은 예외 FAQ, ③ 월간/연간 마감표, "
    "④ 서식·참고파일 맵)가 모두 채워진 한국어 Markdown 문서를 작성합니다. "
    "답변이 없는 항목은 '미확인'으로 남기고, 자료에 없는 사실을 새로 만들지 않습니다."
)

SYSTEM_REVIEWER = (
    "당신은 후임자 입장에서 인계자 답변을 검수하는 실무자입니다. "
    "답변만 읽고 후임자가 첫날 실제로 실행할 수 있는지 판단합니다. "
    "칭찬이나 총평 위주의 답변은 쓰지 않고, 부족한 지점과 다시 물을 질문을 구체적으로 씁니다. "
    "모든 출력은 한국어로 씁니다."
)

# 질문·검수에서 함께 쓰는 분류
QUESTION_CATEGORIES = ("승인·결재", "예외·규정", "일정·마감", "서식·작성법", "연락·창구", "예산·증빙")


def _corpus_for_prompt(files, limit: int = 14000) -> str:
    """프롬프트에 넣을 자료 본문. 파일별로 잘라 전체 길이를 제한한다."""
    blocks, used = [], 0
    for f in files:
        if not f.ok:
            continue
        head = f.text[:4000]
        block = f"### 파일: {f.name} (성격: {f.category}, {f.char_count}자)\n{head}\n"
        if used + len(block) > limit:
            block = block[: max(0, limit - used)]
        blocks.append(block)
        used += len(block)
        if used >= limit:
            break
    return "\n".join(blocks)


def prompt_analysis(files) -> list[dict]:
    corpus = _corpus_for_prompt(files)
    forms = [f for f in files if f.ok and f.category == "양식"]
    docs = [f for f in files if f.ok and f.category == "완료 문서"]
    pair_note = ""
    if forms and docs:
        pair_note = (
            f"\n이번 자료에는 빈 서식 '{forms[0].short_name}'과(와) 완료 문서 "
            f"'{docs[0].short_name}'이(가) 함께 있습니다. 두 문서를 1:1로 맞대어 대조하세요.\n"
        )
    categories = ", ".join(QUESTION_CATEGORIES)
    user = f"""아래 자료를 읽고 '인수인계서 1차 초안'과 '인계자에게 물을 질문'을 만드세요.
{pair_note}
[자료]
{corpus}

[요구사항]
1. draft_markdown: 아래 순서의 Markdown.
   0) `## 0. 빈 서식 vs 완료 문서 대조 요약` — 반드시 첫 섹션으로 둡니다.
      - 빈 서식의 어떤 항목·표 칸이 완료 문서에서 어떤 기준·형식으로 채워졌는지 항목별 표
        (| 양식 항목 | 양식에 적힌 값 | 완료 문서에서 채운 내용 | 상태 |)
      - 서식엔 없지만 완료 문서에서 추가로 챙긴 내용(예외·첨부·출처·검증 지표 등)
      - 양식에는 있는데 완료 문서에도 빠졌거나 값이 어긋나는 지점(숫자·기한·규정 불일치)
      - 완료 문서가 함께 올라오지 않았다면, 양식 안의 빈칸·미기재 항목만 짚어 정리합니다.
   - 자료 요약 : 문서 수, 핵심 키워드, 확인된 기본 정보 표.
   - ① 업무 절차 : 자료에 나온 실제 업무 흐름을 순서대로, 단계마다 근거 자료 표시.
   - ② 숨은 예외 FAQ : 자료에 없어서 후임자가 막힐 상황을 Q/A 형태로. A는 '인계자 확인 필요'로 비워 둠.
   - ③ 월간/연간 마감표 : 자료의 기간·일정에서 뽑은 마감 항목 표(| 구분 | 시기 | 내용 | 근거 |).
   - ④ 서식·참고파일 맵 : 업로드된 파일별 용도 표(| 파일 | 성격 | 쓰는 곳 |).
2. questions: 개수를 5개로 고정하지 마세요. 자료 분량과 누락·모호함의 정도에 따라 필요한 만큼
   (보통 5~15개) 도출합니다. 자료가 방대하고 빈칸이 많으면 15개를 넘겨도 됩니다.
   각 항목은 {{"id": 1부터 증가, "question": "...", "why": "후임자가 왜 막히는지",
   "evidence": "근거 자료의 실제 문장 인용", "priority": "필수" 또는 "권장",
   "category": "{categories}" 중 하나}}.
   - `필수`: 답이 없으면 후임자가 첫 업무를 시작할 수 없거나 되돌릴 수 없는 실수가 나는 항목
     (연락 창구, 마감일, 예산 집행·증빙, 결재권자, 서로 어긋나는 값).
   - `권장`: 있으면 좋은 보강 항목(진행 상태, 개인 메신저에 남은 결정, 예외 경험).
   - 뻔한 일반론 금지. 자료의 실제 빈칸·불일치를 찌르는 질문만 냅니다.
3. 출력은 아래 JSON 하나만. 코드블록/설명 문장 없이.
{{"draft_markdown": "...", "questions": [...]}}"""
    return [{"role": "system", "content": SYSTEM_ANALYST},
            {"role": "user", "content": user}]


def prompt_review_answers(files, questions: list[dict], answers: dict,
                          extra_notes: str = "") -> list[dict]:
    """인계자 답변 검수 프롬프트."""
    corpus = _corpus_for_prompt(files, limit=9000)
    lines = []
    for q in questions:
        ans = (answers.get(q.get("id")) or "").strip() or "(빈 답변)"
        lines.append(
            f"Q{q.get('id')} [{q.get('priority', '권장')}·{q.get('category', '')}] {q.get('question')}"
            f"\n   (질문 배경: {q.get('why', '')})\n   A. {ans}"
        )
    categories = ", ".join(QUESTION_CATEGORIES)
    user = f"""아래 인계자 답변이 후임자에게 충분한지 검수하세요.

[업로드된 자료]
{corpus}

[검수할 질문과 답변]
{chr(10).join(lines)}

[기타 인계 메모]
{extra_notes.strip() or '(없음)'}

[검수 기준 — 하나라도 걸리면 '보완 필요']
1. 너무 짧거나 모호한 답변: "알아서", "문의", "적당히", "팀장님", "그때그때 다름", "확인 후",
   "기존대로"처럼 후임자가 그대로 실행할 수 없는 표현.
2. 자료와 어긋나는 답변: 위 자료의 숫자·기한·규정과 답변의 숫자·기한이 다르거나 모순되는 경우.
3. 누가/언제/어떻게/어디서 중 빠진 기준이 있어 후임자가 실행할 수 없는 답변.

[출력 형식]
아래 JSON 하나만. 코드블록/설명 문장 없이.
{{"reviews": [{{"id": 1, "verdict": "충분함" 또는 "보완 필요",
  "comment": "무엇이 부족한지 1~2문장", "requestion": "인계자가 바로 고쳐 쓸 수 있는 재질문"}}],
 "summary": "전체 총평 한 줄"}}
- 질문 {len(questions)}개 전부에 대해 하나씩 빠짐없이 답합니다.
- verdict가 "충분함"이면 comment는 "누가·언제·어떻게가 모두 확인됩니다"처럼 근거를 한 줄로 적습니다.
- 카테고리({categories})별로 반복되는 누락이 있으면 summary에 적습니다."""
    return [{"role": "system", "content": SYSTEM_REVIEWER},
            {"role": "user", "content": user}]


def prompt_final(files, draft_md: str, questions: list[dict], answers: dict,
                 extra_notes: str = "") -> list[dict]:
    corpus = _corpus_for_prompt(files, limit=8000)
    qa_lines = []
    for q in questions:
        ans = (answers.get(q.get("id")) or "").strip() or "(답변 없음)"
        qa_lines.append(f"Q{q.get('id')}. {q.get('question')}\nA. {ans}")
    user = f"""1차 초안과 인계자 답변을 합쳐 최종 인수인계서를 완성하세요.

[자료 요약]
{corpus}

[1차 초안]
{draft_md[:6000]}

[인계자 Q&A]
{chr(10).join(qa_lines)}

[기타 인계 메모]
{extra_notes.strip() or '(없음)'}

[요구사항]
- 제목: '업무 인수인계서 — <프로젝트/업무명>'
- ① 업무 절차 / ② 숨은 예외 FAQ / ③ 월간·연간 마감표 / ④ 서식·참고파일 맵 4개 파트를 모두 채웁니다.
- 인계자 답변은 해당 위치에 자연스럽게 반영하고, 답변이 없던 항목은 '미확인 — 인계자 확인 필요'로 남깁니다.
- 기타 인계 메모가 있으면 '## 기타 인계 메모' 섹션으로 그대로 옮깁니다.
- 표는 Markdown 표로. 마지막에 '인계자 확인 필요 목록'을 정리합니다.
- 문서 전체를 그대로 출력하세요. 인사말이나 설명은 쓰지 마세요."""
    return [{"role": "system", "content": SYSTEM_FINAL},
            {"role": "user", "content": user}]


def _parse_json_block(text: str) -> dict:
    """모델 응답에서 JSON 객체를 최대한 안전하게 뽑아낸다."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            return json.loads(t[start:end + 1])
        raise LLMError("모델 응답을 JSON으로 해석하지 못했습니다. 다시 시도해 주세요.")


# ----------------------------------------------------------------------------
# 데모(시뮬레이션) 모드 — 자료 기반 생성
# ----------------------------------------------------------------------------

_STOPWORDS = {
    "있습니다", "합니다", "하는", "위한", "대한", "통해", "경우", "때문", "그리고", "또한",
    "있도록", "수행", "업무", "자료", "내용", "관련", "필요", "방안", "제안", "이를", "이는",
    "따라", "각각", "다양한", "주요", "해당", "위해", "통한", "에서", "대해", "대하여",
    "프로젝트", "시간", "활용", "담당자", "주차", "진행", "보고서", "방식", "작성", "문서",
    "결과", "정보", "사항", "처리", "기업", "개인별", "체계적", "효율적", "효과",
}

_ID_KEYS = ("프로젝트명", "과제명", "사업명", "업무명", "문서명")
_PERIOD_KEYS = ("실행기간", "수행기간", "사업기간", "계약기간", "프로젝트기간", "근무기간")
_TEAM_KEYS = ("팀명", "부서명", "담당부서", "소속")
_MENTOR_KEYS = ("멘토링계획", "멘토", "슈퍼바이저")
_LABELISH = re.compile(r"^[가-힣A-Za-z()·/\s]{1,12}$")


def _clean(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" \t·-•■ㅇ")


def _first_after(lines: list[str], keys: tuple[str, ...], lookahead: int = 3) -> str | None:
    """'프로젝트명' 같은 라벨 다음 줄의 실제 값을 찾는다. 라벨은 짧아야 한다."""
    norm = {k.replace(" ", "") for k in keys}
    for i, line in enumerate(lines):
        lab = line.replace(" ", "")
        if lab in norm or any(lab == k.replace(" ", "") for k in keys):
            for j in range(i + 1, min(i + 1 + lookahead, len(lines))):
                cand = lines[j]
                if not cand or len(cand) > 100:
                    continue
                if cand.replace(" ", "") in norm:
                    continue
                if cand.endswith(("다.", "요.", "습니다.")) and len(cand) > 40:
                    continue
                return cand
    return None


def _find_members(lines: list[str]) -> list[tuple[str, str, str]]:
    """팀장/팀원 형태의 역할-이름-설명 3연속 줄을 뽑는다."""
    members, seen = [], set()
    for i, line in enumerate(lines):
        if re.fullmatch(r"(팀장|팀원\s*\d?|부장|과장|대리|사원|담당)", line.replace(" ", "")) and i + 2 < len(lines):
            name, role = lines[i + 1], lines[i + 2]
            if not (1 <= len(name) <= 12 and 4 <= len(role) <= 80):
                continue
            if name in seen:
                continue
            seen.add(name)
            members.append((line.strip(), name, role))
    return members[:8]


def _find_bullets(lines: list[str]) -> list[str]:
    out = []
    for raw in lines:
        if raw.startswith(("ㅇ ", "ㅇ", "- ", "•", "■ ")) and len(raw) >= 20:
            out.append(_clean(raw.lstrip("ㅇ-•■ ")))
        if len(out) >= 60:
            break
    return out


def _find_budget(lines: list[str]) -> tuple[str | None, str | None]:
    """(예산 총액 표기, 총액 숫자) 를 찾는다."""
    total = None
    for i, line in enumerate(lines):
        if line.replace(" ", "") in ("총계", "총액", "합계"):
            for j in range(i + 1, min(i + 4, len(lines))):
                n = lines[j].replace(" ", "").rstrip("원")
                if re.fullmatch(r"[\d,]{3,}", n):
                    total = n if lines[j].strip().endswith("원") else f"{n}원"
                    break
    if total is None:
        for line in lines:
            m = re.fullmatch(r"([\d,]{4,})\s*원\s*[x×]\s*(\d+)\s*개월", line.replace(" ", ""))
            if m:
                total = f"{line.strip()} (월 {m.group(1)}원)"
                break
    amount = None
    if total:
        m = re.search(r"[\d,]{4,}", total)
        if m:
            amount = m.group(0)
    return total, amount


def _keywords(text: str, k: int = 8) -> list[str]:
    words = re.findall(r"[가-힣]{2,}", text)
    freq: dict[str, int] = {}
    for w in words:
        if w in _STOPWORDS or len(w) > 12:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))
    out: list[str] = []
    for w, _c in ranked:
        if any(w in o or o in w for o in out):
            continue
        out.append(w)
        if len(out) >= k:
            break
    return out


def _dates(text: str) -> list[str]:
    found = re.findall(r"20\d{2}[.\-/년]\s?\d{1,2}[.\-/월]\s?\d{1,2}일?", text)
    uniq = {re.sub(r"\s+", "", d) for d in found}
    return list(sorted(uniq))[:10]


@dataclass
class Facts:
    project: str | None
    period: str | None
    team: str | None
    members: list[tuple[str, str, str]]
    budget_total: str | None
    budget_amount: str | None
    mentor: str | None
    share: str | None
    bullets: list[str]
    keywords: list[str]
    dates: list[str]
    placeholders: int
    weeks: list[str]
    files: list

    @property
    def title(self) -> str:
        return self.project or "업무 인수인계"


def _lines_of(files) -> list[tuple[str, int, list[str]]]:
    """(파일명, 0, 정리된 줄 목록). 불릿 기호는 남겨 둔다(분류에 쓰임)."""
    out = []
    for f in files:
        if not f.ok:
            continue
        ls = [re.sub(r"\s+", " ", x).strip() for x in f.text.split("\n")]
        ls = [l for l in ls if l]
        out.append((f.name, 0, ls))
    return out


def _scan(files, needles: tuple[str, ...], window: int = 0,
          min_len: int = 1) -> list[tuple[str, str]]:
    """needles를 앞에서부터 우선순위대로 훑어 (파일명, 문장)을 모은다."""
    hits, seen = [], set()
    for needle in needles:
        if not needle:
            continue
        for fname, _i, lines in _lines_of(files):
            for idx, line in enumerate(lines):
                if needle not in line or len(line) < min_len:
                    continue
                text = line if not window else " ".join(lines[idx:idx + 1 + window])
                key = text[:30]
                if key in seen:
                    continue
                seen.add(key)
                hits.append((fname, text))
    return hits


def _ev(files, needles: tuple[str, ...], window: int = 0, width: int = 90,
        min_len: int = 1) -> tuple[str, str]:
    """근거 인용: (파일명, "문장") — 못 찾으면 빈 문자열."""
    hits = _scan(files, needles, window, min_len=min_len)
    if not hits:
        return "", ""
    fname, line = hits[0]
    line = _strip_marker(line.strip()).replace("**", "")
    if len(line) > width:
        line = line[: width - 1] + "…"
    return fname, f'"{line}"'


_ROUTINE_PREFIX = re.compile(r"^(커뮤니케이션|프로젝트\s?공유|공유\s?[:：]|멘토링|미팅|회의|보고\s?[:：])")
_ROUTINE_PERIOD = re.compile(r"(주\s?\d\s?회|매주|격주|2주|1주|월\s?\d\s?회|월\d회|매월|2주에 한)")


def _routine_lines(files, limit: int = 3) -> list[tuple[str, str]]:
    """정기 진행 방식(미팅·공유·보고) 줄만 골라낸다. 배경 설명 문장은 버린다."""
    out: list[tuple[str, str]] = []
    seen = set()
    for fname, line in _scan(files, ("커뮤니케이션", "프로젝트 공유", "미팅", "회의", "멘토링", "보고"),
                             min_len=12):
        core = _strip_marker(line)
        if not (_ROUTINE_PREFIX.match(core) or _ROUTINE_PERIOD.search(core)):
            continue
        if len(core) > 160:
            continue
        key = core[:24]
        if key in seen:
            continue
        seen.add(key)
        out.append((fname, core))
        if len(out) >= limit:
            break
    return out


def _share_line(files) -> str | None:
    for _fname, line in _scan(files, ("프로젝트 공유", "공유", "노션", "Notion", "드라이브"), min_len=8):
        if any(k in line for k in ("공유", "보관", "관리")):
            return re.sub(r"^.*?[:：]\s*", "", line).strip() or line
    return None


def _mentor_line(files) -> str | None:
    """'멘토링 계획 : 이하진 교수님, 2주에 한번씩 …' 같은 실제 보고 라인을 뽑는다."""
    best = None
    for _fname, line in _scan(files, ("멘토링", "멘토", "슈퍼바이저")):
        if len(line) < 8 or line.rstrip().endswith("계획"):
            continue
        if re.search(r"멘토.{0,12}[:：]|교수|주에 한|주 1회|주1회", line):
            return re.sub(r"^.*?[:：]\s*", "", line).strip()
        best = best or line
    return best


def _facts(files) -> Facts:
    lines = [l for _n, _i, ls in _lines_of(files) for l in ls]
    text = "\n".join(lines)
    total, amount = _find_budget(lines)
    mentor_line = _mentor_line(files)
    return Facts(
        project=_first_after(lines, _ID_KEYS),
        period=_first_after(lines, _PERIOD_KEYS),
        team=_first_after(lines, _TEAM_KEYS),
        members=_find_members(lines),
        budget_total=total, budget_amount=amount,
        mentor=mentor_line,
        share=_share_line(files),
        bullets=_find_bullets(lines),
        keywords=_keywords(text),
        dates=_dates(text),
        placeholders=sum(text.count(p) for p in ("추후 기재", "작성 예정", "○○")),
        weeks=list(dict.fromkeys(re.findall(r"\d+주차", text))),
        files=files,
    )


def _strip_marker(line: str) -> str:
    return re.sub(r"^[ㅇ■□•·\-–]\s+", "", line).strip()


def _shorten(md: str, files) -> str:
    """본문에 남은 업로드 원본 파일명을 짧은 이름으로 바꾼다."""
    for f in files:
        md = md.replace(f.name, f.short_name)
    return md


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return "\n".join(out)


def _period_months(period: str) -> list[str]:
    found = re.findall(r"(20\d{2})[.\-/년]\s?(\d{1,2})", period)
    if len(found) < 2:
        return []
    (y, m), (y2, m2) = (int(found[0][0]), int(found[0][1])), (int(found[1][0]), int(found[1][1]))
    out = []
    for _ in range(36):
        out.append(f"{y}년 {m}월")
        if (y, m) == (y2, m2):
            break
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


# ----------------------------------------------------------------------------
# 빈 서식 ↔ 완료 문서 1:1 대조
# ----------------------------------------------------------------------------

# 서식에서 자주 쓰는 항목 이름과 같은 뜻의 표현
_COMPARE_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("제안기업·기관명", ("제안기업", "기관명", "회사명", "업체명")),
    ("사업장 주소", ("사업장", "주소")),
    ("담당자명", ("담당자명", "담당자 성명", "성명")),
    ("직급·직위", ("직급", "직위")),
    ("담당자 연락처", ("담당자 연락처", "연락처", "전화번호")),
    ("이메일", ("이메일", "메일 주소")),
    ("프로젝트·업무명", ("프로젝트명", "과제명", "업무명", "사업명")),
    ("직무·업무 분야", ("직무 분야", "직무분야", "업무 분야")),
    ("프로젝트 소개", ("프로젝트 소개", "업무 소개")),
    ("추진 배경·필요성", ("제안 배경", "추진배경", "추진 배경", "필요성")),
    ("실행·수행 기간", ("실행기간", "수행기간", "사업기간", "계약기간")),
    ("팀·소속", ("팀명", "부서명", "소속")),
    ("역할 분담", ("역할 분담", "역할분담", "팀원 및", "주요 역할")),
    ("주요 기능", ("주요기능", "주요 기능")),
    ("필요 역량", ("필요역량", "필요 역량")),
    ("예상 결과물", ("예상 결과물", "결과물", "산출물")),
    ("기대효과", ("기대효과", "기대 효과")),
    ("활용 분야", ("활용 분야", "활용분야")),
    ("추진 일정", ("추진일정", "추진 일정", "일정표")),
    ("수행·진행 방법", ("수행방법", "수행 계획", "수행계획", "진행 방식", "커뮤니케이션")),
    ("멘토·보고 체계", ("멘토링", "멘토", "보고 체계")),
    ("예산 총액", ("예산 총액", "예산총액", "총 계", "총계", "총액")),
    ("세부 집행 계획", ("세부 활용", "세부활용", "지출항목", "집행 계획")),
    ("증빙·정산", ("증빙", "정산")),
    ("보안·개인정보", ("보안", "개인정보", "유의사항")),
)

_PLACEHOLDER_WORDS = ("추후 기재", "추후기재", "작성 예정", "작성예정", "○○", "ㅇㅇ",
                      "미정", "기입", "해당 없음", "해당없음", "빈칸")
_AMOUNT_TOKEN = re.compile(r"(\d[\d,]{2,})\s*(만\s?원|원)\s*(이상|이하|미만|초과)?")
_RANGE_RE = re.compile(r"20\d{2}[.\-/년]\s?\d{1,2}[.\-/월]?\s?\d{0,2}일?\s*[~∼\-–]\s*"
                       r"20\d{2}[.\-/년]\s?\d{1,2}[.\-/월]?\s?\d{0,2}일?")
_LABEL_NORMS = {re.sub(r"[^가-힣A-Za-z0-9]", "", s)
                for _l, _syns in _COMPARE_FIELDS for s in _syns}


def _norm_label(text: str) -> str:
    return re.sub(r"[^가-힣A-Za-z0-9]", "", text or "")


def _is_blank(value: str | None) -> bool:
    if value is None:
        return True
    v = (value or "").strip()
    if len(re.sub(r"\s", "", v)) < 2:
        return True
    return any(w in v for w in _PLACEHOLDER_WORDS)


def _amount_values(text: str) -> list[int]:
    """글 안의 금액. '100만 원 이상' 같은 기준선은 총액과 비교하지 않도록 뺀다."""
    out: list[int] = []
    for m in _AMOUNT_TOKEN.finditer(text or ""):
        if m.group(3):
            continue
        try:
            n = int(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if m.group(2).replace(" ", "").startswith("만"):
            n *= 10000
        if n >= 10000:
            out.append(n)
    return out


def _file_lines(f) -> list[str]:
    return [re.sub(r"\s+", " ", x).strip() for x in (f.text or "").split("\n")
            if x.strip()]


_TABLE_WORDS = {"구분", "항목", "설명", "내용", "비고", "이름", "역할", "금액", "수량",
                "단위", "날짜", "기간", "방법", "제목", "세부내역", "주요역할", "합계", "총계", "계"}


def _labelish(text: str) -> bool:
    """그 자체가 항목 이름인 짧은 줄(다른 칸의 라벨·표 머리글)인지."""
    core = _norm_label(text)
    if not core or len(core) > 14:
        return False
    if core in _TABLE_WORDS:
        return True
    if re.match(r"^\d+[.\-)]\s", text.strip()):
        return True
    return any(ns and ns in core for ns in _LABEL_NORMS)


def _plain(text: str | None, limit: int = 90) -> str:
    """표·질문에 넣기 좋게 Markdown 기호를 덜어 낸다."""
    t = re.sub(r"[*_`>#]", "", str(text or "")).replace("~", "∼").strip(" -•·[]")
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) <= limit else t[: limit - 1] + "…"


def _next_value(lines: list[str], i: int) -> str | None:
    """라벨 다음 칸의 값. 라벨이 연속으로 나오는 표 머리글은 건너뛰다."""
    for j in range(i + 1, min(i + 4, len(lines))):
        cand = lines[j]
        if not cand or _labelish(cand):
            continue
        return cand if len(cand) <= 120 else cand[:119] + "…"
    return None


def _lookup(lines: list[str], synonyms: tuple[str, ...], kind: str | None = None,
            strict: bool = False) -> str | None:
    """파일 안에서 항목 이름을 찾아 그 값을 돌려준다. 없으면 None.

    strict=True(서식 쪽)는 '항목 이름이 줄 앞쪽에 있는 경우'만 인정해
    표 머리글·다른 칸의 설명이 값으로 잡히지 않게 한다.
    """
    if not lines:
        return None
    if kind == "budget":
        total, _amount = _find_budget(lines)
        if total:
            return total
        for line in lines:
            if "예산" in line and len(line) >= 6:
                return line if len(line) <= 120 else line[:119] + "…"
        return None
    if kind == "period":
        m = _RANGE_RE.search("\n".join(lines))
        if m:
            return re.sub(r"\s+", " ", m.group(0))
    for i, line in enumerate(lines):
        core = _norm_label(line)
        for s in synonyms:
            ns = _norm_label(s)
            if not ns or ns not in core:
                continue
            parts = re.split(r"[:：]", line, maxsplit=1)
            if len(parts) == 2 and parts[1].strip() and "http" not in parts[1]:
                return parts[1].strip()[:120]
            if core.index(ns) > (3 if strict else 12):
                continue
            if len(core) <= len(ns) + (3 if strict else 10):
                val = _next_value(lines, i)
                if val:
                    return val
            elif len(line) <= 80 and not strict:
                return line[:120]
    return None


def _doc_headings(f) -> list[str]:
    out = []
    for line in (f.text or "").split("\n"):
        m = re.match(r"^#{1,4}\s+(.+)$", line.strip())
        if m and len(m.group(1)) <= 60:
            out.append(m.group(1).strip())
    return out


def _form_comparison(files) -> dict | None:
    """빈 서식과 완료 문서를 1:1로 맞대어 본다(데모 모드).

    양식·완료 문서가 함께 있을 때, 또는 양식 안에 빈칸·미기재가 있을 때 구성한다.
    """
    forms = [f for f in files if f.ok and f.category == "양식"]
    docs = [f for f in files if f.ok and f.category == "완료 문서"]
    if not forms:
        return None
    form = forms[0]
    has_blank = any(w in form.text for w in _PLACEHOLDER_WORDS)
    doc = docs[0] if docs else None
    if doc is None and not has_blank:
        return None

    flines = _file_lines(form)
    dlines = _file_lines(doc) if doc else []
    rows: list[dict] = []
    blank: list[tuple[str, str | None]] = []
    missing: list[tuple[str, str]] = []
    mismatch: list[tuple[str, str, str]] = []
    from_doc: list[tuple[str, str]] = []

    for label, syns in _COMPARE_FIELDS:
        kind = "budget" if label == "예산 총액" else ("period" if label.startswith("실행") else None)
        fv = _lookup(flines, syns, kind, strict=True)
        dv = _lookup(dlines, syns, kind) if dlines else None
        if fv is None and dv is None:
            continue

        if fv is None:
            status = "서식에 없음(완료 문서만)"
            from_doc.append((label, dv or ""))
        elif _is_blank(fv) and _is_blank(dv):
            status = "양식 빈칸 · 완료 문서에도 없음"
            blank.append((label, fv))
        elif _is_blank(fv):
            status = "완료 문서에서 채움"
            from_doc.append((label, dv or ""))
        elif doc is None:
            status = "양식에 기재됨"
        elif _is_blank(dv):
            status = "완료 문서에 없음"
            missing.append((label, fv))
        else:
            same = True
            if kind == "budget":
                same = bool(set(_amount_values(fv)) & set(_amount_values(dv)))
            elif kind == "period":
                same = bool(set(_dates(fv)) & set(_dates(dv)))
            if kind is None:
                status = "확인됨"
            else:
                status = "일치" if same else "값 다름"
                if not same:
                    mismatch.append((label, fv, dv))
        rows.append({"label": label, "form": fv, "doc": dv, "status": status})

    extra_sections: list[str] = []
    if doc:
        form_norm = _norm_label(form.text)
        for h in _doc_headings(doc):
            words = [w for w in re.findall(r"[가-힣A-Za-z]{3,}", h) if w not in _STOPWORDS]
            if words and not any(w in form_norm for w in words):
                extra_sections.append(h)
        extra_sections = extra_sections[:6]

    confirmed = sum(1 for r in rows if r["status"] in ("확인됨", "일치", "완료 문서에서 채움"))
    return {
        "form": form, "doc": doc, "rows": rows, "blank": blank, "missing": missing,
        "mismatch": mismatch, "from_doc": from_doc, "extra_sections": extra_sections,
        "confirmed": confirmed,
    }


def _comparison_md(cmp: dict) -> list[str]:
    """1차 초안 맨 앞에 들어갈 '0. 빈 서식 vs 완료 문서 대조 요약'."""
    form, doc = cmp["form"], cmp["doc"]
    md = ["## 0. 빈 서식 vs 완료 문서 대조 요약", ""]
    if doc is None:
        md.append(
            f"> 양식 `{form.short_name}` 하나만 올라와 있어, 양식 안의 빈칸·미기재 항목만 짚었습니다. "
            "같은 업무의 완료 문서를 함께 올리면 항목별로 무엇이 어떻게 채워졌는지 1:1로 대조해 드립니다."
        )
    else:
        md.append(
            f"> 서식 `{form.short_name}` ↔ 완료 문서 `{doc.short_name}`를 항목 단위로 맞대어 봤습니다. "
            "상태가 '값 다름'인 항목은 제출 전에 어느 쪽이 맞는지 확인해야 합니다."
        )
    md.append("")
    if doc is None:
        filled = sum(1 for r in cmp["rows"] if r["status"] == "양식에 기재됨")
        md.append(f"- 양식 항목 {len(cmp['rows'])}개 중 기재됨 {filled}개 · 빈칸 {len(cmp['blank'])}개")
    else:
        md.append(
            f"- 양식 항목 {len(cmp['rows'])}개 중 완료 문서에서 확인 {cmp['confirmed']}개 · "
            f"완료 문서에 없음 {len(cmp['missing'])}개 · 양식 빈칸 {len(cmp['blank'])}개 · "
            f"서식에 없는 추가 항목 {len(cmp['from_doc'])}개"
        )
    if cmp["extra_sections"]:
        md.append(f"- 완료 문서에만 있는 목차: {', '.join(cmp['extra_sections'][:4])}")
    md.append("")
    if cmp["rows"]:
        table = [[r["label"], _plain(r["form"], 55) or "—", _plain(r["doc"], 55) or "—", r["status"]]
                 for r in cmp["rows"]]
        md.append(_table(["양식 항목", "양식에 적힌 값", "완료 문서에서 채운 내용", "상태"], table))
        md.append("")

    if cmp["from_doc"]:
        md.append("**서식엔 없지만 완료 문서에서 추가로 챙긴 내용**")
        for label, value in cmp["from_doc"][:8]:
            md.append(f"- **{label}** — {_plain(value, 100)}")
        if cmp["extra_sections"]:
            md.append(f"- 완료 문서에만 있는 절: {', '.join(cmp['extra_sections'])}")
        md.append("")

    if cmp["mismatch"]:
        md.append("**양식과 완료 문서의 값이 어긋나는 지점**")
        for label, fv, dv in cmp["mismatch"]:
            md.append(f"- **{label}** — 양식: {_plain(fv, 70)} / 완료 문서: {_plain(dv, 70)}")
        md.append("")

    if cmp["missing"]:
        md.append("**양식에는 있는데 완료 문서에 빠진 항목**")
        for label, fv in cmp["missing"]:
            md.append(f"- {label} (양식 기재: {_plain(fv, 60)})")
        md.append("")

    if cmp["blank"]:
        md.append("**양식이 비워 둔 칸 (후임자가 반드시 물어야 할 곳)**")
        for label, fv in cmp["blank"]:
            md.append(f"- {label}" + (f" (양식 표기: {fv})" if fv and not _is_blank(fv) else ""))
        md.append("")
    return md


# ----------------------------------------------------------------------------
# 질문 세트 — 자료의 실제 공백을 찌르고, 개수는 자료에 따라 달라진다
# ----------------------------------------------------------------------------

def _question_set(fx: Facts, cmp: dict | None = None) -> list[dict]:
    """자료에서 확인된 공백만 질문으로 만든다. 개수는 고정하지 않는다(5~15).

    각 질문은 priority('필수'/'권장')와 category를 가진다.
    """
    files = fx.files
    fname0 = files[0].short_name if files else "자료"
    cands: list[dict] = []

    def add(question: str, why: str, needles: tuple[str, ...] = (), *,
            priority: str = "필수", category: str = "예외·규정",
            window: int = 0, evidence: str | None = None):
        if evidence is None:
            f, q = _ev(files, needles, window) if needles else ("", "")
            evidence = f"{q} — {f}" if q else f"{fname0} 기준"
        cands.append({"question": question, "why": why, "evidence": evidence,
                      "priority": priority, "category": category})

    # 1) 서식·완료 문서 대조에서 드러난 공백이 가장 구체적이다.
    if cmp:
        if cmp["mismatch"]:
            label, fv, dv = cmp["mismatch"][0]
            add(
                f"양식의 {label}({_plain(fv, 40)})과(와) 완료 문서의 값({_plain(dv, 40)})이 서로 다릅니다. "
                "제출된 확정 값은 어느 쪽이고, 차이가 생긴 이유와 근거는 무엇입니까?",
                "두 문서의 값이 어긋난 채로 넘어가면 후임자가 잘못된 금액·기한으로 업무를 처리합니다.",
                priority="필수", category="서식·작성법",
                evidence=f"양식: {_plain(fv, 60)} / 완료 문서: {_plain(dv, 60)}",
            )
        if cmp["blank"]:
            labels = ", ".join(label for label, _v in cmp["blank"][:5])
            add(
                f"양식의 {labels} 칸이 비어 있습니다('추후 기재'). 실제로 적어 넣은 값과 "
                "담당자 부재 시 대체 창구는 어디입니까?",
                "비어 있는 칸을 채우지 못하면 후임자가 첫 문의를 어디로 넣어야 할지 몰라 일이 멈춥니다.",
                ("담당자", "연락처", "이메일"),
                priority="필수", category="연락·창구",
            )
        if cmp["missing"]:
            labels = ", ".join(label for label, _v in cmp["missing"][:5])
            add(
                f"양식에 있는 {labels} 항목이 완료 문서에서는 보이지 않습니다. 실제 제출본에는 어떻게 적었습니까?",
                "서식에 있는 칸을 빼고 제출하면 접수 단계에서 반려되거나 보완 요구를 받습니다.",
                priority="권장", category="서식·작성법",
                evidence=f"완료 문서에서 찾지 못한 양식 항목: {labels}",
            )
        if cmp["extra_sections"]:
            add(
                f"완료 문서에만 있는 '{cmp['extra_sections'][0]}' 내용은 서식에 없는 항목입니다. "
                "제출본에 넣지 않아도 되는 내용입니까, 아니면 서식에 추가해 달라고 요청해야 합니까?",
                "서식에 없는 내용을 빼고 제출했다가 기업이 요구하면 처음부터 다시 쓰게 됩니다.",
                priority="권장", category="서식·작성법",
                evidence=f"완료 문서에만 있는 절: {cmp['extra_sections'][0]}",
            )

    if fx.placeholders and not (cmp and cmp["blank"]):
        add(
            "자료의 담당자 연락처·이메일이 '추후 기재'로 비어 있습니다. 실제 연락 창구(성명·직통번호·메일)와 "
            "담당자 부재 시 대체 창구는 누구입니까?",
            "비어 있는 칸을 그대로 넘기면 후임자가 첫 문의를 어디로 넣어야 할지 몰라 일이 멈춥니다.",
            ("담당자 연락처", "연락처", "이메일"), window=1,
            priority="필수", category="연락·창구",
        )
    if fx.period:
        add(
            f"실행기간이 '{fx.period}'로만 적혀 있습니다. 중간보고·정산·최종 제출의 실제 마감일과 "
            "제출처(양식 포함)는 각각 언제·어디입니까?",
            "기간만 있고 중간 마감이 없어, 마감을 놓치면 되돌릴 수 없는 지연이 생깁니다.",
            (fx.period[:10], "마감", "제출"),
            priority="필수", category="일정·마감",
        )
    if fx.budget_total:
        add(
            f"예산 {fx.budget_total}의 집행·증빙 절차는 어떻게 됩니까? (품의선, 증빙 양식, 정산 담당자, 잔액 처리)",
            "금액만 있고 집행·정산 절차가 없으면 돈을 쓰고도 증빙을 못 맞춰 개인이 부담하게 됩니다.",
            ("총 계", "총계", "예산 총액", "증빙"),
            priority="필수", category="예산·증빙",
        )
    if fx.members:
        names = ", ".join(m[1] for m in fx.members[:4])
        add(
            f"역할 분담표에 {names} 등이 있지만 각자 업무의 경계가 없습니다. 겹치는 업무와 "
            "최종 결재·승인권자는 누구입니까?",
            "역할만 나열돼 있어, 일이 겹치거나 비었을 때 누가 책임지는지 판단할 수 없습니다.",
            (fx.members[0][1], "역할"),
            priority="필수", category="승인·결재",
        )
    if fx.weeks:
        add(
            f"일정표가 {fx.weeks[0]}부터 {fx.weeks[-1]}까지 주 단위로만 짜여 있습니다. 실제로 지연되거나 "
            "빠진 주차가 있었다면 어느 주차이고, 그때 무엇으로 대체했습니까?",
            "계획표만 보면 '계획대로 됐다'고 오해해 이미 밀린 일을 새 일처럼 다시 잡게 됩니다.",
            (fx.weeks[0], "추진일정"),
            priority="권장", category="일정·마감",
        )
    if fx.share:
        add(
            f"진행 공유를 '{fx.share[:30]}' 식으로 했다고 되어 있습니다. 최종본·증빙이 보관된 위치와 "
            "후임자에게 넘겨야 할 편집 권한은 무엇입니까?",
            "도구 이름만 있고 위치·권한이 없으면, 후임자가 계정을 넘겨받지 못해 기록이 끊깁니다.",
            ("노션", "Notion", "공유"),
            priority="필수", category="서식·작성법",
        )
    if fx.mentor:
        add(
            f"'{fx.mentor[:50]}' 라인으로 보고가 있었습니다. 보고 주기, 보고 양식, 지적받은 사항 중 "
            "다시 나올 만한 것은 무엇입니까?",
            "보고 라인만 남으면 같은 지적을 반복해 신뢰가 먼저 깎입니다.",
            ("멘토", "보고"),
            priority="권장", category="예외·규정",
        )
    if fx.bullets:
        add(
            f"'{fx.bullets[min(1, len(fx.bullets) - 1)][:40]}…' 항목은 실제로 어디까지 진행됐습니까? "
            "(완료 / 진행중 / 보류)",
            "자료는 계획만 말하고 현재 상태를 말하지 않아, 후임자가 끝난 일을 다시 합니다.",
            (fx.bullets[min(1, len(fx.bullets) - 1)][:12],),
            priority="권장", category="예외·규정",
        )
    add(
        f"{fname0} 외에 개인 메일·메신저에만 남아 있는 결정 사항이 있습니까? (누가, 언제, 무엇을 정했는지)",
        "문서에 없는 결정이 개인 대화에 남아 있으면, 후임자는 같은 논의를 처음부터 다시 하게 됩니다.",
        ("메일", "카카오톡", "메신저"),
        priority="권장", category="연락·창구",
    )
    add(
        "이 업무에서 '이건 꼭 물어보라'고 했던 예외 상황이 있습니까?",
        "예외는 문서에 잘 남지 않아, 후임자가 처음 겪을 때 되돌릴 수 없는 실수가 됩니다.",
        ("예외", "유의"),
        priority="권장", category="예외·규정",
    )

    out, seen = [], set()
    for c in cands:
        key = c["question"][:18]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)

    # 자료가 아주 적어도 다섯 개는 물어볼 수 있게 보강한다.
    fallbacks = [
        ("이 업무를 처음 맡는 사람이 첫 주에 가장 먼저 해야 할 일은 무엇입니까?",
         "시작 순서가 없으면 첫날이 비어 버립니다.", "예외·규정"),
        ("이 업무에서 자주 틀리는 지점(반려·재작성 원인)은 무엇입니까?",
         "같은 실수를 반복하면 신뢰가 먼저 깎입니다.", "예외·규정"),
        ("일이 막혔을 때 어디까지 혼자 처리하고, 어디서부터 결재를 받아야 합니까?",
         "결재선을 모르면 작은 일도 멈춥니다.", "승인·결재"),
        ("최근에 급하게 처리해야 했던 일이 있었습니까? 그때의 판단 기준은 무엇이었습니까?",
         "급한 상황의 판단 기준은 문서에 남지 않습니다.", "일정·마감"),
        ("담당자가 바뀌면 가장 먼저 끊기는 정보는 무엇입니까?",
         "끊기는 지점을 알면 대비할 수 있습니다.", "연락·창구"),
    ]
    for question, why, category in fallbacks:
        if len(out) >= 5:
            break
        out.append({"question": question, "why": why, "evidence": f"{fname0} 기준",
                    "priority": "권장", "category": category})

    out = out[:15]
    for i, c in enumerate(out, 1):
        c["id"] = i
    return out


def _priority_counts(questions: list[dict]) -> tuple[int, int]:
    must = sum(1 for q in questions if q.get("priority") == "필수")
    return must, len(questions) - must


# ----------------------------------------------------------------------------
# Step 2 — 데모 초안
# ----------------------------------------------------------------------------

def demo_analysis(files) -> dict[str, Any]:
    fx = _facts(files)
    cmp = _form_comparison(files)
    n_chars = sum(f.char_count for f in files)
    cat: dict[str, int] = {}
    for f in files:
        cat[f.category] = cat.get(f.category, 0) + 1
    cat_txt = ", ".join(f"{k} {v}개" for k, v in cat.items()) or "없음"

    md: list[str] = [f"# {fx.title} — 인수인계서 1차 초안", ""]
    md.append(
        f"> **데모(시뮬레이션) 모드 결과입니다.** 실제 API를 호출하지 않고, 업로드한 자료 "
        f"{len(files)}개({n_chars:,}자)에서 뽑은 문장·수치를 근거로 구성했습니다. "
        "실제 LLM으로 다시 만들려면 사이드바에서 '실제 API 연동'을 고르세요."
    )
    md.append("")

    # 0. 빈 서식 vs 완료 문서 대조 요약 (양식이 있을 때)
    if cmp:
        md.extend(_comparison_md(cmp))

    # 자료 요약
    md.append("## 자료 요약")
    md.append(f"- 문서 {len(files)}개 ({cat_txt}), 총 {n_chars:,}자 추출")
    if fx.keywords:
        md.append(f"- 핵심 키워드: {', '.join(fx.keywords)}")
    if fx.placeholders:
        md.append(f"- 빈칸·미기재 표시 {fx.placeholders}건 — 후임자가 여기서 먼저 막힙니다")
    rows = []
    for label, value, needles in (
        ("프로젝트/업무명", fx.project, _ID_KEYS),
        ("실행 기간", fx.period, (fx.period[:10],) if fx.period else _PERIOD_KEYS),
        ("팀/부서", fx.team, _TEAM_KEYS),
        ("예산", fx.budget_total, ("총 계", "총계")),
        ("보고 라인", fx.mentor, ("멘토",)),
        ("공유 수단", fx.share, ("노션", "Notion", "공유")),
    ):
        if value:
            f, _q = _ev(files, needles)
            short = next((x.short_name for x in files if x.name == f), f)
            rows.append([label, value, short or "-"])
    if rows:
        md.append("")
        md.append(_table(["항목", "확인된 내용", "출처"], rows))
    md.append("")

    # ① 업무 절차
    md.append("## ① 업무 절차")
    md.append("")
    step = 1
    md.append(f"**{step}. 업무 정의**")
    if fx.bullets:
        f, q = _ev(files, (fx.bullets[0][:14],))
        md.append(f"- 인계 대상: {fx.bullets[0][:140]}")
        md.append(f"- 근거: {f or '-'} {q}")
    else:
        md.append("- 자료에서 업무명을 확정하지 못했습니다(인계자 확인 필요).")
    md.append("")
    step += 1
    md.append(f"**{step}. 정기 진행 방식**")
    comm = _routine_lines(files, limit=3)
    if comm:
        for fname, line in comm:
            md.append(f"- {_strip_marker(line)[:160]}  *(근거: {fname})*")
    else:
        md.append("- 정기 미팅·보고 주기가 자료에 없습니다(인계자 확인 필요).")
    md.append("")
    step += 1
    if fx.members:
        md.append(f"**{step}. 역할 분담 (연락 창구 포함)**")
        md.append("")
        md.append(_table(["역할", "이름", "담당"],
                         [[m[0], m[1], m[2]] for m in fx.members]))
        md.append("")
        md.append("- 인계할 때 각 담당자에게 '내 업무가 어디까지인지' 한 줄씩 확인받습니다.")
        md.append("")
        step += 1
    if fx.weeks:
        md.append(f"**{step}. 주차별 진행**")
        f, q = _ev(files, ("추진일정", fx.weeks[0]), window=2)
        md.append(f"- 자료의 일정표는 {fx.weeks[0]}부터 {fx.weeks[-1]}까지 {len(fx.weeks)}주 구성입니다.")
        md.append(f"- 근거: {f or '-'} {q}")
        md.append("")

    # ② 숨은 예외 FAQ
    questions = _question_set(fx, cmp)
    md.append("## ② 숨은 예외 FAQ")
    md.append("> 자료에 답이 없는 항목입니다. Step 3에서 인계자 답변을 받아 채웁니다.")
    md.append("")
    for q in questions:
        md.append(f"**Q{q['id']}. {q['question']}**")
        md.append(f"- 왜 막히는가: {q['why']}")
        md.append("- A: *(인계자 답변 대기)*")
        md.append(f"- 근거: {q['evidence']}")
        md.append("")

    # ③ 마감표
    md.append("## ③ 월간/연간 마감표")
    rows = []
    if fx.period:
        for mon in _period_months(fx.period):
            rows.append([mon, "월간 점검 (자료 기간 기준)", "추정 — 실제 마감일 미확인"])
    for fname, line in _scan(files, ("마감", "정산"))[:2]:
        rows.append(["상시", line[:50], fname])
    if fx.mentor:
        rows.append(["2주마다", f"멘토 보고 — {fx.mentor[:40]}", "자료 기재"])
    rows.append(["프로젝트 종료", "산출물·증빙 최종 정리", "미확인 — 종료일 역산 필요"])
    md.append(_table(["시기", "마감 항목", "상태/출처"], rows))
    md.append("")
    if fx.dates:
        md.append(f"- 자료에 나온 날짜: {', '.join(fx.dates)}")
        md.append("")

    # ④ 파일 맵
    md.append("## ④ 서식·참고파일 맵")
    use_map = {"양식": "작성 서식 — 그대로 채워 제출", "완료 문서": "작성 사례·수치 참고",
               "규정·메모": "판단 기준·배경 근거", "미분류": "용도 확인 필요"}
    rows = [[f.short_name, f.category, f"{f.char_count:,}자", use_map.get(f.category, "-"),
             ("⚠ " + f.warnings[0][:40]) if f.warnings else "정상"] for f in files]
    md.append(_table(["파일", "성격", "분량", "쓰는 곳", "상태"], rows))
    md.append("")

    md.append("## 후임자가 막힐 지점 (요약)")
    md.extend(f"- Q{q['id']}. {q['question']}" for q in questions)

    for q in questions:
        q["evidence"] = _shorten(q.get("evidence", ""), files)
    return {"draft_markdown": _shorten("\n".join(md), files), "questions": questions,
            "facts": fx, "mode": "demo"}


# ----------------------------------------------------------------------------
# Step 4 — 데모 최종본
# ----------------------------------------------------------------------------

def demo_final(files, draft_md: str, questions: list[dict], answers: dict,
               extra_notes: str = "", reviews: dict | None = None) -> str:
    fx = _facts(files)
    a: dict[int, str] = {}
    for k, v in (answers or {}).items():
        try:
            a[int(k)] = (v or "").strip()
        except (TypeError, ValueError):
            continue
    answered = sum(1 for v in a.values() if v)
    total_q = len(questions) or 5
    reviews = reviews or {}

    def review_of(qid: int) -> dict:
        return reviews.get(qid) or reviews.get(str(qid)) or {}

    md: list[str] = [f"# 업무 인수인계서 — {fx.title}", ""]
    md.append(f"- 작성 방식: 데모(시뮬레이션) · 자료 {len(files)}개에서 {sum(f.char_count for f in files):,}자 추출 기반")
    md.append(f"- 인계자 답변 반영: {answered}/{total_q}문항")
    if reviews:
        enough = sum(1 for q in questions if review_of(int(q.get("id", 0))).get("verdict") == "충분함")
        md.append(f"- AI 답변 검수: 충분함 {enough}건 / 보완 필요 {len(reviews) - enough}건")
    if fx.period:
        md.append(f"- 대상 기간: {fx.period}")
    if fx.team:
        md.append(f"- 소속·팀: {fx.team}")
    if fx.budget_total:
        md.append(f"- 예산: {fx.budget_total}")
    md.append("")

    # ① 업무 절차
    md.append("## ① 업무 절차")
    md.append("")
    n = 1
    if fx.bullets:
        md.append(f"1. **업무 정의** — {fx.bullets[0][:150]}")
        f, q = _ev(files, (fx.bullets[0][:14],))
        md.append(f"   - 근거: {f or '-'} {q}")
    else:
        md.append("1. **업무 정의** — 미확인 (인계자 확인 필요)")
    n = 2
    md.append("")
    md.append(f"{n}. **일상 절차**")
    comm = _routine_lines(files, limit=3)
    if comm:
        for i, (fname, line) in enumerate(comm, 1):
            md.append(f"   {i}) {_strip_marker(line)[:150]}  *(근거: {fname})*")
    else:
        md.append("   - 자료에 정기 절차가 없음 — 미확인")
    n += 1
    md.append("")
    if fx.members:
        md.append(f"{n}. **역할과 연락 창구**")
        md.append("")
        md.append(_table(["역할", "이름", "담당 업무", "인계 상태"],
                         [[m[0], m[1], m[2], "구두 확인 완료" if answered else "확인 예정"]
                          for m in fx.members]))
        md.append("")
        n += 1
    md.append(f"{n}. **인계자가 붙인 실제 요령**")
    notes = [(qid, ans) for qid, ans in sorted(a.items()) if ans]
    if notes:
        for qid, ans in notes:
            md.append(f"- (Q{qid}) {ans}")
    else:
        md.append("- 아직 입력된 답변이 없습니다 — Step 3에서 받은 답변이 여기에 들어갑니다.")
    md.append("")

    # ② 숨은 예외 FAQ
    md.append("## ② 숨은 예외 FAQ")
    md.append("")
    for q in questions:
        qid = int(q.get("id", 0))
        md.append(f"**Q{qid}. {q['question']}**")
        md.append("")
        md.append(f"- **A.** {a.get(qid) or '미확인 — 인계자 확인 필요'}")
        md.append(f"- 근거: {q.get('evidence', '')}")
        md.append("")

    # ③ 마감표
    md.append("## ③ 월간·연간 마감표")
    md.append("")
    rows = []
    if fx.period:
        for mon in _period_months(fx.period):
            rows.append([mon, "월간 점검·보고", "미확인 — 실제 마감일 확인 필요"])
    for qid, ans in sorted(a.items()):
        qtxt = next((q["question"] for q in questions if int(q.get("id", 0)) == qid), "")
        if ans and any(k in qtxt for k in ("마감", "보고", "정산", "제출")):
            rows.append(["인계자 확인", qtxt[:36] + "…", ans[:70]])
    rows.append(["프로젝트 종료", "산출물·증빙 최종 정리", "미확인 — 종료일 역산 필요"])
    md.append(_table(["시기", "마감 항목", "확정 내용/상태"], rows))
    md.append("")

    # ④ 파일 맵
    md.append("## ④ 서식·참고파일 맵")
    md.append("")
    use_map = {"양식": "제출용 서식 — 그대로 채워 사용", "완료 문서": "작성 사례·수치 참고",
               "규정·메모": "판단 기준·배경", "미분류": "용도 확인 필요"}
    md.append(_table(["파일", "성격", "분량", "쓰는 곳", "원본 위치"],
                     [[f.short_name, f.category, f"{f.char_count:,}자", use_map.get(f.category, "-"),
                       f.origin or "업로드 원본"] for f in files]))
    md.append("")
    store = ""
    for q in questions:
        qtext = q.get("question", "")
        if any(k in qtext for k in ("보관", "권한", "공유")):
            cand = a.get(int(q.get("id", 0)), "")
            if cand:
                store = cand
                break
    if not store:
        store = next((v for v in a.values() if v and any(k in v for k in ("노션", "드라이브", "보관", "폴더"))), "")
    md.append(f"- **보관 위치·권한:** {store or '미확인 — 인계자 확인 필요'}")
    md.append("")

    # ⑤ 기타 인계 메모 (인계자가 직접 적은 자유 메모)
    if (extra_notes or "").strip():
        md.append("## ⑤ 기타 인계 메모 (인계자 직접 작성)")
        md.append("")
        for line in extra_notes.strip().split("\n"):
            md.append(f"- {line.strip()}" if line.strip() else "")
        md.append("")

    # 답변 검수 이력
    if reviews:
        md.append("## AI 답변 검수 이력")
        md.append("")
        rows = []
        for q in questions:
            qid = int(q.get("id", 0))
            rv = review_of(qid)
            if not rv:
                continue
            rows.append([f"Q{qid}", rv.get("verdict", ""), (rv.get("comment") or "")[:90],
                         (rv.get("requestion") or "—")[:90]])
        if rows:
            md.append(_table(["질문", "판정", "검수 의견", "재질문"], rows))
            md.append("")

    md.append("## 인계자 확인 필요 목록")
    missing = [f"- Q{int(q.get('id', 0))}. {q['question']}" for q in questions
               if not a.get(int(q.get("id", 0)))]
    md.extend(missing or ["- 없음 (모든 문항에 답변이 반영되었습니다)"])
    md.append("")
    md.append("---")
    md.append(f"> 데모 모드 산출물입니다. 사이드바에서 실제 API를 연결하면 같은 구조를 실제 LLM이 다시 작성합니다. "
              f"원자료: {', '.join(f.short_name for f in files)}")
    return _shorten("\n".join(md), files)


# ----------------------------------------------------------------------------
# 인계자 답변 검수 (데모·실제 API 공통)
# ----------------------------------------------------------------------------

# 후임자가 실행할 수 없는 모호한 표현
_VAGUE_PHRASES = (
    "알아서", "적당히", "대충", "그때그때", "상황에 따라", "상황봐서", "상황 봐서",
    "알아서들", "비슷하게", "기존대로", "적절히", "봐서", "나중에", "팀장님께",
    "팀장님한테", "문의하세요", "문의하면", "확인 후", "담당자에게 물어",
)

_WHO_RE = re.compile(
    r"(팀장|팀원|대리|과장|차장|부장|담당|교수|실장|센터장|주무관|사원|매니저|전원)"
    r"|[가-힣]{2,4}\s?(님|씨)"
)
_WHEN_RE = re.compile(
    r"(\d{1,2}\s?월|\d{1,2}\s?일|매월|매주|매일|매년|분기|주\s?\d|\d\s?주차|주마다|주에\s?한|격주|"
    r"까지|이내|당일|익일|월말|연말|종료\s?전|20\d{2}|상시|수시|월초|중순)"
)
_HOW_RE = re.compile(
    r"(메일|이메일|노션|Notion|드라이브|카톡|카카오톡|전화|양식|서식|품의|결재|전자결재|"
    r"시스템|링크|폴더|문서|엑셀|공유|업로드|출력|발송|제출|보관|등록|신청|미팅|회의|보고)"
)


def _contradiction(fx: Facts, text: str) -> str | None:
    """답변의 숫자·기한이 업로드 자료와 어긋나는지 본다."""
    amounts = _amount_values(text)
    if amounts and fx.budget_amount:
        try:
            total = int(fx.budget_amount.replace(",", ""))
        except ValueError:
            total = 0
        if total and all(abs(a - total) > total * 0.2 for a in amounts):
            shown = ", ".join(f"{a:,}원" for a in amounts[:3])
            return (f"자료의 예산({fx.budget_total or f'{total:,}원'})과 답변의 금액({shown})이 "
                    "20% 넘게 어긋납니다")
    if fx.period:
        span = set(_dates(fx.period))
        years = {re.search(r"20\d{2}", d).group(0) for d in span if re.search(r"20\d{2}", d)}
        answer_years = set(re.findall(r"20\d{2}", text))
        if years and answer_years and not (years & answer_years):
            return (f"자료의 기간({fx.period})과 답변의 연차({', '.join(sorted(answer_years))}년)가 "
                    "서로 맞지 않습니다")
    return None


def _review_answer(fx: Facts, question: dict, answer: str) -> dict:
    """답변 하나를 검수해 {verdict, comment, requestion, issues}를 만든다(데모)."""
    text = (answer or "").strip()
    if not text:
        return {
            "verdict": "보완 필요",
            "comment": "답변이 비어 있습니다. 이 질문은 후임자가 첫 업무를 시작할 때 바로 막히는 지점입니다.",
            "requestion": "누가, 언제까지, 어떤 양식·창구로 처리하는지 한 문장으로 적어 주세요.",
            "issues": ["빈 답변"],
        }

    issues: list[str] = []
    requestion: list[str] = []
    compact = re.sub(r"\s+", "", text)

    if len(compact) < 12:
        issues.append("너무 짧음")
        requestion.append("한 문장 이상으로, 처리 주체와 방법을 함께 적어 주세요.")

    who = bool(_WHO_RE.search(text))
    when = bool(_WHEN_RE.search(text))
    how = bool(_HOW_RE.search(text))
    missing_axes = [name for name, ok in (("누가", who), ("언제", when), ("어떻게", how)) if not ok]
    if len(missing_axes) >= 2 and len(compact) < 45:
        issues.append("실행 기준 부족")
        requestion.append(
            "" + "/".join(missing_axes) + "를 넣어 다시 적어 주세요"
            "(예: '김소희 팀원이 매월 말일까지 노션 예산 시트에 증빙을 올리고, 고승준 팀장이 다음 달 3일까지 결재')."
        )

    vague = [w for w in _VAGUE_PHRASES if w in text]
    if vague and (len(compact) < 30 or len(missing_axes) >= 2):
        issues.append("모호한 표현")
        requestion.append(f"'{vague[0]}' 대신 실제 담당자 이름과 처리 방법을 적어 주세요.")

    contra = _contradiction(fx, text)
    if contra:
        issues.append("자료와 불일치")
        requestion.append(f"{contra}. 어느 쪽이 맞는지 확인해 적어 주세요.")

    if not issues:
        return {
            "verdict": "충분함",
            "comment": "누가·언제·어떻게가 확인됩니다. 자료와 어긋나는 숫자·기한도 없습니다.",
            "requestion": "",
            "issues": [],
        }

    reason = " / ".join(dict.fromkeys(issues))
    clauses = []
    if "너무 짧음" in issues:
        clauses.append(f"답변이 {len(compact)}자로 짧습니다")
    if "실행 기준 부족" in issues:
        clauses.append(f"{', '.join(missing_axes)} 기준이 빠졌습니다")
    if "모호한 표현" in issues:
        clauses.append(f"'{vague[0]}' 같은 표현은 후임자가 그대로 실행할 수 없습니다")
    if contra:
        clauses.append(contra)
    comment = f"{reason}. " + ". ".join(clauses) + "."
    comment = re.sub(r"\.{2,}", ".", comment).strip()
    return {
        "verdict": "보완 필요",
        "comment": comment,
        "requestion": " ".join(dict.fromkeys(requestion)),
        "issues": issues,
    }


def demo_review_answers(files, questions: list[dict], answers: dict,
                        extra_notes: str = "") -> dict[str, Any]:
    """데모 검수 — 자료의 사실과 답변 길이·구체성·모순을 규칙으로 본다."""
    fx = _facts(files)
    reviews: dict[int, dict] = {}
    for q in questions:
        qid = int(q.get("id", 0))
        reviews[qid] = _review_answer(fx, q, answers.get(qid, "") or answers.get(str(qid), ""))
    enough = sum(1 for r in reviews.values() if r["verdict"] == "충분함")
    need = len(reviews) - enough
    summary = (f"{len(reviews)}문항 중 충분함 {enough}건, 보완 필요 {need}건. "
               + ("필수 항목의 빈 답변부터 고치세요." if need else "그대로 최종본으로 넘어가도 됩니다."))
    return {"reviews": reviews, "summary": summary, "mode": "demo"}


def _normalize_reviews(data: dict, questions: list[dict]) -> dict[str, Any]:
    """API 응답을 id 기준 dict로 정리하고, 빠진 문항은 보완 필요로 채운다."""
    got: dict[int, dict] = {}
    for r in data.get("reviews") or []:
        try:
            qid = int(r.get("id"))
        except (TypeError, ValueError):
            continue
        verdict = "충분함" if str(r.get("verdict", "")).strip() == "충분함" else "보완 필요"
        got[qid] = {"verdict": verdict, "comment": (r.get("comment") or "").strip(),
                    "requestion": (r.get("requestion") or "").strip(), "issues": []}
    for q in questions:
        qid = int(q.get("id", 0))
        got.setdefault(qid, {"verdict": "보완 필요", "comment": "이 문항은 검수 결과가 오지 않았습니다.",
                             "requestion": "다시 검수받아 주세요.", "issues": []})
    enough = sum(1 for r in got.values() if r["verdict"] == "충분함")
    return {"reviews": got,
            "summary": data.get("summary") or f"충분함 {enough}건 / 보완 필요 {len(got) - enough}건",
            "mode": "api"}


def run_review(files, questions: list[dict], answers: dict, *, mode: str,
               base_url: str = "", api_key: str = "", model: str = "",
               extra_notes: str = "") -> dict[str, Any]:
    if mode == "demo":
        return demo_review_answers(files, questions, answers, extra_notes)
    raw = call_llm(base_url=base_url, api_key=api_key, model=model,
                   messages=prompt_review_answers(files, questions, answers, extra_notes))
    return _normalize_reviews(_parse_json_block(raw), questions)


# ----------------------------------------------------------------------------
# 공통 진입점
# ----------------------------------------------------------------------------

def run_analysis(files, *, mode: str, base_url: str = "", api_key: str = "",
                 model: str = "") -> dict[str, Any]:
    if mode == "demo":
        return demo_analysis(files)
    raw = call_llm(base_url=base_url, api_key=api_key, model=model,
                   messages=prompt_analysis(files))
    data = _parse_json_block(raw)
    qs = [q for q in (data.get("questions") or []) if isinstance(q, dict) and q.get("question")]
    if len(qs) < 5:
        have = {str(q.get("question"))[:18] for q in qs}
        for q in demo_analysis(files)["questions"]:
            if len(qs) >= 5:
                break
            if q["question"][:18] in have:
                continue
            qs.append(q)
    for i, q in enumerate(qs, 1):
        q.setdefault("id", i)
        q["id"] = i
        if q.get("priority") not in ("필수", "권장"):
            q["priority"] = "권장"
        if not q.get("category"):
            q["category"] = "예외·규정"
    return {"draft_markdown": data.get("draft_markdown", ""), "questions": qs,
            "mode": "api", "raw": raw}


def run_final(files, draft_md: str, questions: list[dict], answers: dict,
              *, mode: str, base_url: str = "", api_key: str = "", model: str = "",
              extra_notes: str = "", reviews: dict | None = None) -> str:
    if mode == "demo":
        return demo_final(files, draft_md, questions, answers,
                          extra_notes=extra_notes, reviews=reviews)
    return call_llm(base_url=base_url, api_key=api_key, model=model,
                    messages=prompt_final(files, draft_md, questions, answers, extra_notes))
