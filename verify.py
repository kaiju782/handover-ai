"""
verify.py — 자체 점검 스크립트 (v2)
==================================
업로드 없이도 파서와 데모 생성기가 정상 동작하는지 확인한다.

    python verify.py                       # 내장 샘플(양식 HWP + 완료 문서)로 점검
    python verify.py 파일1 파일2 ...        # 원하는 파일로 점검

확인 항목
  1) 파일별 텍스트 추출 글자 수(한글 글자 수 포함) · 자동 분류
  2) 민감정보 마스킹 동작
  3) 빈 서식 ↔ 완료 문서 1:1 대조 요약 생성
  4) 데모 모드 질문이 5~15개로 유동적이고, 모두 priority/category를 갖는지
  5) 모호한 답변 → '보완 필요', 구체적인 답변 → '충분함' 판정
  6) 기타 인계 메모·검수 이력이 최종본에 들어가고 Word로 변환되는지
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import exporters
import llm
import parsers

SAMPLE_GLOBS = [
    "attachments/*/*.hwp",
    "../attachments/*/*.hwp",
]
SAMPLE_EXTRA = ["../무사경영_수행계획서_초안.md", "samples/*"]


def find_samples() -> list[Path]:
    """실제 HWP가 있으면 그것으로, 없으면 동봉한 samples/로 점검한다."""
    found: list[Path] = []
    for pattern in SAMPLE_GLOBS:
        found.extend(Path(p) for p in glob.glob(pattern))
    if found:
        found = found[:1]
        found += [Path(p) for p in SAMPLE_EXTRA[:1] if Path(p).exists()]
        return found
    for pattern in SAMPLE_EXTRA:
        found.extend(Path(p) for p in glob.glob(pattern))
    return found[:4]


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv[1:]] or find_samples()
    if not paths:
        print("점검할 파일이 없습니다. 사용법: python verify.py <파일> [<파일> ...]")
        return 1

    problems: list[str] = []

    print("=" * 78)
    print(f"1) 텍스트 추출 — {len(paths)}개 파일")
    print("=" * 78)
    files = parsers.extract_uploads([(p.name, p.read_bytes()) for p in paths])
    for f in files:
        hangul = sum(1 for c in f.text if "\uac00" <= c <= "\ud7a3")
        print(f"\n■ {f.short_name}")
        print(f"   형식      : {f.ext}  ({f.size:,} bytes)")
        print(f"   추출 글자 : {f.char_count:,}자  (한글 {hangul:,}자)")
        print(f"   자동 분류 : {f.category} — {f.category_reason}")
        print(f"   마스킹    : {f.masked_count}건")
        if f.warnings:
            for w in f.warnings:
                print(f"   ⚠ {w}")
        if f.text:
            head = f.text[:120].replace("\n", " ⏎ ")
            print(f"   미리보기  : {head}…")

    print("\n" + "=" * 78)
    print("2) 민감정보 마스킹")
    print("=" * 78)
    sample = ("연락처 010-1234-5678 / 주민 900101-1234567 / 계좌 123-456-789012 "
              "/ 카드 1234-5678-1234-5678 / 날짜 2026-11-21")
    masked, n = parsers.mask_sensitive(sample)
    print(f"   원문  : {sample}")
    print(f"   마스킹: {masked}   ({n}건)")
    if n < 4:
        problems.append("민감정보 마스킹 건수가 4건 미만")

    print("\n" + "=" * 78)
    print("3) 데모 모드 분석 — 서식 대조 + 유동 질문")
    print("=" * 78)
    result = llm.run_analysis(files, mode="demo")
    draft = result["draft_markdown"]
    questions = result["questions"]

    has_cmp = "## 0. 빈 서식 vs 완료 문서 대조 요약" in draft
    print(f"   1차 초안     : {len(draft):,}자")
    print(f"   0장 서식 대조: {'있음' if has_cmp else '없음'}")
    if not has_cmp:
        problems.append("1차 초안에 '0. 빈 서식 vs 완료 문서 대조 요약'이 없음")
    for part in ("자료 요약", "① 업무 절차", "② 숨은 예외 FAQ", "③ 월간/연간 마감표", "④ 서식·참고파일 맵"):
        print(f"     - '{part}' 포함: {part in draft}")
        if part not in draft:
            problems.append(f"초안에 '{part}' 누락")

    must = [q for q in questions if q.get("priority") == "필수"]
    rec = [q for q in questions if q.get("priority") == "권장"]
    print(f"   질문 개수    : {len(questions)} (필수 {len(must)} · 권장 {len(rec)})")
    if not 5 <= len(questions) <= 15:
        problems.append(f"질문 개수가 5~15 밖 ({len(questions)})")
    if not must or not rec:
        problems.append("priority가 필수·권장으로 나뉘지 않음")
    cats = {q.get("category") for q in questions}
    print(f"   카테고리     : {', '.join(sorted(c for c in cats if c))}")
    if len(cats) < 2:
        problems.append("category가 한 종류뿐")
    for q in questions:
        print(f"     Q{q['id']} [{q['priority']}·{q['category']}] {q['question'][:52]}…")

    print("\n" + "=" * 78)
    print("4) AI 답변 검수 — 모호한 답변 vs 구체적인 답변")
    print("=" * 78)
    vague_answers = {q["id"]: "" for q in questions}
    vague_answers[questions[0]["id"]] = "알아서 팀장님께 문의"
    rev_vague = llm.run_review(files, questions, vague_answers, mode="demo")
    v0 = rev_vague["reviews"][questions[0]["id"]]
    print(f"   모호한 답변 판정: {v0['verdict']}")
    print(f"     코멘트: {v0['comment'][:80]}")
    print(f"     재질문: {v0['requestion'][:80]}")
    if v0["verdict"] != "보완 필요":
        problems.append("모호한 답변이 '보완 필요'로 판정되지 않음")

    concrete = ("정산은 김소희 팀원이 매월 말일까지 노션 예산 시트에 증빙을 올리고, "
                "고승준 팀장이 다음 달 3일까지 결재합니다.")
    good_answers = {q["id"]: concrete for q in questions}
    rev_good = llm.run_review(files, questions, good_answers, mode="demo")
    ok_n = sum(1 for r in rev_good["reviews"].values() if r["verdict"] == "충분함")
    print(f"   구체적 답변 판정: 충분함 {ok_n}/{len(questions)}")
    if ok_n == 0:
        problems.append("구체적인 답변이 '충분함'으로 판정되지 않음")
    print(f"   총평: {rev_good['summary']}")

    print("\n" + "=" * 78)
    print("5) 최종본 — 메모·검수 이력 반영 + Word 변환")
    print("=" * 78)
    memo = "팀 노션 '회의록 아카이브'에 결정 사항이 모여 있습니다."
    final = llm.run_final(files, draft, questions, good_answers, mode="demo",
                          extra_notes=memo, reviews=rev_good["reviews"])
    print(f"   최종본       : {len(final):,}자")
    missing = [p for p in ("① 업무 절차", "② 숨은 예외 FAQ", "③ 월간·연간 마감표", "④ 서식·참고파일 맵")
               if p not in final]
    print(f"   4개 파트 누락: {missing or '없음'}")
    if missing:
        problems.append(f"최종본 파트 누락: {missing}")
    print(f"   답변 반영    : {'예' if '김소희 팀원이 매월 말일까지' in final else '아니오'}")
    print(f"   기타 메모    : {'예' if memo in final else '아니오'}")
    print(f"   검수 이력    : {'예' if 'AI 답변 검수 이력' in final else '아니오'}")
    if memo not in final:
        problems.append("기타 인계 메모가 최종본에 없음")
    if "AI 답변 검수 이력" not in final:
        problems.append("검수 이력이 최종본에 없음")

    docx = exporters.docx_bytes(final, "점검용 인수인계서")
    print(f"   Word 변환    : {len(docx):,} bytes")
    if len(docx) < 5000:
        problems.append("Word 변환 결과가 비정상적으로 작음")

    print("\n결과:", "✅ 모두 정상" if not problems else "❌ 확인 필요 — " + " / ".join(problems))
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
