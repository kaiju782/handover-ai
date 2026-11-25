"""
경영사무 AI 인수인계 도우미 — Streamlit 앱
=========================================
실행:  streamlit run app.py

4단계 흐름
  1) 자료 통째 업로드 (다중 파일 / ZIP → 자동 분류·텍스트 추출)
  2) AI 1차 분석 & 부족 정보 질문 5개
  3) 인계자 Q&A 보충
  4) 최종 인수인계서 완성 & Markdown / Word 내보내기
"""

from __future__ import annotations

import hashlib
import html
import re

import streamlit as st

import exporters
import llm
import parsers


# ----------------------------------------------------------------------------
# 페이지 설정
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="경영사무 AI 인수인계 도우미",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; max-width: 1180px; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      .stepbar { display:flex; gap:8px; margin: 6px 0 18px 0; }
      .stepbar div { flex:1; padding:9px 10px; border-radius:10px; font-size:0.82rem;
                     background:#f3f4f6; color:#6b7280; text-align:center; border:1px solid #e5e7eb; }
      .stepbar div.on { background:#1f6feb; color:#fff; border-color:#1f6feb; font-weight:600; }
      .stepbar div.done { background:#e8f1ff; color:#1f4fa8; border-color:#c9dbff; }
      .badge { display:inline-block; padding:3px 9px; border-radius:999px; font-size:0.75rem;
               background:#eef2ff; color:#3b4ea8; margin-right:6px; }
      .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.72rem;
              font-weight:600; margin-right:5px; vertical-align:middle; }
      .pill.must { background:#fdecec; color:#b42318; border:1px solid #f7c9c4; }
      .pill.rec { background:#eef2ff; color:#3b4ea8; border:1px solid #d9e0ff; }
      .pill.cat { background:#f3f4f6; color:#4b5563; border:1px solid #e5e7eb; font-weight:500; }
      .qmeta { color:#6b7280; font-size:0.8rem; margin:2px 0 6px 0; line-height:1.4; }
      .rev { margin-top:8px; padding:9px 11px; border-radius:9px; font-size:0.85rem; line-height:1.45; }
      .rev.ok { background:#ecfdf3; border:1px solid #b7ebc9; color:#05603a; }
      .rev.warn { background:#fff7e8; border:1px solid #f5d9a0; color:#8a5300; }
      .rev .rt { font-weight:700; margin-bottom:2px; }
      .rev .rq { margin-top:4px; font-size:0.82rem; opacity:0.92; }
      .qcard { border:1px solid #e5e7eb; border-radius:12px; padding:14px 16px; margin-bottom:10px;
               background:#fafbfc; }
      .qcard .q { font-weight:600; margin-bottom:4px; }
      .qcard .why { color:#6b7280; font-size:0.86rem; }
      .qcard .ev { color:#8a6d3b; font-size:0.82rem; margin-top:6px; }
      /* 문서 미리보기는 문서답게 보이되 화면을 압도하지 않도록 */
      [data-testid="stTabs"] h1 { font-size: 1.7rem; line-height: 1.35; }
      [data-testid="stTabs"] h2 { font-size: 1.25rem; margin-top: 1.1rem; }
      [data-testid="stTabs"] h3 { font-size: 1.05rem; }
      [data-testid="stTabs"] blockquote { font-size: 0.86rem; color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# 상태 초기화
# ----------------------------------------------------------------------------

DEFAULTS = {
    "step": 1,
    "files": [],
    "sig": None,
    "analysis": None,
    "questions": [],
    "answers": {},
    "reviews": {},
    "review_summary": None,
    "extra_notes": "",
    "final_md": None,
    "final_sig": None,
    "last_error": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def go(step: int):
    st.session_state.step = step
    st.rerun()


def stepbar(current: int):
    labels = ["1. 자료 업로드", "2. AI 1차 분석", "3. 인계자 Q&A·검수", "4. 최종 인수인계서"]
    cells = []
    for i, label in enumerate(labels, 1):
        cls = "on" if i == current else ("done" if i < current else "")
        cells.append(f'<div class="{cls}">{label}</div>')
    st.markdown(f'<div class="stepbar">{"".join(cells)}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 사이드바 — 실행 모드 / 민감정보
# ----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ 실행 설정")

    mode_label = st.radio(
        "실행 모드",
        ["데모 모드 (API 키 불필요)", "실제 API 연동 (K-water GPT / OpenAI 호환)"],
        index=0,
        help="데모 모드는 업로드한 자료의 실제 문장·수치로 초안을 만듭니다. 실제 API 모드는 설정한 LLM이 작성합니다.",
    )
    mode = "demo" if mode_label.startswith("데모") else "api"
    st.session_state["mode_label"] = mode_label
    st.session_state["mode"] = mode

    base_url, api_key, model = "", "", ""
    if mode == "api":
        base_url = st.text_input(
            "base_url", value="https://api.openai.com/v1",
            help="OpenAI 호환이면 그대로 동작합니다. K-water GPT는 담당자에게 받은 게이트웨이 주소(…/v1)를 넣으세요.",
        )
        model = st.text_input("model", value="gpt-4o-mini",
                             help="K-water GPT에서 안내받은 모델명으로 바꿔 주세요.")
        api_key = st.text_input("api_key", type="password",
                                help="입력값은 이 브라우저 세션에만 남고 파일로 저장되지 않습니다.")
        if st.button("연결 확인 (1회 호출)", use_container_width=True):
            try:
                with st.spinner("연결 확인 중…"):
                    reply = llm.call_llm(base_url=base_url, api_key=api_key, model=model,
                                         messages=[{"role": "user", "content": "핑" }],
                                         timeout=30)
                st.success(f"연결됨 — 응답: {reply[:60]}")
            except Exception as exc:
                st.error(f"연결 실패: {exc}")
        st.caption("※ '연결 확인'은 실제 API를 1회 호출합니다(제공처 정책에 따라 소액 과금될 수 있습니다).")
        st.divider()
    else:
        st.caption("API 키 없이 4단계 전체 흐름을 그대로 체험할 수 있습니다.")
    st.session_state["base_url"] = base_url
    st.session_state["api_key"] = api_key
    st.session_state["model"] = model

    st.markdown("### 🔒 민감정보")
    mask_on = st.toggle(
        "민감정보 자동 마스킹", value=True,
        help="주민등록번호·계좌번호·휴대폰·카드·사업자번호를 *로 가린 뒤 화면과 LLM에 전달합니다.",
    )
    st.session_state["_mask_on"] = mask_on

    if st.session_state.files:
        st.divider()
        stats = parsers.summary_stats(st.session_state.files)
        st.markdown("### 📊 현재 자료")
        st.caption(
            f"파일 {stats['files']}개 · 읽힘 {stats['readable']}개 · "
            f"{stats['chars']:,}자 · 마스킹 {stats['masked']}건"
        )

    st.divider()
    if st.button("처음부터 다시", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    with st.expander("이 앱에 대해"):
        st.markdown(
            "- 업로드한 파일은 이 실행 환경 안에서만 처리됩니다.\n"
            "- 실제 API 모드에서는 **추출된 텍스트가 설정한 LLM 제공처로 전송**됩니다. "
            "사내 자료라면 마스킹을 켜고, 반출 정책을 먼저 확인하세요.\n"
            "- ZIP은 압축을 풀어 안의 파일을 자동 분류합니다."
        )


# ----------------------------------------------------------------------------
# 헤더
# ----------------------------------------------------------------------------

st.title("🗂️ 경영사무 AI 인수인계 도우미")
st.caption("자료를 통째로 올리면 → 서식 1:1 대조 → 필요한 만큼 질문(필수/권장) → AI 답변 검수 → 인수인계서 완성")
stepbar(st.session_state.step)


# ----------------------------------------------------------------------------
# Step 1 — 자료 업로드
# ----------------------------------------------------------------------------

def render_upload():
    st.subheader("Step 1. 자료 통째 업로드")
    st.write(
        "양식, 완료 문서, 규정·메모를 **구분 없이 한꺼번에** 올리세요. ZIP으로 묶어 올려도 됩니다. "
        "(지원: hwp · hwpx · pdf · docx · xlsx · xls · csv · txt · md · zip)"
    )

    uploaded = st.file_uploader(
        "파일 선택 (여러 개 가능)",
        type=["hwp", "hwpx", "pdf", "docx", "xlsx", "xls", "csv", "txt", "md", "zip"],
        accept_multiple_files=True,
        key="uploader",
    )

    mask_on = st.session_state.get("_mask_on", True)
    payload = [(u.name, u.getvalue()) for u in uploaded] if uploaded else []
    sig = None
    if payload:
        h = hashlib.sha1()
        for name, data in payload:
            h.update(name.encode("utf-8"))
            h.update(hashlib.sha1(data).digest())
        sig = (h.hexdigest(), mask_on)

    if sig and sig != st.session_state.sig:
        with st.spinner("파일에서 글자를 뽑는 중…"):
            st.session_state.files = parsers.extract_uploads(payload, apply_mask=mask_on)
        st.session_state.sig = sig
        st.session_state.analysis = None
        st.session_state.questions = []
        st.session_state.answers = {}
        st.session_state.reviews = {}
        st.session_state.review_summary = None
        st.session_state.final_md = None

    files = st.session_state.files

    if not uploaded:
        st.info("먼저 자료를 올려 주세요. HWP 파일도 그대로 열립니다(별도 프로그램 설치 불필요).")
        return

    stats = parsers.summary_stats(files)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("파일", f"{stats['files']}개", border=True)
    c2.metric("읽힌 문서", f"{stats['readable']}개", border=True)
    c3.metric("추출 글자", f"{stats['chars']:,}자", border=True)
    c4.metric("마스킹", f"{stats['masked']}건", border=True)

    if stats["masked"]:
        st.caption(f"민감정보 {stats['masked']}건이 마스킹되었습니다. (사이드바에서 끌 수 있습니다)")

    st.markdown("#### 자동 분류 결과")
    rows = [{
        "파일": f.short_name,
        "형식": f.ext,
        "성격": f.category,
        "분류 근거": f.category_reason,
        "글자수": f"{f.char_count:,}",
        "상태": "정상" if f.ok else "읽기 실패",
    } for f in files]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    problems = [f for f in files if f.warnings]
    if problems:
        with st.expander(f"⚠ 확인이 필요한 파일 {len(problems)}개"):
            for f in problems:
                for w in f.warnings:
                    st.write(f"- **{f.short_name}** — {w}")

    with st.expander("📄 추출 텍스트 미리보기 (파일별)"):
        for f in files:
            st.markdown(f"**{f.short_name}** · {f.category} · {f.char_count:,}자")
            if f.text:
                st.text_area("본문", f.text[:4000] + ("…" if f.char_count > 4000 else ""),
                             height=180, key=f"preview_{hashlib.md5(f.name.encode()).hexdigest()}",
                             label_visibility="collapsed")
            else:
                st.caption("추출된 텍스트가 없습니다.")
            st.divider()

    if stats["readable"] == 0:
        st.error("읽을 수 있는 문서가 없습니다. 다른 형식으로 올려 주세요.")
        return

    st.divider()
    if st.button("Step 2로 — AI 1차 분석 시작 →", type="primary"):
        go(2)


# ----------------------------------------------------------------------------
# Step 2 — 1차 분석 + 질문 5개
# ----------------------------------------------------------------------------

def render_analysis():
    st.subheader("Step 2. AI 1차 분석 — 빈 서식 대조 · 필요한 만큼 질문")
    files = st.session_state.files
    if not files:
        st.warning("Step 1에서 자료를 먼저 올려 주세요.")
        if st.button("← Step 1로"): go(1)
        return

    mode = st.session_state.get("mode", "demo")
    analysis = st.session_state.analysis

    if analysis is not None and analysis.get("mode") != mode:
        st.warning("실행 모드가 바뀌었습니다. 이 모드로 다시 분석해 주세요.")
        if st.button("이 모드로 다시 분석"):
            st.session_state.analysis = None
            st.rerun()

    if analysis is None:
        st.write("추출된 자료를 읽고 **1차 인수인계서 초안**과 **후임자가 막힐 질문**을 만듭니다. "
                 "양식과 완료 문서가 함께 있으면 항목별로 1:1 대조하고, 질문 개수는 자료에 맞춰 달라집니다.")
        st.caption(f"문서 {len(files)}개 · {sum(f.char_count for f in files):,}자"
                   + (" · 데모 모드" if mode == "demo" else " · 실제 API"))
        if st.button("분석 시작", type="primary"):
            try:
                with st.spinner("자료를 읽는 중…"):
                    st.session_state.analysis = llm.run_analysis(
                        files, mode=mode,
                        base_url=st.session_state.get("base_url", ""),
                        api_key=st.session_state.get("api_key", ""),
                        model=st.session_state.get("model", ""),
                    )
                st.session_state.questions = st.session_state.analysis.get("questions", [])
                st.session_state.reviews = {}
                st.session_state.review_summary = None
                st.rerun()
            except llm.LLMError as exc:
                st.session_state.last_error = str(exc)
                st.error(f"실제 API 호출에 실패했습니다: {exc}")
                st.info("사이드바에서 설정을 확인하거나, '데모 모드'로 바꾸면 같은 흐름을 그대로 진행할 수 있습니다.")
                if st.button("데모 모드로 진행"):
                    st.session_state["analysis"] = llm.run_analysis(files, mode="demo")
                    st.session_state.questions = st.session_state.analysis["questions"]
                    st.session_state.reviews = {}
                    st.rerun()
        if st.button("← Step 1로"):
            go(1)
        return

    if analysis.get("mode") == "demo":
        st.info("데모(시뮬레이션) 결과입니다 — 실제 API를 호출하지 않고, 업로드한 자료의 문장·수치를 근거로 구성했습니다.")
    else:
        st.success(f"실제 LLM({st.session_state.get('model', '')}) 응답으로 생성했습니다.")

    tab1, tab2 = st.tabs(["① 1차 초안", f"② 질문 {len(st.session_state.questions)}개"])
    with tab1:
        st.markdown(analysis["draft_markdown"])
    with tab2:
        questions = st.session_state.questions
        must = sum(1 for q in questions if q.get("priority") == "필수")
        st.caption(f"질문 {len(questions)}개 · 필수 {must}개 · 권장 {len(questions) - must}개 "
                   "— 자료의 빈칸·모호한 지점 수에 따라 개수가 달라집니다.")
        for q in questions:
            pri = q.get("priority", "권장")
            cls = "must" if pri == "필수" else "rec"
            st.markdown(
                f'<div class="qcard">'
                f'<span class="pill {cls}">{html.escape(pri)}</span>'
                f'<span class="pill cat">{_meta(q.get("category", ""))}</span>'
                f'<div class="q">Q{q["id"]}. {_meta(q["question"])}</div>'
                f'<div class="why">왜 막히는가: {_meta(q.get("why", ""))}</div>'
                f'<div class="ev">근거: {_meta(q.get("evidence", ""))}</div></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    c1, c2 = st.columns([1, 1])
    if c1.button("← Step 1로"):
        go(1)
    if c2.button("Step 3로 — 인계자 답변 입력·검수 →", type="primary"):
        go(3)


# ----------------------------------------------------------------------------
# Step 3 — 인계자 Q&A
# ----------------------------------------------------------------------------

def _meta(text: str) -> str:
    """질문 배경·근거를 Markdown 해석 없이 그대로 보여 준다."""
    clean = (text or "").replace("\n", " ").replace("~", "∼")
    return html.escape(clean)


def _collect_answers(questions) -> dict:
    return {q["id"]: st.session_state.get(f"ans_{q['id']}", "") for q in questions}


def render_qa():
    st.subheader("Step 3. 인계자 Q&A + AI 답변 검수")
    questions = st.session_state.questions
    files = st.session_state.files
    if not questions:
        st.warning("Step 2 분석을 먼저 실행해 주세요.")
        if st.button("← Step 2로"): go(2)
        return

    mode = st.session_state.get("mode", "demo")
    reviews = st.session_state.reviews or {}
    must = sum(1 for q in questions if q.get("priority") == "필수")

    st.write(
        "자료에 없어서 후임자가 막힐 지점입니다. **짧아도 실제 경험을 적어 주세요.** "
        "답변을 쓴 뒤 **🔍 AI로 답변 검수받기**를 누르면, 너무 짧거나 모호한 답변과 "
        "자료의 숫자·기한과 어긋나는 답변을 먼저 짚어 드립니다."
    )
    st.caption(f"질문 {len(questions)}개 · 필수 {must}개 · 권장 {len(questions) - must}개")

    if reviews:
        enough = sum(1 for r in reviews.values() if r.get("verdict") == "충분함")
        need = len(reviews) - enough
        c1, c2, c3 = st.columns(3)
        c1.metric("충분함 ✅", f"{enough}건", border=True)
        c2.metric("보완 필요 ⚠️", f"{need}건", border=True)
        c3.metric("검수한 문항", f"{len(reviews)}/{len(questions)}", border=True)
        if st.session_state.review_summary:
            st.caption(f"AI 총평: {st.session_state.review_summary}")
        if need:
            st.warning(f"{need}건은 보완이 필요합니다. 아래 코멘트대로 고친 뒤 다시 검수받거나, 그대로 최종본으로 넘어갈 수 있습니다.")
        else:
            st.success("모든 답변이 충분합니다. 최종 인수인계서를 만들 수 있습니다.")

    if "extra_widget" not in st.session_state:
        st.session_state["extra_widget"] = st.session_state.extra_notes

    for q in questions:
        qid = q["id"]
        key = f"ans_{qid}"
        if key not in st.session_state:
            st.session_state[key] = st.session_state.answers.get(qid, "")
        rv = reviews.get(qid) or reviews.get(str(qid)) or {}
        pri = q.get("priority", "권장")
        pill_cls = "must" if pri == "필수" else "rec"
        with st.container(border=True):
            st.markdown(
                f'<span class="pill {pill_cls}">{html.escape(pri)}</span>'
                f'<span class="pill cat">{html.escape(q.get("category", ""))}</span>'
                f'<span style="font-weight:600">Q{qid}. {_meta(q["question"])}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="qmeta">왜 막히는가: {_meta(q.get("why", ""))} · 근거: {_meta(q.get("evidence", ""))}</div>',
                unsafe_allow_html=True,
            )
            st.text_area(
                "답변", key=key, height=86, label_visibility="collapsed",
                placeholder="예: 담당은 김소희 팀원, 부재 시 고승준 팀장에게 먼저 연락합니다. 매월 말일까지 노션에 증빙을 올립니다.",
            )
            if rv:
                ok = rv.get("verdict") == "충분함"
                cls = "ok" if ok else "warn"
                badge = "충분함 ✅" if ok else "보완 필요 ⚠️"
                rq = (f'<div class="rq">↳ 다시 적을 질문: {html.escape(rv.get("requestion", ""))}</div>'
                      if rv.get("requestion") else "")
                st.markdown(
                    f'<div class="rev {cls}"><div class="rt">{badge}</div>'
                    f'{html.escape(rv.get("comment", ""))}{rq}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("#### ✍️ 기타 추가로 남길 인계 메모")
    st.caption("질문에 담기지 않은 판단 기준·요령·비상 연락처를 인계자가 직접 적는 칸입니다. 최종본에 그대로 들어갑니다.")
    st.text_area(
        "기타 인계 메모", key="extra_widget", height=110, label_visibility="collapsed",
        placeholder="예: 팀 노션의 '회의록 아카이브'에 결정 사항이 모여 있습니다. 급할 때는 고승준 팀장에게 먼저 전화하세요.",
    )

    st.divider()
    if reviews:
        c1, c2, c3 = st.columns([1.2, 1, 1])
        make_final = c1.button("최종 인수인계서 생성 →", type="primary", use_container_width=True)
        redo = c2.button("🔍 다시 검수받기", use_container_width=True)
        skip = False
        st.caption("답변을 고쳤다면 '다시 검수받기'로 확인한 뒤 생성하세요.")
    else:
        c1, c2 = st.columns([1.2, 1])
        redo = c1.button("🔍 AI로 답변 검수받기", type="primary", use_container_width=True)
        skip = c2.button("검수 없이 바로 생성", type="tertiary", use_container_width=True)
        make_final = False
        st.caption("기본 흐름은 '검수 → 최종본 생성'입니다. 검수를 건너뛰려면 오른쪽 보조 버튼을 쓰세요.")

    if st.button("← Step 2로 (질문 다시 보기)"):
        go(2)

    if redo:
        answers = _collect_answers(questions)
        extra = st.session_state.get("extra_widget", "")
        st.session_state.answers = answers
        st.session_state.extra_notes = extra
        try:
            with st.spinner("답변을 검수하는 중…"):
                res = llm.run_review(
                    files, questions, answers, mode=mode,
                    base_url=st.session_state.get("base_url", ""),
                    api_key=st.session_state.get("api_key", ""),
                    model=st.session_state.get("model", ""),
                    extra_notes=extra,
                )
            st.session_state.reviews = res["reviews"]
            st.session_state.review_summary = res.get("summary")
            st.session_state.final_md = None
            st.rerun()
        except llm.LLMError as exc:
            st.error(f"답변 검수 호출에 실패했습니다: {exc}")
            st.info("사이드바에서 설정을 확인하거나, '데모 모드'로 바꾸면 같은 검수를 그대로 체험할 수 있습니다.")
            if st.button("데모 모드로 검수"):
                res = llm.run_review(files, questions, answers, mode="demo", extra_notes=extra)
                st.session_state.reviews = res["reviews"]
                st.session_state.review_summary = res.get("summary")
                st.rerun()

    if make_final or skip:
        st.session_state.answers = _collect_answers(questions)
        st.session_state.extra_notes = st.session_state.get("extra_widget", "")
        st.session_state.final_md = None
        go(4)


# ----------------------------------------------------------------------------
# Step 4 — 최종본 + 내보내기
# ----------------------------------------------------------------------------

def render_final():
    st.subheader("Step 4. 최종 인수인계서 완성 & 내보내기")
    files = st.session_state.files
    questions = st.session_state.questions
    answers = st.session_state.answers
    if not questions:
        st.warning("Step 2·3을 먼저 진행해 주세요.")
        if st.button("← Step 2로"): go(2)
        return

    answered = sum(1 for v in answers.values() if str(v).strip())
    reviews = st.session_state.reviews or {}
    enough = sum(1 for r in reviews.values() if r.get("verdict") == "충분함")
    need = len(reviews) - enough
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("답변 완료", f"{answered}/{len(questions)}", border=True)
    c2.metric("검수 충분", f"{enough}/{len(questions)}", border=True)
    c3.metric("검수 보완", f"{need}건", border=True)
    c4.metric("자료", f"{len(files)}개", border=True)

    if answered == 0:
        st.warning("답변이 하나도 없습니다. 그래도 만들 수는 있지만, 빈 항목은 '미확인'으로 남습니다.")
    elif not reviews:
        st.info("AI 답변 검수를 거치지 않았습니다. Step 3에서 검수받으면 모호한 답변을 먼저 걸러낼 수 있습니다.")
    elif need:
        st.warning(f"검수에서 보완 필요 {need}건이 남아 있습니다. 지금 생성해도 되지만, 그 항목은 확인이 필요한 상태로 표시됩니다.")

    mode = "demo" if st.session_state.get("analysis", {}).get("mode") == "demo" else "api"
    if st.session_state.final_md is None:
        if st.button("최종 인수인계서 만들기", type="primary"):
            try:
                with st.spinner("4개 파트를 채우는 중…"):
                    st.session_state.final_md = llm.run_final(
                        files, st.session_state.analysis["draft_markdown"], questions, answers,
                        mode=mode,
                        base_url=st.session_state.get("base_url", ""),
                        api_key=st.session_state.get("api_key", ""),
                        model=st.session_state.get("model", ""),
                        extra_notes=st.session_state.get("extra_notes", ""),
                        reviews=reviews,
                    )
                st.rerun()
            except llm.LLMError as exc:
                st.error(f"최종본 생성 실패: {exc}")
        if st.button("← Step 3로"):
            go(3)
        return

    final_md = st.session_state.final_md
    st.success("4개 파트가 모두 포함된 최종 인수인계서입니다.")

    title_match = re.search(r"^#\s+(.+)$", final_md, re.M)
    title = title_match.group(1).strip() if title_match else "업무 인수인계서"

    tab1, tab2 = st.tabs(["📄 미리보기", "⬇️ 다운로드"])
    with tab1:
        st.markdown(final_md)
    with tab2:
        st.write("Markdown은 그대로 편집·공유하고, Word는 사내 결재·게시용으로 쓰세요.")
        col1, col2 = st.columns(2)
        col1.download_button(
            "Markdown(.md) 내려받기",
            data=exporters.markdown_bytes(final_md),
            file_name=exporters.safe_filename(title, "md"),
            mime="text/markdown",
            use_container_width=True,
        )
        try:
            docx = exporters.docx_bytes(final_md, title=title)
            col2.download_button(
                "Word(.docx) 내려받기",
                data=docx,
                file_name=exporters.safe_filename(title, "docx"),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        except Exception as exc:      # pragma: no cover
            col2.error(f"Word 변환 실패: {exc}")

        st.divider()
        st.download_button(
            "전체 결과(.md, 초안+최종 합본)",
            data=exporters.markdown_bytes(
                st.session_state.analysis["draft_markdown"] + "\n\n---\n\n" + final_md
            ),
            file_name=exporters.safe_filename(title, "md").replace(".md", "_초안포함.md"),
            mime="text/markdown",
        )

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("← Step 3로 (답변 수정·재검수)"):
        st.session_state.final_md = None
        go(3)
    if c2.button("새 자료로 다시 시작"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()


# ----------------------------------------------------------------------------
# 라우팅
# ----------------------------------------------------------------------------

# 사이드바 위젯 값은 사이드바 안에서 이미 session_state에 저장된다.

step = st.session_state.step
if step == 1:
    render_upload()
elif step == 2:
    render_analysis()
elif step == 3:
    render_qa()
else:
    render_final()

st.divider()
st.caption("경영사무 AI 인수인계 도우미 · Streamlit · HWP/HWPX/PDF/DOCX/XLSX/CSV/TXT/ZIP 지원")
