"""LLM 기반 AI 인사이트 생성 (배포 버전).

Streamlit secrets에서 API 키를 로드.

변경 이력:
  2026-04-10  고도화 v2
    - _build_daily_prompt() 분리 (streaming/non-streaming 공용)
    - _parse_sections() 분리
    - [CLEANING] 섹션 추가 (6번째 섹션)
    - _compute_weekday_block(): 같은 요일 기대값 vs 오늘 이상도
    - _compute_hourly_usage_block(): 시간대별 이용률 (유동인구 대비)
    - generate_insights_streaming(): 실시간 streaming 출력
    - generate_comparison_insights(): 단일 텍스트 → 4섹션 dict 반환
    - generate_comparison_insights_streaming(): streaming 버전
    - build_metrics_for_llm(): hourly_visits + hourly_usage_rate 추가
"""

import logging
from datetime import datetime as _dt

import pandas as pd
import anthropic
import streamlit as st

logger = logging.getLogger(__name__)

_client = None
_DAY_KR = ["월", "화", "수", "목", "금", "토", "일"]

MODEL = "claude-sonnet-4-20250514"


def _get_client() -> anthropic.Anthropic | None:
    global _client
    if _client is None:
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            key = ""
        if not key:
            return None
        _client = anthropic.Anthropic(api_key=key)
    return _client


# ─── 내부 헬퍼 ───────────────────────────────────────────────

def _format_hourly_ft(hft: dict) -> str:
    if not hft:
        return "- 데이터 없음"
    return "\n".join(f"  {int(h):02d}시: {hft[h]:,}명" for h in sorted(hft.keys()))


def _compute_weekday_block(today_date: str, enriched_other_days: list[dict]) -> str:
    """같은 요일 기대값 vs 오늘 이상도 블록."""
    if not today_date or not enriched_other_days:
        return ""
    try:
        today_dt = _dt.strptime(today_date, "%Y-%m-%d")
        today_dow = today_dt.weekday()
    except Exception:
        return ""

    same_dow = [od for od in enriched_other_days if od.get("dow") == today_dow]
    if not same_dow:
        return ""

    n = len(same_dow)
    avg = {
        "foot_traffic":   sum(o.get("foot_traffic", 0)   for o in same_dow) / n,
        "total_users":    sum(o.get("total_users", 0)    for o in same_dow) / n,
        "usage_rate":     sum(o.get("usage_rate", 0)     for o in same_dow) / n,
        "male_avg_min":   sum(o.get("male_avg_min", 0)   for o in same_dow) / n,
        "female_avg_min": sum(o.get("female_avg_min", 0) for o in same_dow) / n,
    }
    dow_kr = _DAY_KR[today_dow]
    return (
        f"[같은 요일({dow_kr}) 기대값 — {n}일 평균]\n"
        f"  유동인구 기대: {avg['foot_traffic']:,.0f}명\n"
        f"  이용자 기대: {avg['total_users']:.0f}명 (이용률 {avg['usage_rate']:.1f}%)\n"
        f"  남자 평균 체류 기대: {avg['male_avg_min']:.1f}분\n"
        f"  여자 평균 체류 기대: {avg['female_avg_min']:.1f}분"
    )


def _compute_hourly_usage_block(hourly_usage_rate: dict) -> str:
    """시간대별 이용률 (유동인구 대비) 블록."""
    if not hourly_usage_rate:
        return ""
    lines = ["[시간대별 이용률 (유동인구 대비 %) — 높은 시간대 = 실제 방문객 비중 높음]"]
    for h in sorted(hourly_usage_rate.keys()):
        lines.append(f"  {int(h):02d}시: {hourly_usage_rate[h]:.1f}%")
    return "\n".join(lines)


def _enrich_other_days(other_days: list[dict]) -> list[dict]:
    enriched = []
    for od in other_days:
        try:
            dt = _dt.strptime(od.get("date", ""), "%Y-%m-%d")
            dow = dt.weekday()
        except Exception:
            dow = -1
        enriched.append({**od, "dow": dow})
    return enriched


def _fmt_detail(od: dict) -> str:
    w = od.get("weather_info", {})
    w_str = f" {w.get('weather', '')}" if w.get("weather") and w["weather"] != "Unknown" else ""
    return (
        f"  {od['date']}({od.get('day_kr', '?')}){w_str}: "
        f"유동 {od.get('foot_traffic', 0):,} / 이용 {od.get('total_users', 0):,}명 "
        f"(남 {od.get('male_users', 0)} 여 {od.get('female_users', 0)}) "
        f"이용률 {od.get('usage_rate', 0):.1f}%, "
        f"남 평균 {od.get('male_avg_min', 0):.1f}분 / 여 평균 {od.get('female_avg_min', 0):.1f}분, "
        f"피크 {od.get('peak_hour', 0)}시({od.get('peak_visits', 0)}회)"
    )


def _avg_group_str(group: list[dict], label: str) -> str | None:
    if not group:
        return None
    n = len(group)
    return (
        f"  {label} 평균 ({n}일): "
        f"유동 {sum(o.get('foot_traffic',0) for o in group)/n:,.0f} / "
        f"이용 {sum(o.get('total_users',0) for o in group)/n:.0f}명, "
        f"이용률 {sum(o.get('usage_rate',0) for o in group)/n:.1f}%, "
        f"남 체류 {sum(o.get('male_avg_min',0) for o in group)/n:.1f}분 / "
        f"여 체류 {sum(o.get('female_avg_min',0) for o in group)/n:.1f}분"
    )


SYSTEM_PROMPT = """당신은 백화점 화장실 이용 데이터를 분석하는 시설 운영 전문가입니다.

[현장 배경]
- 장소: 더현대서울 (여의도 소재 대형 백화점, 10:30 개점)
- 위치: 1층 게이트 바로 옆 화장실
- 특성: 게이트 인접으로 통행량이 매우 많지만, 화장실 이용률은 상대적으로 낮은 편
- 데이터: BLE 센서(S-Ward) 기반 실시간 감지, 3중 필터(신호세기+통과율+최소체류)로 실제 이용자만 추출
- 활용 목적: 화장실 청소 주기 최적화, 시설 관리 효율화, 혼잡 시간대 대응

[여의도 입지 특성 — 분석 시 반드시 고려]
- 여의도는 한국 최대 금융/오피스 밀집 지역으로, 주변 직장인 유입이 큼
- 주중: 직장인 행동이 시간대에 묶여 패턴이 뚜렷함
  · 오전 개점~12시: 쇼핑 목적 방문객 입장
  · 12시경: 오전 입장객 식당가 이동 + 직장인 아직 점심 중 → 1F 통행 일시 감소 (딥 현상)
  · 13시~: 점심 마친 직장인 유입 + 오후 방문객 재상승
  · 17~18시: 퇴근 후 유입 피크
- 주말: 가족/커플 등 자유 시간대 방문 → 부드러운 곡선, 주중보다 1.5~2배 유동인구

[청소/운영 권장 판단 기준]
- 청소 타이밍: 피크 30분 전이 이상적 (피크 중 청소 → 혼잡 가중)
- 용품 보충: AST가 높은 시간대 직후
- 인력 배치: 동시 이용자 피크 시간대
- 이용률이 높은 시간대(유동인구 대비 이용자 비율 높음) = 실제 수요 집중 → 청소 우선순위 상향

규칙:
- 한국어로 답변
- 각 섹션은 2~3문장으로 간결하게
- 데이터에 근거한 구체적 수치 인용 (시간, 인원수, 비율 포함)
- 주중/주말, 요일별, 날씨별 차이를 입지 특성과 연결하여 해석
- 마크다운 서식 사용 금지, 순수 텍스트로만"""


# ─── 프롬프트 빌더 ───────────────────────────────────────────

def _build_daily_prompt(metrics: dict) -> str:
    """일별 분석 프롬프트 구성 (streaming/non-streaming 공용)."""
    wi = metrics.get("weather_info", {})
    day_str = f" ({wi['day_kr']}요일)" if wi.get("day_kr") else ""
    weather_str = ""
    if wi.get("weather") and wi["weather"] != "Unknown":
        weather_str = f", 날씨: {wi['weather']}"
        if wi.get("temp_max") is not None:
            weather_str += f" ({wi.get('temp_min', 0):.0f}~{wi['temp_max']:.0f}°C)"

    other_days = metrics.get("all_dates_summary", [])
    other_block = ""

    if other_days:
        today_date = metrics.get("date", "")
        today_dow = -1
        today_dt = None
        try:
            today_dt = _dt.strptime(today_date, "%Y-%m-%d")
            today_dow = today_dt.weekday()
        except Exception:
            pass

        enriched = _enrich_other_days(other_days)
        same_weekday, yesterday = [], None
        weekday_group, weekend_group = [], []

        for od in enriched:
            od_dow = od.get("dow", -1)
            owi = od.get("weather_info", {})
            od["day_kr"] = owi.get("day_kr", _DAY_KR[od_dow] if od_dow >= 0 else "?")

            if od_dow == today_dow:
                same_weekday.append(od)
            if today_dt:
                try:
                    od_dt = _dt.strptime(od.get("date", ""), "%Y-%m-%d")
                    if (today_dt - od_dt).days == 1:
                        yesterday = od
                except Exception:
                    pass
            if od_dow < 5:
                weekday_group.append(od)
            elif od_dow >= 5:
                weekend_group.append(od)

        blocks = []
        if yesterday:
            blocks.append(f"[전일 비교 (어제)]\n{_fmt_detail(yesterday)}")
        if same_weekday:
            dow_kr = _DAY_KR[today_dow] if today_dow >= 0 else "?"
            sw_lines = "\n".join(_fmt_detail(od) for od in same_weekday)
            blocks.append(f"[같은 요일({dow_kr}요일) 비교]\n{sw_lines}")

        avg_lines = []
        wd = _avg_group_str(weekday_group, "주중(월~금)")
        we = _avg_group_str(weekend_group, "주말(토~일)")
        if wd:
            avg_lines.append(wd)
        if we:
            avg_lines.append(we)
        if avg_lines:
            today_type = "주말" if today_dow >= 5 else "주중"
            blocks.append(f"[주중/주말 평균 비교 — 오늘은 {today_type}]\n" + "\n".join(avg_lines))

        all_lines = "\n".join(_fmt_detail(od) for od in enriched)
        blocks.append(f"[전체 기록 날짜]\n{all_lines}")

        other_block = (
            "\n" + "\n\n".join(blocks) +
            "\n\n※ 위 비교 데이터를 적극 활용하여 분석하세요:\n"
            "  - 전일 대비 변화 (증감, 원인 추정)\n"
            "  - 같은 요일 대비 추세 (일관성 또는 변화)\n"
            "  - 주중/주말 패턴 차이와 오늘의 위치\n"
            "  - 날씨가 이용 패턴에 미치는 영향\n"
        )

        wb = _compute_weekday_block(metrics.get("date", ""), enriched)
        if wb:
            other_block = "\n" + wb + "\n" + other_block

    hourly_usage_rate = metrics.get("hourly_usage_rate", {})
    hourly_usage_block = ""
    if hourly_usage_rate:
        hourly_usage_block = "\n" + _compute_hourly_usage_block(hourly_usage_rate) + "\n"

    hourly_visits = metrics.get("hourly_visits", {})
    hourly_visits_block = ""
    if hourly_visits:
        lines = ["[시간대별 방문 (남자/여자)]"]
        for h in sorted(hourly_visits.keys()):
            hv = hourly_visits[h]
            lines.append(f"  {int(h):02d}시: 남 {hv.get('남자화장실', 0)} / 여 {hv.get('여자화장실', 0)}")
        hourly_visits_block = "\n" + "\n".join(lines) + "\n"

    return f"""아래는 백화점 1F 화장실 하루 이용 데이터 분석 결과입니다.

[기본 정보]
- 날짜: {metrics.get('date', '?')}{day_str}{weather_str}
- 분석 시간대: {metrics.get('time_range', '7~23시')}
- 유동인구: {metrics.get('foot_traffic', 0):,}명
- 총 이용자: {metrics.get('total_users', 0):,}명 (이용률 {metrics.get('usage_rate', 0):.1f}%)

[남녀별]
- 남자: {metrics.get('male_users', 0):,}명, 평균 {metrics.get('male_avg_min', 0):.1f}분, 중앙값 {metrics.get('male_median_min', 0):.1f}분
- 여자: {metrics.get('female_users', 0):,}명, 평균 {metrics.get('female_avg_min', 0):.1f}분, 중앙값 {metrics.get('female_median_min', 0):.1f}분
- 남녀 비율: 남 {metrics.get('male_pct', 0):.0f}% / 여 {metrics.get('female_pct', 0):.0f}%

[피크]
- 피크 시간대: {metrics.get('peak_hour', 0)}시 ({metrics.get('peak_hour_visits', 0)}회)
- 남자 동시 최대: {metrics.get('male_peak_occ', 0)}명
- 여자 동시 최대: {metrics.get('female_peak_occ', 0)}명

[체류 시간 분포]
- 남자 1~2분: {metrics.get('male_1_2min_pct', 0):.0f}%, 3~5분: {metrics.get('male_3_5min_pct', 0):.0f}%, 5분+: {metrics.get('male_5plus_pct', 0):.0f}%
- 여자 1~2분: {metrics.get('female_1_2min_pct', 0):.0f}%, 3~5분: {metrics.get('female_3_5min_pct', 0):.0f}%, 5분+: {metrics.get('female_5plus_pct', 0):.0f}%

[디바이스]
- iPhone: {metrics.get('iphone_pct', 0):.0f}% / Android: {metrics.get('android_pct', 0):.0f}%

[누적 체류 시간 (AST)]
- 남자 AST: {metrics.get('male_ast_hours', 0):.1f}시간
- 여자 AST: {metrics.get('female_ast_hours', 0):.1f}시간

[시간대별 유동인구 (백화점 방문 패턴)]
{_format_hourly_ft(metrics.get('hourly_foot_traffic', {}))}
※ 유동인구는 화장실 앞 통행량으로 백화점 방문객의 시간대별 유입 패턴을 반영합니다.
{hourly_visits_block}{hourly_usage_block}{other_block}
아래 6개 섹션에 대해 각각 2~3문장의 인사이트를 작성하세요.
비교 데이터가 있으면 반드시 활용하여 "전일 대비", "같은 요일 대비", "주중/주말 평균 대비" 등 상대적 분석을 포함하세요.
단순 수치 나열이 아닌, 변화의 원인(요일/날씨/시간대)을 추정하세요.
반드시 아래 형식을 지켜주세요:

[SUMMARY]
전체 요약 + 과거 데이터 대비 오늘의 특이점

[GENDER]
남녀 비교 + 과거 대비 비율/체류 변화

[PEAK]
피크 시간대 + 과거 동일 요일/전일과의 피크 비교

[DURATION]
체류 시간 분포 + 과거 대비 변화 추세

[OCCUPANCY]
동시 이용자 + 용량 여유/부족 판단 및 과거 대비

[CLEANING]
오늘 데이터 기반 청소/용품 보충/인력 배치 구체적 권장 시간과 이유.
시간대별 이용률(유동인구 대비)이 높은 시간대를 반드시 반영하고, 피크 30분 전 청소 원칙을 적용하세요."""


def _build_comparison_prompt(daily_summaries: list[dict]) -> str:
    lines = []
    ft_lines = []
    for d in daily_summaries:
        wi = d.get("weather_info", {})
        day_tag = f"({wi.get('day_kr', '?')})" if wi.get("day_kr") else ""
        w_tag = f" {wi.get('weather', '')}" if wi.get("weather") and wi["weather"] != "Unknown" else ""
        t_tag = (
            f" {wi.get('temp_min', 0):.0f}~{wi.get('temp_max', 0):.0f}°C"
            if wi.get("temp_max") is not None else ""
        )
        lines.append(
            f"- {d.get('date', '?')}{day_tag}{w_tag}{t_tag}: "
            f"유동인구 {d.get('foot_traffic', 0):,}명, "
            f"남 {d.get('male_users', 0)}명 / 여 {d.get('female_users', 0)}명, "
            f"총 {d.get('total_users', 0)}명 (이용률 {d.get('usage_rate', 0):.1f}%), "
            f"남 평균 {d.get('male_avg_min', 0):.1f}분 / 여 평균 {d.get('female_avg_min', 0):.1f}분, "
            f"피크 {d.get('peak_hour', 0)}시({d.get('peak_hour_visits', 0)}회)"
        )
        hft = d.get("hourly_foot_traffic", {})
        if hft:
            top3 = sorted(hft.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join(f"{int(h):02d}시 {c:,}명" for h, c in top3)
            ft_lines.append(f"  {d.get('date', '?')} 유동인구 Top3: {top_str}")

    data_block = "\n".join(lines)
    ft_block = "\n".join(ft_lines) if ft_lines else "  데이터 없음"

    return f"""아래는 백화점 1F 화장실의 여러 날짜 이용 데이터 요약입니다.

[날짜별 이용 현황]
{data_block}

[시간대별 유동인구 피크 (백화점 방문 패턴)]
{ft_block}
※ 유동인구는 화장실 앞 통행량으로, 백화점 방문객 유입 패턴을 반영합니다.

아래 4개 섹션에 대해 각각 2~3문장으로 작성하세요.
반드시 아래 형식을 지켜주세요:

[TREND]
날짜별 이용자 수/이용률 추세 — 증가/감소 방향과 원인 추정

[PATTERN]
요일·시간대·날씨 패턴 — 주중/주말 차이, 특정 요일 특성

[ANOMALY]
평균 대비 특이 날짜 — 유독 높거나 낮은 날과 추정 원인

[ACTION]
데이터 기반 청소 주기 최적화 및 운영 개선 제안 (구체적 시간/요일 명시)"""


# ─── 섹션 파서 ───────────────────────────────────────────────

_TAG_MAP = {
    "SUMMARY": "summary", "GENDER": "gender", "PEAK": "peak",
    "DURATION": "duration", "OCCUPANCY": "occupancy", "CLEANING": "cleaning",
    "TREND": "trend", "PATTERN": "pattern", "ANOMALY": "anomaly", "ACTION": "action",
}


def _parse_sections(text: str, keys: list[str]) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_key = None
    for line in text.strip().split("\n"):
        stripped = line.strip()
        matched = False
        for tag, key in _TAG_MAP.items():
            if stripped.startswith(f"[{tag}]"):
                current_key = key
                sections.setdefault(current_key, [])
                matched = True
                break
        if not matched and stripped and current_key:
            sections[current_key].append(stripped)
    return {k: " ".join(sections.get(k, [])).strip() for k in keys}


# ─── 일별 분석 인사이트 ──────────────────────────────────────

def generate_insights(metrics: dict) -> dict[str, str]:
    """6섹션 AI 인사이트 (non-streaming)."""
    client = _get_client()
    if client is None:
        return {}
    prompt = _build_daily_prompt(metrics)
    try:
        response = client.messages.create(
            model=MODEL, max_tokens=1800, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
    except Exception as e:
        logger.warning(f"LLM API error: {e}")
        return {}
    return _parse_sections(text, ["summary", "gender", "peak", "duration", "occupancy", "cleaning"])


def generate_insights_streaming(metrics: dict, on_chunk) -> dict[str, str]:
    """스트리밍 버전 일별 분석 인사이트.

    on_chunk(text_so_far: str) — 청크마다 호출됨.
    Returns: 완료 후 파싱된 섹션 dict
    """
    client = _get_client()
    if client is None:
        return {}
    prompt = _build_daily_prompt(metrics)
    full_text = ""
    try:
        with client.messages.stream(
            model=MODEL, max_tokens=1800, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for chunk in stream.text_stream:
                full_text += chunk
                on_chunk(full_text)
    except Exception as e:
        logger.warning(f"LLM streaming error: {e}")
        return {}
    return _parse_sections(full_text, ["summary", "gender", "peak", "duration", "occupancy", "cleaning"])


# ─── 날짜 비교 인사이트 ─────────────────────────────────────

def generate_comparison_insights(daily_summaries: list[dict]) -> dict[str, str]:
    """날짜 비교 4섹션 dict 반환."""
    client = _get_client()
    if client is None:
        return {}
    prompt = _build_comparison_prompt(daily_summaries)
    try:
        response = client.messages.create(
            model=MODEL, max_tokens=800, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
    except Exception as e:
        logger.warning(f"LLM API error: {e}")
        return {}
    return _parse_sections(text, ["trend", "pattern", "anomaly", "action"])


def generate_comparison_insights_streaming(daily_summaries: list[dict], on_chunk) -> dict[str, str]:
    """스트리밍 버전 날짜 비교 인사이트."""
    client = _get_client()
    if client is None:
        return {}
    prompt = _build_comparison_prompt(daily_summaries)
    full_text = ""
    try:
        with client.messages.stream(
            model=MODEL, max_tokens=800, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for chunk in stream.text_stream:
                full_text += chunk
                on_chunk(full_text)
    except Exception as e:
        logger.warning(f"LLM streaming error: {e}")
        return {}
    return _parse_sections(full_text, ["trend", "pattern", "anomaly", "action"])


# ─── 메트릭 빌더 ─────────────────────────────────────────────

def build_metrics_for_llm(
    visits,
    occupancy,
    foot_traffic,
    date_str: str,
    time_range: tuple,
    hourly_foot_traffic=None,
    weather_info: dict | None = None,
    all_dates_summary: list[dict] | None = None,
) -> dict:
    """대시보드 데이터 → LLM용 메트릭 dict 구성."""
    m: dict = {
        "date": date_str,
        "time_range": f"{time_range[0]}~{time_range[1]}시",
        "foot_traffic": foot_traffic.get("total_unique", 0) if isinstance(foot_traffic, dict) else 0,
    }
    if weather_info:
        m["weather_info"] = weather_info
    if all_dates_summary:
        m["all_dates_summary"] = all_dates_summary

    if hourly_foot_traffic is not None:
        if isinstance(hourly_foot_traffic, pd.DataFrame) and not hourly_foot_traffic.empty:
            hft = hourly_foot_traffic.set_index("hour")["unique_count"].to_dict()
            m["hourly_foot_traffic"] = {int(k): int(v) for k, v in hft.items()}
        elif isinstance(hourly_foot_traffic, dict):
            m["hourly_foot_traffic"] = hourly_foot_traffic

    if visits.empty:
        return m

    v_male = visits[visits["restroom"] == "남자화장실"]
    v_female = visits[visits["restroom"] == "여자화장실"]
    total = visits["mac_address"].nunique()

    m["total_users"] = total
    m["usage_rate"] = (total / m["foot_traffic"] * 100) if m["foot_traffic"] > 0 else 0
    m["male_users"] = v_male["mac_address"].nunique() if not v_male.empty else 0
    m["female_users"] = v_female["mac_address"].nunique() if not v_female.empty else 0
    m["male_pct"] = (m["male_users"] / total * 100) if total > 0 else 0
    m["female_pct"] = (m["female_users"] / total * 100) if total > 0 else 0

    m["male_avg_min"] = float(v_male["duration_min"].mean()) if not v_male.empty else 0.0
    m["male_median_min"] = float(v_male["duration_min"].median()) if not v_male.empty else 0.0
    m["female_avg_min"] = float(v_female["duration_min"].mean()) if not v_female.empty else 0.0
    m["female_median_min"] = float(v_female["duration_min"].median()) if not v_female.empty else 0.0
    m["male_ast_hours"] = float(v_male["duration_sec"].sum() / 3600) if not v_male.empty else 0.0
    m["female_ast_hours"] = float(v_female["duration_sec"].sum() / 3600) if not v_female.empty else 0.0

    hourly = visits.groupby("start_hour").size()
    if not hourly.empty:
        m["peak_hour"] = int(hourly.idxmax())
        m["peak_hour_visits"] = int(hourly.max())

    if not occupancy.empty:
        for r, key in [("남자화장실", "male_peak_occ"), ("여자화장실", "female_peak_occ")]:
            o = occupancy[occupancy["restroom"] == r]
            m[key] = int(o["device_count"].max()) if not o.empty else 0

    for rv, prefix in [(v_male, "male"), (v_female, "female")]:
        if rv.empty:
            continue
        n = len(rv)
        m[f"{prefix}_1_2min_pct"] = float(((rv["duration_min"] >= 1) & (rv["duration_min"] < 2)).sum() / n * 100)
        m[f"{prefix}_3_5min_pct"] = float(((rv["duration_min"] >= 3) & (rv["duration_min"] < 5)).sum() / n * 100)
        m[f"{prefix}_5plus_pct"] = float((rv["duration_min"] >= 5).sum() / n * 100)

    if "device_type" in visits.columns:
        dev = visits["device_type"].value_counts()
        dev_total = dev.sum()
        if dev_total > 0:
            m["iphone_pct"] = float(dev.get("iPhone", 0) / dev_total * 100)
            m["android_pct"] = float(dev.get("Android", 0) / dev_total * 100)

    # ── 신규: 시간대별 방문 (남녀) ──
    if "start_hour" in visits.columns:
        hourly_visits_dict: dict[int, dict] = {}
        for hour, group in visits.groupby("start_hour"):
            hourly_visits_dict[int(hour)] = {
                "남자화장실": int((group["restroom"] == "남자화장실").sum()),
                "여자화장실": int((group["restroom"] == "여자화장실").sum()),
            }
        m["hourly_visits"] = hourly_visits_dict

    # ── 신규: 시간대별 이용률 (유동인구 대비) ──
    hft_dict = m.get("hourly_foot_traffic", {})
    hourly_v_dict = m.get("hourly_visits", {})
    if hft_dict and hourly_v_dict:
        hourly_usage: dict[int, float] = {}
        for h, ft_count in hft_dict.items():
            hv = hourly_v_dict.get(h, {})
            total_v = hv.get("남자화장실", 0) + hv.get("여자화장실", 0)
            hourly_usage[h] = round(total_v / ft_count * 100, 1) if ft_count > 0 else 0.0
        m["hourly_usage_rate"] = hourly_usage

    return m
