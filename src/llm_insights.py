"""LLM 기반 AI 인사이트 생성 (배포 버전).

Streamlit secrets에서 API 키를 로드.
"""

import logging

import pandas as pd
import anthropic
import streamlit as st

logger = logging.getLogger(__name__)

_client = None

MODEL = "claude-sonnet-4-20250514"


def _format_hourly_ft(hft: dict) -> str:
    if not hft:
        return "- 데이터 없음"
    return "\n".join(f"  {int(h):02d}시: {hft[h]:,}명" for h in sorted(hft.keys()))


SYSTEM_PROMPT = """당신은 백화점 화장실 이용 데이터를 분석하는 시설 운영 전문가입니다.

[현장 배경]
- 장소: 더현대서울 (여의도 소재 대형 백화점)
- 위치: 1층 게이트 바로 옆 화장실
- 특성: 게이트 인접으로 통행량이 매우 많지만, 오히려 화장실 이용률은 상대적으로 낮은 편
- 데이터: BLE 센서(S-Ward) 기반 실시간 감지, 3중 필터(신호세기+통과율+최소체류)로 실제 이용자만 추출
- 활용 목적: 화장실 청소 주기 최적화, 시설 관리 효율화, 혼잡 시간대 대응

규칙:
- 한국어로 답변
- 각 섹션은 1~2문장으로 간결하게
- 데이터에 근거한 구체적 수치 인용
- 청소 타이밍, 용품 보충, 인력 배치 등 시설 관리 관점의 실질적 제안 포함
- 마크다운 서식 사용 금지, 순수 텍스트로만"""


def _get_client():
    global _client
    if _client is None:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if not key:
            return None
        _client = anthropic.Anthropic(api_key=key)
    return _client


def generate_insights(metrics: dict) -> dict[str, str]:
    client = _get_client()
    if client is None:
        return {}

    # 요일/날씨
    wi_data = metrics.get("weather_info", {})
    day_str = f" ({wi_data['day_kr']}요일)" if wi_data.get("day_kr") else ""
    weather_str = ""
    if wi_data.get("weather") and wi_data["weather"] != "Unknown":
        weather_str = f", 날씨: {wi_data['weather']}"
        if wi_data.get("temp_max") is not None:
            weather_str += f" ({wi_data['temp_min']:.0f}~{wi_data['temp_max']:.0f}°C)"

    # 다른 날짜 비교 맥락
    other_days = metrics.get("all_dates_summary", [])
    other_block = ""
    if other_days:
        lines = []
        for od in other_days:
            owi = od.get("weather_info", {})
            od_day = f"({owi.get('day_kr', '?')})" if owi.get("day_kr") else ""
            od_w = f" {owi.get('weather', '')}" if owi.get("weather") and owi["weather"] != "Unknown" else ""
            lines.append(
                f"  {od['date']}{od_day}{od_w}: "
                f"유동인구 {od.get('foot_traffic', 0):,}명, "
                f"이용자 {od.get('total_users', 0):,}명, "
                f"이용률 {od.get('usage_rate', 0):.1f}%"
            )
        other_block = "\n[다른 날짜 비교 데이터]\n" + "\n".join(lines) + "\n※ 위 데이터를 참고하여 오늘의 수치가 평소 대비 높은지 낮은지, 요일/날씨 영향이 있는지 분석해주세요.\n"

    prompt = f"""아래는 백화점 1F 화장실 하루 이용 데이터 분석 결과입니다.

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
※ 유동인구는 화장실 앞을 지나간 전체 통행량으로, 백화점 방문객의 시간대별 유입 패턴을 반영합니다.
  화장실 이용 피크와 유동인구 피크의 차이, 이용률 변화 등을 분석해주세요.
{other_block}
아래 5개 섹션에 대해 각각 1~2문장의 인사이트를 작성하세요.
반드시 아래 형식을 지켜주세요:

[SUMMARY]
전체 요약 인사이트

[GENDER]
남녀 비교 인사이트

[PEAK]
피크 시간대 및 혼잡도 인사이트

[DURATION]
체류 시간 분포 인사이트

[OCCUPANCY]
동시 이용자 및 용량 인사이트"""

    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_sections(resp.content[0].text)
    except Exception as e:
        logger.warning(f"LLM API error: {e}")
        return {}


def generate_comparison_insights(daily_summaries: list[dict]) -> str:
    client = _get_client()
    if client is None:
        return ""

    lines, ft_lines = [], []
    for d in daily_summaries:
        wi_data = d.get("weather_info", {})
        day_tag = f"({wi_data.get('day_kr', '?')})" if wi_data.get("day_kr") else ""
        w_tag = f" {wi_data.get('weather', '')}" if wi_data.get("weather") and wi_data["weather"] != "Unknown" else ""
        t_tag = f" {wi_data.get('temp_min', 0):.0f}~{wi_data.get('temp_max', 0):.0f}°C" if wi_data.get("temp_max") is not None else ""
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
            top_str = ", ".join([f"{int(h):02d}시 {c:,}명" for h, c in top3])
            ft_lines.append(f"  {d.get('date', '?')} 유동인구 Top3: {top_str}")

    prompt = f"""아래는 백화점 1F 화장실의 여러 날짜 이용 데이터 요약입니다.

[날짜별 이용 현황]
{chr(10).join(lines)}

[시간대별 유동인구 피크 (백화점 방문 패턴)]
{chr(10).join(ft_lines) if ft_lines else '  데이터 없음'}
※ 유동인구는 화장실 앞 통행량으로, 백화점 전체 방문객 유입 패턴을 반영합니다.
  날짜별 유동인구 차이와 화장실 이용률 변화의 상관관계도 분석해주세요.

날짜 간 비교 인사이트를 3~5문장으로 작성하세요.
- 날짜별 유동인구 패턴 차이와 화장실 이용률 변화
- 요일/시간대별 백화점 방문 트렌드
- 유동인구 대비 이용률 개선 또는 운영 최적화 제안
마크다운 서식 사용 금지, 순수 텍스트로만."""

    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logger.warning(f"LLM API error: {e}")
        return ""


def build_metrics_for_llm(
    visits, occupancy, foot_traffic, date_str, time_range,
    hourly_foot_traffic=None,
    weather_info: dict | None = None,
    all_dates_summary: list[dict] | None = None,
) -> dict:
    m = {
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
    m["male_avg_min"] = v_male["duration_min"].mean() if not v_male.empty else 0
    m["male_median_min"] = v_male["duration_min"].median() if not v_male.empty else 0
    m["female_avg_min"] = v_female["duration_min"].mean() if not v_female.empty else 0
    m["female_median_min"] = v_female["duration_min"].median() if not v_female.empty else 0
    m["male_ast_hours"] = v_male["duration_sec"].sum() / 3600 if not v_male.empty else 0
    m["female_ast_hours"] = v_female["duration_sec"].sum() / 3600 if not v_female.empty else 0

    hourly = visits.groupby("start_hour").size()
    if not hourly.empty:
        m["peak_hour"] = int(hourly.idxmax())
        m["peak_hour_visits"] = int(hourly.max())

    if not occupancy.empty:
        for r, key in [("남자화장실", "male_peak_occ"), ("여자화장실", "female_peak_occ")]:
            o = occupancy[occupancy["restroom"] == r]
            m[key] = int(o["device_count"].max()) if not o.empty else 0

    for rv, prefix in [(v_male, "male"), (v_female, "female")]:
        if rv.empty: continue
        n = len(rv)
        m[f"{prefix}_1_2min_pct"] = ((rv["duration_min"] >= 1) & (rv["duration_min"] < 2)).sum() / n * 100
        m[f"{prefix}_3_5min_pct"] = ((rv["duration_min"] >= 3) & (rv["duration_min"] < 5)).sum() / n * 100
        m[f"{prefix}_5plus_pct"] = (rv["duration_min"] >= 5).sum() / n * 100

    if "device_type" in visits.columns:
        dev = visits["device_type"].value_counts()
        dev_total = dev.sum()
        if dev_total > 0:
            m["iphone_pct"] = dev.get("iPhone", 0) / dev_total * 100
            m["android_pct"] = dev.get("Android", 0) / dev_total * 100

    return m


def _parse_sections(text: str) -> dict[str, str]:
    tag_map = {
        "[SUMMARY]": "summary", "[GENDER]": "gender", "[PEAK]": "peak",
        "[DURATION]": "duration", "[OCCUPANCY]": "occupancy",
    }
    sections = {}
    current_key = None
    current_lines = []
    for line in text.strip().split("\n"):
        line = line.strip()
        matched = False
        for tag, key in tag_map.items():
            if line.startswith(tag):
                if current_key:
                    sections[current_key] = " ".join(current_lines).strip()
                current_key, current_lines = key, []
                matched = True
                break
        if not matched and line and current_key:
            current_lines.append(line)
    if current_key:
        sections[current_key] = " ".join(current_lines).strip()
    return sections
