"""더현대서울 화장실 이용 모니터링 대시보드 (배포 버전)."""

import json

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import get_available_dates, load_cached_data, load_hourly_foot_traffic, SWARD_MAP
from src.metrics import (
    compute_summary,
    compute_hourly_stats,
    compute_duration_distribution,
    compute_peak_analysis,
    compute_daily_comparison,
)
from src.llm_insights import (
    generate_insights,
    generate_comparison_insights,
    build_metrics_for_llm,
)

# ─── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(page_title="화장실 이용 모니터링", page_icon="🚻", layout="wide")

# ─── 상수 ────────────────────────────────────────────────────
COLORS = {"남자화장실": "#4A90D9", "여자화장실": "#E85D75"}
MALE_FILL = "rgba(74,144,217,0.1)"
FEMALE_FILL = "rgba(232,93,117,0.1)"
BG = "#0E1117"
GRID = "#1a2035"
PCFG = {"displayModeBar": False}

# ─── 스타일 ──────────────────────────────────────────────────
st.markdown("""<style>
.mc{background:linear-gradient(135deg,#1a1f36,#252b48);border-radius:12px;padding:20px;text-align:center;border:1px solid #2d3456}
.mv{font-size:2.2rem;font-weight:700;color:#FFF;margin:4px 0}
.ml{font-size:.85rem;color:#8892b0;text-transform:uppercase;letter-spacing:.5px}
.ms{font-size:.75rem;color:#5a6785;margin-top:4px}
.sh{font-size:1.1rem;font-weight:600;color:#ccd6f6;margin:24px 0 12px;padding-bottom:8px;border-bottom:1px solid #2d3456}
.ai{background:linear-gradient(135deg,#1a2332,#1e2d3d);border-left:3px solid #64ffda;border-radius:8px;padding:14px 18px;margin:10px 0;font-size:.88rem;color:#b8c9e0;line-height:1.6}
</style>""", unsafe_allow_html=True)


def mc(label: str, value: str, sub: str = ""):
    s = f'<div class="ms">{sub}</div>' if sub else ""
    st.markdown(f'<div class="mc"><div class="ml">{label}</div><div class="mv">{value}</div>{s}</div>', unsafe_allow_html=True)


def sh(text: str):
    st.markdown(f'<div class="sh">{text}</div>', unsafe_allow_html=True)


def _lay(title: str = "", height: int = 400) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14, color="#ccd6f6")),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color="#8892b0", size=11), height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )


def _ti2hm(ti: int) -> str:
    s = ti * 10
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


# ─── 인증 ────────────────────────────────────────────────────
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.markdown("## 🚻 화장실 이용 모니터링")
    st.caption("더현대서울 1F")
    pw = st.text_input("비밀번호를 입력하세요", type="password")
    if pw:
        if pw == st.secrets.get("password", ""):
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")
    st.stop()

# ─── 사이드바 ───────────────────────────────────────────────
with st.sidebar:
    st.title("🚻 화장실 모니터링")
    st.caption("더현대서울 1F")
    dates = get_available_dates()
    if not dates:
        st.error("데이터가 없습니다.")
        st.stop()
    selected_date = st.selectbox("날짜 선택", dates, index=len(dates) - 1)
    st.divider()
    view_mode = st.radio("분석 모드", ["일별 분석", "날짜 비교"], index=0)
    st.divider()
    st.markdown("**필터 설정**")
    time_range = st.slider("시간대 필터", 0, 24, (7, 23), help="분석할 시간대 범위")
    st.divider()
    st.markdown('<div style="color:#5a6785;font-size:.75rem">S-Ward: 210002D5 (남자) / 210003C6 (여자)<br>10초 단위 BLE 감지 기반 분석</div>', unsafe_allow_html=True)


# ─── 데이터 로드 + 시간 필터 (캐싱) ──────────────────────────
@st.cache_data(show_spinner=False)
def _load(date_str: str):
    return load_cached_data(date_str)


@st.cache_data(show_spinner=False)
def _filter(date_str: str, t0: int, t1: int):
    visits, occ, ft = _load(date_str)
    ti_s, ti_e = t0 * 360, t1 * 360
    if not visits.empty:
        visits = visits[(visits["start_ti"] >= ti_s) & (visits["start_ti"] < ti_e)]
    if not occ.empty:
        occ = occ[(occ["time_bin"] >= ti_s) & (occ["time_bin"] < ti_e)]
    return visits, occ, ft


@st.cache_data(show_spinner=False, ttl=3600)
def _ai(metrics_json: str) -> dict:
    return generate_insights(json.loads(metrics_json))


# ═══════════════════════════════════════════════════════════
#  일별 분석 모드
# ═══════════════════════════════════════════════════════════
if view_mode == "일별 분석":
    visits, occupancy, ft = _filter(selected_date, time_range[0], time_range[1])

    st.markdown(f"## {selected_date} 화장실 이용 현황")
    if visits.empty:
        st.warning("해당 날짜/시간대에 방문 기록이 없습니다.")
        st.stop()

    # 남녀 분리 (재사용)
    v_male = visits[visits["restroom"] == "남자화장실"]
    v_female = visits[visits["restroom"] == "여자화장실"]

    # ─── 요약 메트릭 ────────────────────────────────────
    summary = compute_summary(visits, occupancy)
    c = st.columns(5)
    with c[0]: mc("총 방문 횟수", f"{summary['total_visits']:,}", "감지된 세션 수")
    with c[1]: mc("고유 디바이스", f"{summary['unique_devices']:,}", "Unique MAC 수")
    with c[2]: mc("평균 체류 시간", f"{summary['avg_duration_min']}분", f"중앙값 {summary['median_duration_min']}분")
    with c[3]: mc("피크 시간대", f"{summary['peak_hour']:02d}시", f"{summary['peak_hour_visits']}회 방문")
    with c[4]: mc("누적 체류 시간", f"{summary['total_ast_hours']}h", "AST 합산")
    st.markdown("")

    # ─── 남녀별 요약 ────────────────────────────────────
    col_m, col_f = st.columns(2)
    for col, label, rv, icon in [
        (col_m, "남자화장실", v_male, "🚹"),
        (col_f, "여자화장실", v_female, "🚺"),
    ]:
        with col:
            color = COLORS[label]
            st.markdown(f'<div style="font-size:1rem;font-weight:600;color:{color};margin-bottom:8px">{icon} {label}</div>', unsafe_allow_html=True)
            if rv.empty:
                st.info("방문 기록 없음")
                continue
            m = st.columns(4)
            with m[0]: mc("방문", f"{len(rv):,}")
            with m[1]: mc("디바이스", f"{rv['mac_address'].nunique():,}")
            with m[2]: mc("평균 체류", f"{rv['duration_min'].mean():.1f}분")
            with m[3]: mc("AST", f"{rv['duration_sec'].sum()/3600:.1f}h")

    # ─── 유동인구 vs 이용자 + 디바이스 비율 ───────────────
    sh("유동인구 대비 화장실 이용률 & 디바이스 비율")
    col_flow, col_device = st.columns(2)

    with col_flow:
        total_passerby = ft["total_unique"]
        total_users = visits["mac_address"].nunique()
        non_users = max(0, total_passerby - total_users)
        usage_rate = (total_users / total_passerby * 100) if total_passerby > 0 else 0
        fc = st.columns(3)
        with fc[0]: mc("유동인구", f"{total_passerby:,}", "화장실 앞 통행자")
        with fc[1]: mc("이용자", f"{total_users:,}", "1분 이상 체류")
        with fc[2]: mc("이용률", f"{usage_rate:.1f}%", f"비이용자 {non_users:,}명")

        flow_rows = []
        for label, rv in [("남자화장실", v_male), ("여자화장실", v_female)]:
            users = rv["mac_address"].nunique() if not rv.empty else 0
            rate = (users / total_passerby * 100) if total_passerby > 0 else 0
            flow_rows.append({
                "화장실": label, "이용자": f"{users:,}", "이용률": f"{rate:.1f}%",
                "평균 체류": f"{rv['duration_min'].mean():.1f}분" if not rv.empty else "0분",
                "총 방문": f"{len(rv):,}",
            })
        st.dataframe(pd.DataFrame(flow_rows), hide_index=True, use_container_width=True)

    with col_device:
        if "device_type" in visits.columns:
            dev_counts = visits["device_type"].value_counts()
            iph = int(dev_counts.get("iPhone", 0))
            andr = int(dev_counts.get("Android", 0))
        else:
            iph, andr = 0, 0
        total_dev = iph + andr
        fig = go.Figure(go.Pie(
            labels=["iPhone", "Android"], values=[iph, andr],
            marker=dict(colors=["#007AFF", "#3DDC84"]),
            textinfo="label+percent", textfont=dict(size=13), hole=0.45,
        ))
        lay = _lay("iPhone vs Android (이용자)", 280)
        lay.pop("xaxis", None); lay.pop("yaxis", None)
        fig.update_layout(**lay, showlegend=False,
            annotations=[dict(text=f"{total_dev:,}", x=0.5, y=0.5, font_size=20, font_color="#ccd6f6", showarrow=False)])
        st.plotly_chart(fig, use_container_width=True, config=PCFG)

    # ─── 시간대별 방문 추이 ─────────────────────────────
    sh("시간대별 방문 추이")
    hourly = compute_hourly_stats(visits)
    if not hourly.empty:
        fig = px.bar(hourly, x="hour_label", y="visit_count", color="restroom",
                     barmode="group", color_discrete_map=COLORS,
                     labels={"hour_label": "시간", "visit_count": "방문 횟수", "restroom": ""})
        fig.update_layout(**_lay("시간대별 방문 횟수"))
        st.plotly_chart(fig, use_container_width=True, config=PCFG)

    # ─── 동시 이용자 수 ───────────────────────────────
    sh("시간대별 동시 이용자 수 (1분 단위)")
    if not occupancy.empty:
        fig = go.Figure()
        fills = {"남자화장실": MALE_FILL, "여자화장실": FEMALE_FILL}
        for r in ["남자화장실", "여자화장실"]:
            o = occupancy[occupancy["restroom"] == r].sort_values("time_bin")
            if o.empty: continue
            tl = o["time_bin"].map(_ti2hm)
            fig.add_trace(go.Scatter(
                x=o["time_bin"], y=o["device_count"], mode="lines", name=r,
                line=dict(color=COLORS[r], width=1.5), fill="tozeroy", fillcolor=fills[r],
                customdata=tl, hovertemplate="%{customdata} | %{y}명<extra>%{fullData.name}</extra>"))
        tv = [h * 360 for h in range(time_range[0], time_range[1] + 1)]
        tt = [f"{h:02d}:00" for h in range(time_range[0], time_range[1] + 1)]
        fig.update_layout(**_lay("동시 이용자 수 추이 (1분 단위)", 350))
        fig.update_xaxes(tickmode="array", tickvals=tv, ticktext=tt)
        st.plotly_chart(fig, use_container_width=True, config=PCFG)

    # ─── 체류 시간 분포 ─────────────────────────────────
    sh("체류 시간 분포")
    dist = compute_duration_distribution(visits)
    if not dist.empty:
        dist = dist.copy()
        dist["pct"] = 0.0
        for r in ["남자화장실", "여자화장실"]:
            mask = dist["restroom"] == r
            t = dist.loc[mask, "count"].sum()
            if t > 0:
                dist.loc[mask, "pct"] = (dist.loc[mask, "count"] / t * 100).round(1)
        dist["label"] = dist.apply(lambda x: f"{x['count']}명 ({x['pct']:.0f}%)", axis=1)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(dist, x="duration_bin", y="count", color="restroom",
                         barmode="group", color_discrete_map=COLORS, text="label",
                         labels={"duration_bin": "체류 시간", "count": "방문 수", "restroom": ""})
            fig.update_traces(textposition="outside", textfont_size=10)
            fig.update_layout(**_lay("체류 시간 분포", 400))
            st.plotly_chart(fig, use_container_width=True, config=PCFG)
        with c2:
            rows = []
            for label, rv in [("남자화장실", v_male), ("여자화장실", v_female)]:
                if not rv.empty:
                    rows.append({"화장실": label, "평균": f"{rv['duration_min'].mean():.1f}분",
                                 "중앙값": f"{rv['duration_min'].median():.1f}분",
                                 "최대": f"{rv['duration_min'].max():.1f}분",
                                 "표준편차": f"{rv['duration_min'].std():.1f}분"})
            if rows:
                st.markdown("**체류 시간 통계**")
                st.dataframe(pd.DataFrame(rows), hide_index=True)

    # ─── 30분 단위 피크 분석 ────────────────────────────
    sh("30분 단위 피크 분석")
    peak = compute_peak_analysis(visits)
    if not peak.empty:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("방문 횟수", "평균 체류 시간 (분)"))
        for r in ["남자화장실", "여자화장실"]:
            pr = peak[peak["restroom"] == r]
            if pr.empty: continue
            fig.add_trace(go.Bar(x=pr["half_hour_label"], y=pr["visit_count"], name=r, marker_color=COLORS[r], showlegend=True), row=1, col=1)
            fig.add_trace(go.Bar(x=pr["half_hour_label"], y=pr["avg_duration_min"], name=r, marker_color=COLORS[r], showlegend=False), row=1, col=2)
        lay = _lay(height=380); lay.pop("xaxis", None); lay.pop("yaxis", None)
        fig.update_layout(**lay)
        fig.update_xaxes(gridcolor=GRID); fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig, use_container_width=True, config=PCFG)

    # ─── AST 히트맵 ─────────────────────────────────────
    sh("시간대별 누적 체류 시간 (AST)")
    if not hourly.empty:
        hm = hourly.pivot_table(index="restroom", columns="hour_label", values="total_ast_min", fill_value=0)
        fig = px.imshow(hm.values, x=hm.columns.tolist(), y=hm.index.tolist(),
                        color_continuous_scale="YlOrRd", labels=dict(x="시간", y="화장실", color="AST (분)"), aspect="auto")
        fig.update_layout(**_lay("AST 히트맵 (분)", 250))
        st.plotly_chart(fig, use_container_width=True, config=PCFG)

    # ─── 누적 체류 시간 추이 (벡터화) ─────────────────────
    sh("누적 체류 시간 추이 (AST)")
    if not visits.empty:
        ast_df = visits[["start_ti", "restroom", "duration_min"]].copy()
        ast_df["time_bin"] = (ast_df["start_ti"] // 6) * 6
        ast_by = ast_df.groupby(["time_bin", "restroom"])["duration_min"].sum().reset_index().sort_values("time_bin")

        fig = go.Figure()
        for r in ["남자화장실", "여자화장실"]:
            rd = ast_by[ast_by["restroom"] == r]
            if rd.empty: continue
            cum = rd["duration_min"].cumsum()
            tl = rd["time_bin"].map(_ti2hm)
            fig.add_trace(go.Scatter(
                x=rd["time_bin"], y=cum, mode="lines", name=r,
                line=dict(color=COLORS[r], width=2.5),
                customdata=tl, hovertemplate="%{customdata} | %{y:.1f}분<extra>%{fullData.name}</extra>"))
        tv = [h * 360 for h in range(time_range[0], time_range[1] + 1)]
        tt = [f"{h:02d}:00" for h in range(time_range[0], time_range[1] + 1)]
        fig.update_layout(**_lay("시간에 따른 누적 체류 시간 (분)", 380))
        fig.update_xaxes(tickmode="array", tickvals=tv, ticktext=tt)
        fig.update_yaxes(title_text="누적 AST (분)")
        st.plotly_chart(fig, use_container_width=True, config=PCFG)

    # ─── 분류 신뢰도 ──────────────────────────────────────
    sh("남녀 분류 신뢰도")
    if "win_rate" in visits.columns:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(visits, x="win_rate", color="restroom", nbins=20,
                               color_discrete_map=COLORS, barmode="overlay", opacity=0.7,
                               labels={"win_rate": "H2H 승률", "count": "방문 수", "restroom": ""})
            fig.update_layout(**_lay("분류 승률 분포", 300))
            st.plotly_chart(fig, use_container_width=True, config=PCFG)
        with c2:
            if "classify_method" in visits.columns:
                mc_data = visits["classify_method"].value_counts()
                ml = {"h2h": "Head-to-Head", "count_only": "감지 횟수 비교", "single": "한쪽만 감지"}
                fig = go.Figure(go.Pie(
                    labels=[ml.get(m, m) for m in mc_data.index], values=mc_data.values,
                    hole=0.4, marker=dict(colors=["#64ffda", "#ffd166", "#a78bfa"]),
                    textinfo="label+percent", textfont=dict(size=11)))
                lay = _lay("분류 방법 비율", 300); lay.pop("xaxis", None); lay.pop("yaxis", None)
                fig.update_layout(**lay, showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config=PCFG)
            conf = []
            for label, rv in [("남자화장실", v_male), ("여자화장실", v_female)]:
                if not rv.empty:
                    conf.append({"화장실": label, "평균 승률": f"{rv['win_rate'].mean():.0%}",
                                 "승률>=70%": f"{(rv['win_rate']>=0.7).sum():,} ({(rv['win_rate']>=0.7).mean()*100:.0f}%)",
                                 "방문 수": f"{len(rv):,}"})
            if conf:
                st.dataframe(pd.DataFrame(conf), hide_index=True, use_container_width=True)

    # ─── 상세 데이터 ────────────────────────────────────
    with st.expander("방문 상세 데이터"):
        dcols = [c for c in ["restroom", "start_time", "end_time", "duration_min", "device_type", "win_rate", "classify_method", "mac_address"] if c in visits.columns]
        st.dataframe(visits[dcols].sort_values("start_time"), use_container_width=True, height=400)

    # ─── AI Analysis ──────────────────────────────────────
    st.markdown("---")
    sh("🤖 AI Analysis")
    st.caption("Claude AI가 오늘의 화장실 이용 데이터를 종합 분석합니다.")

    if st.button("AI 분석 실행", type="primary", use_container_width=True):
        with st.spinner("AI가 데이터를 분석하고 있습니다..."):
            _hft = load_hourly_foot_traffic(selected_date)
            _hft = _hft[(_hft["hour"] >= time_range[0]) & (_hft["hour"] < time_range[1])] if not _hft.empty else _hft
            llm_m = build_metrics_for_llm(visits, occupancy, ft, selected_date, time_range, hourly_foot_traffic=_hft)
            ai = _ai(json.dumps(llm_m, default=str))

        if ai:
            titles = {
                "summary": "📊 종합 요약",
                "gender": "🚻 남녀 이용 패턴",
                "peak": "⏰ 피크 시간대 & 혼잡도",
                "duration": "⏱️ 체류 시간 분석",
                "occupancy": "👥 동시 이용 & 용량",
            }
            for key, title in titles.items():
                text = ai.get(key, "")
                if text:
                    st.markdown(
                        f'<div class="ai"><span style="color:#64ffda;font-weight:600">{title}</span><br>{text}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.warning("AI 분석을 실행할 수 없습니다. API 키를 확인해주세요.")

# ═══════════════════════════════════════════════════════════
#  날짜 비교 모드
# ═══════════════════════════════════════════════════════════
else:
    st.markdown("## 날짜별 비교 분석")
    all_visits, all_occ, all_ft = {}, {}, {}
    for date in dates:
        v, o, f = _filter(date, time_range[0], time_range[1])
        all_visits[date] = v
        all_occ[date] = o
        all_ft[date] = f

    comparison = compute_daily_comparison(all_visits)
    if comparison.empty:
        st.warning("비교할 데이터가 없습니다.")
        st.stop()

    sh("날짜별 요약")
    st.dataframe(comparison.rename(columns={
        "date": "날짜", "restroom": "화장실", "total_visits": "총 방문",
        "unique_devices": "고유 디바이스", "avg_duration_min": "평균 체류(분)", "total_ast_hours": "AST(시간)",
    }), use_container_width=True, hide_index=True)

    sh("날짜별 방문 추이")
    fig = go.Figure()
    dc = px.colors.qualitative.Set2
    for i, (date, v) in enumerate(all_visits.items()):
        if v.empty: continue
        h = v.groupby("start_hour").size().reset_index(name="count")
        fig.add_trace(go.Scatter(
            x=h["start_hour"].apply(lambda x: f"{int(x):02d}:00"), y=h["count"],
            mode="lines+markers", name=date, line=dict(color=dc[i % len(dc)], width=2)))
    fig.update_layout(**_lay("날짜별 시간대 방문 비교"))
    st.plotly_chart(fig, use_container_width=True, config=PCFG)

    sh("날짜별 남녀 방문 비율")
    rd = []
    for date, v in all_visits.items():
        if v.empty: continue
        for r in ["남자화장실", "여자화장실"]:
            rd.append({"date": date, "restroom": r, "count": len(v[v["restroom"] == r])})
    if rd:
        fig = px.bar(pd.DataFrame(rd), x="date", y="count", color="restroom",
                     barmode="group", color_discrete_map=COLORS,
                     labels={"date": "날짜", "count": "방문 횟수", "restroom": ""})
        fig.update_layout(**_lay("남녀 방문 비교", 350))
        st.plotly_chart(fig, use_container_width=True, config=PCFG)

    sh("날짜별 체류 시간 분포")
    av = pd.concat([v.assign(date=d) for d, v in all_visits.items() if not v.empty], ignore_index=True)
    if not av.empty:
        fig = px.box(av, x="date", y="duration_min", color="restroom", color_discrete_map=COLORS,
                     labels={"date": "날짜", "duration_min": "체류 시간(분)", "restroom": ""})
        fig.update_layout(**_lay("체류 시간 분포 비교", 400))
        st.plotly_chart(fig, use_container_width=True, config=PCFG)

    # ─── 날짜별 시간대 유동인구 비교 ──────────────────────
    sh("날짜별 시간대 유동인구 비교")
    st.caption("시간대별 화장실 앞 통행량 — 마케팅·운영 시간대 분석에 활용 가능")
    fig_ft = go.Figure()
    for i, date in enumerate(dates):
        hft = load_hourly_foot_traffic(date)
        if hft.empty:
            continue
        # 시간대 필터 적용
        hft = hft[(hft["hour"] >= time_range[0]) & (hft["hour"] < time_range[1])]
        fig_ft.add_trace(go.Scatter(
            x=hft["hour"].apply(lambda h: f"{int(h):02d}:00"),
            y=hft["unique_count"],
            mode="lines+markers", name=date,
            line=dict(color=dc[i % len(dc)], width=2),
        ))
    fig_ft.update_layout(**_lay("시간대별 유동인구 (Unique MAC)", 400))
    fig_ft.update_yaxes(title_text="고유 디바이스 수")
    st.plotly_chart(fig_ft, use_container_width=True, config=PCFG)

    # ─── AI Analysis (비교 모드) ──────────────────────────
    st.markdown("---")
    sh("🤖 AI Analysis")
    st.caption("Claude AI가 날짜 간 이용 패턴 변화를 분석합니다.")

    if st.button("AI 비교 분석 실행", type="primary", use_container_width=True):
        with st.spinner("AI가 데이터를 분석하고 있습니다..."):
            daily_summaries = []
            for date, v in all_visits.items():
                if v.empty: continue
                _hft = load_hourly_foot_traffic(date)
                _hft = _hft[(_hft["hour"] >= time_range[0]) & (_hft["hour"] < time_range[1])] if not _hft.empty else _hft
                daily_summaries.append(
                    build_metrics_for_llm(v, all_occ.get(date, pd.DataFrame()), all_ft.get(date, {}), date, time_range, hourly_foot_traffic=_hft)
                )
            result = generate_comparison_insights(daily_summaries) if daily_summaries else ""

        if result:
            st.markdown(
                f'<div class="ai"><span style="color:#64ffda;font-weight:600">📊 날짜 비교 종합 분석</span><br>{result}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("AI 분석을 실행할 수 없습니다. API 키를 확인해주세요.")
