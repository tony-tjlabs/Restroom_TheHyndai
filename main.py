"""더현대서울 화장실 이용 모니터링 대시보드 (배포 버전)."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import get_available_dates, load_cached_data, SWARD_MAP
from src.metrics import (
    compute_summary,
    compute_hourly_stats,
    compute_duration_distribution,
    compute_peak_analysis,
    compute_daily_comparison,
)

# ─── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(
    page_title="화장실 이용 모니터링",
    page_icon="🚻",
    layout="wide",
)

# ─── 색상 팔레트 ────────────────────────────────────────────
COLORS = {
    "남자화장실": "#4A90D9",
    "여자화장실": "#E85D75",
}
BG_COLOR = "#0E1117"
CARD_BG = "#1E2130"

# ─── 스타일 ─────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1f36 0%, #252b48 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3456;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #5a6785;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ccd6f6;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d3456;
    }
</style>
""", unsafe_allow_html=True)


def render_metric_card(label: str, value: str, sub: str = ""):
    """메트릭 카드 렌더링."""
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def ti_to_hhmm(ti: int) -> str:
    """time_index → 'HH:MM' 문자열 변환."""
    total_sec = ti * 10
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    return f"{h:02d}:{m:02d}"


def make_plotly_layout(title: str = "", height: int = 400) -> dict:
    """공통 Plotly 레이아웃."""
    return dict(
        title=dict(text=title, font=dict(size=14, color="#ccd6f6")),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#8892b0", size=11),
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor="#1a2035", zerolinecolor="#1a2035"),
        yaxis=dict(gridcolor="#1a2035", zerolinecolor="#1a2035"),
    )


# ─── 인증 ────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("## 🚻 화장실 이용 모니터링")
    st.caption("더현대서울 1F")
    pw = st.text_input("비밀번호를 입력하세요", type="password")
    if pw:
        if pw == st.secrets.get("password", "wonderful2$"):
            st.session_state.authenticated = True
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
    view_mode = st.radio(
        "분석 모드",
        ["일별 분석", "날짜 비교"],
        index=0,
    )

    st.divider()
    st.markdown("**필터 설정**")
    time_range = st.slider(
        "시간대 필터",
        min_value=0, max_value=24, value=(7, 23),
        help="분석할 시간대 범위",
    )

    st.divider()
    st.markdown(
        '<div style="color:#5a6785; font-size:0.75rem;">'
        "S-Ward: 210002D5 (남자) / 210003C6 (여자)<br>"
        "10초 단위 BLE 감지 기반 분석"
        "</div>",
        unsafe_allow_html=True,
    )


# ─── 데이터 로드 (캐시 기반) ──────────────────────────────────
@st.cache_data(show_spinner="데이터 로딩 중...")
def load_and_process(date_str: str):
    visits, occupancy, foot_traffic = load_cached_data(date_str)
    return visits, occupancy, foot_traffic


# ═══════════════════════════════════════════════════════════
#  일별 분석 모드
# ═══════════════════════════════════════════════════════════
if view_mode == "일별 분석":
    visits, occupancy, foot_traffic = load_and_process(selected_date)

    # 시간대 필터 적용
    ti_start = time_range[0] * 360  # 1시간 = 360 time_index
    ti_end = time_range[1] * 360

    # foot_traffic은 전체 캐시 값 사용 (시간 필터 미적용 — 캐시에 raw 없음)
    foot_traffic_filtered = foot_traffic

    if not visits.empty:
        visits = visits[(visits["start_ti"] >= ti_start) & (visits["start_ti"] < ti_end)]
    if not occupancy.empty:
        occupancy = occupancy[
            (occupancy["time_bin"] >= ti_start) & (occupancy["time_bin"] < ti_end)
        ]

    # ─── 헤더 ───────────────────────────────────────────
    st.markdown(f"## {selected_date} 화장실 이용 현황")

    if visits.empty:
        st.warning("해당 날짜/시간대에 방문 기록이 없습니다.")
        st.stop()

    # ─── 요약 메트릭 ────────────────────────────────────
    summary = compute_summary(visits, occupancy)

    cols = st.columns(5)
    with cols[0]:
        render_metric_card("총 방문 횟수", f"{summary['total_visits']:,}", "감지된 세션 수")
    with cols[1]:
        render_metric_card("고유 디바이스", f"{summary['unique_devices']:,}", "Unique MAC 수")
    with cols[2]:
        render_metric_card(
            "평균 체류 시간",
            f"{summary['avg_duration_min']}분",
            f"중앙값 {summary['median_duration_min']}분",
        )
    with cols[3]:
        render_metric_card(
            "피크 시간대",
            f"{summary['peak_hour']:02d}시",
            f"{summary['peak_hour_visits']}회 방문",
        )
    with cols[4]:
        render_metric_card(
            "누적 체류 시간",
            f"{summary['total_ast_hours']}h",
            "AST 합산",
        )

    st.markdown("")

    # ─── 남녀별 요약 ────────────────────────────────────
    col_m, col_f = st.columns(2)
    for col, restroom in [(col_m, "남자화장실"), (col_f, "여자화장실")]:
        rv = visits[visits["restroom"] == restroom]
        with col:
            color = COLORS[restroom]
            st.markdown(
                f'<div style="font-size:1rem; font-weight:600; color:{color}; '
                f'margin-bottom:8px;">{"🚹" if "남" in restroom else "🚺"} {restroom}</div>',
                unsafe_allow_html=True,
            )
            if rv.empty:
                st.info("방문 기록 없음")
                continue
            mc = st.columns(4)
            with mc[0]:
                render_metric_card("방문", f"{len(rv):,}", "")
            with mc[1]:
                render_metric_card("디바이스", f"{rv['mac_address'].nunique():,}", "")
            with mc[2]:
                render_metric_card("평균 체류", f"{rv['duration_min'].mean():.1f}분", "")
            with mc[3]:
                render_metric_card("AST", f"{rv['duration_sec'].sum()/3600:.1f}h", "")

    # ─── 유동인구 vs 이용자 + 디바이스 비율 ───────────────
    st.markdown('<div class="section-header">유동인구 대비 화장실 이용률 & 디바이스 비율</div>', unsafe_allow_html=True)

    ft = foot_traffic_filtered
    col_flow, col_device = st.columns(2)

    with col_flow:
        # 유동인구 vs 이용자 비교
        total_passerby = ft["total_unique"]
        total_users = visits["mac_address"].nunique() if not visits.empty else 0
        non_users = max(0, total_passerby - total_users)
        usage_rate = (total_users / total_passerby * 100) if total_passerby > 0 else 0

        fc = st.columns(3)
        with fc[0]:
            render_metric_card("유동인구", f"{total_passerby:,}", "화장실 앞 통행자")
        with fc[1]:
            render_metric_card("이용자", f"{total_users:,}", "1분 이상 체류 (H2H 분류)")
        with fc[2]:
            render_metric_card("이용률", f"{usage_rate:.1f}%", f"비이용자 {non_users:,}명")

        # 남녀별 이용자 (H2H 분류 기반)
        flow_rows = []
        for restroom in ["남자화장실", "여자화장실"]:
            rv = visits[visits["restroom"] == restroom] if not visits.empty else pd.DataFrame()
            users = rv["mac_address"].nunique() if not rv.empty else 0
            rate = (users / total_passerby * 100) if total_passerby > 0 else 0
            avg_dur = rv["duration_min"].mean() if not rv.empty else 0
            flow_rows.append({
                "화장실": restroom,
                "이용자": f"{users:,}",
                "이용률": f"{rate:.1f}%",
                "평균 체류": f"{avg_dur:.1f}분",
                "총 방문": f"{len(rv):,}",
            })
        st.dataframe(pd.DataFrame(flow_rows), hide_index=True, use_container_width=True)

    with col_device:
        # iPhone vs Android 비율 (전체 감지 기준)
        iphone_count = ft["by_type"].get("iPhone", 0)
        android_count = ft["by_type"].get("Android", 0)
        total_dev = iphone_count + android_count

        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=["iPhone", "Android"],
            values=[iphone_count, android_count],
            marker=dict(colors=["#007AFF", "#3DDC84"]),
            textinfo="label+percent",
            textfont=dict(size=13),
            hole=0.45,
        ))
        layout = make_plotly_layout("iPhone vs Android (전체 감지)", height=280)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(**layout)
        fig.update_layout(
            annotations=[dict(
                text=f"{total_dev:,}",
                x=0.5, y=0.5, font_size=20, font_color="#ccd6f6",
                showarrow=False,
            )],
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ─── 시간대별 방문 추이 ─────────────────────────────
    st.markdown('<div class="section-header">시간대별 방문 추이</div>', unsafe_allow_html=True)

    hourly = compute_hourly_stats(visits)
    if not hourly.empty:
        fig = px.bar(
            hourly,
            x="hour_label",
            y="visit_count",
            color="restroom",
            barmode="group",
            color_discrete_map=COLORS,
            labels={"hour_label": "시간", "visit_count": "방문 횟수", "restroom": ""},
        )
        fig.update_layout(**make_plotly_layout("시간대별 방문 횟수"))
        st.plotly_chart(fig, use_container_width=True)

    # ─── 실시간 점유 현황 ───────────────────────────────
    st.markdown('<div class="section-header">시간대별 동시 이용자 수 (1분 단위)</div>', unsafe_allow_html=True)

    if not occupancy.empty:
        fig = go.Figure()
        for restroom in ["남자화장실", "여자화장실"]:
            occ_r = occupancy[occupancy["restroom"] == restroom].sort_values("time_bin")
            if not occ_r.empty:
                time_labels = occ_r["time_bin"].apply(ti_to_hhmm)
                fig.add_trace(go.Scatter(
                    x=occ_r["time_bin"],
                    y=occ_r["device_count"],
                    mode="lines",
                    name=restroom,
                    line=dict(color=COLORS[restroom], width=1.5),
                    fill="tozeroy",
                    fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(COLORS[restroom])) + [0.1])}",
                    customdata=time_labels,
                    hovertemplate="%{customdata} | %{y}명<extra>%{fullData.name}</extra>",
                ))
        # time_bin(숫자) → 시간 라벨로 변환
        tick_vals = [h * 360 for h in range(time_range[0], time_range[1] + 1)]
        tick_text = [f"{h:02d}:00" for h in range(time_range[0], time_range[1] + 1)]
        fig.update_layout(**make_plotly_layout("동시 이용자 수 추이 (1분 단위)", height=350))
        fig.update_xaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_text)
        st.plotly_chart(fig, use_container_width=True)

    # ─── 체류 시간 분포 ─────────────────────────────────
    st.markdown('<div class="section-header">체류 시간 분포</div>', unsafe_allow_html=True)

    dist = compute_duration_distribution(visits)
    if not dist.empty:
        # 남녀별 비율 계산
        dist = dist.copy()
        dist["pct"] = 0.0
        for restroom in ["남자화장실", "여자화장실"]:
            mask = dist["restroom"] == restroom
            total = dist.loc[mask, "count"].sum()
            if total > 0:
                dist.loc[mask, "pct"] = (dist.loc[mask, "count"] / total * 100).round(1)
        dist["label"] = dist.apply(lambda r: f"{r['count']}명 ({r['pct']:.0f}%)", axis=1)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.bar(
                dist,
                x="duration_bin",
                y="count",
                color="restroom",
                barmode="group",
                color_discrete_map=COLORS,
                text="label",
                labels={"duration_bin": "체류 시간", "count": "방문 수", "restroom": ""},
            )
            fig.update_traces(textposition="outside", textfont_size=10)
            fig.update_layout(**make_plotly_layout("체류 시간 분포", height=400))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 통계 요약 테이블
            stats_data = []
            for restroom in ["남자화장실", "여자화장실"]:
                rv = visits[visits["restroom"] == restroom]
                if not rv.empty:
                    stats_data.append({
                        "화장실": restroom,
                        "평균": f"{rv['duration_min'].mean():.1f}분",
                        "중앙값": f"{rv['duration_min'].median():.1f}분",
                        "최대": f"{rv['duration_min'].max():.1f}분",
                        "표준편차": f"{rv['duration_min'].std():.1f}분",
                    })
            if stats_data:
                st.markdown("**체류 시간 통계**")
                st.dataframe(pd.DataFrame(stats_data), hide_index=True)

    # ─── 30분 단위 피크 분석 ────────────────────────────
    st.markdown('<div class="section-header">30분 단위 피크 분석</div>', unsafe_allow_html=True)

    peak = compute_peak_analysis(visits)
    if not peak.empty:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("방문 횟수", "평균 체류 시간 (분)"),
        )
        for restroom in ["남자화장실", "여자화장실"]:
            pr = peak[peak["restroom"] == restroom]
            if pr.empty:
                continue
            fig.add_trace(
                go.Bar(
                    x=pr["half_hour_label"], y=pr["visit_count"],
                    name=restroom, marker_color=COLORS[restroom],
                    showlegend=True,
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Bar(
                    x=pr["half_hour_label"], y=pr["avg_duration_min"],
                    name=restroom, marker_color=COLORS[restroom],
                    showlegend=False,
                ),
                row=1, col=2,
            )
        layout = make_plotly_layout(height=380)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(**layout)
        fig.update_xaxes(gridcolor="#1a2035")
        fig.update_yaxes(gridcolor="#1a2035")
        st.plotly_chart(fig, use_container_width=True)

    # ─── AST 히트맵 ─────────────────────────────────────
    st.markdown('<div class="section-header">시간대별 누적 체류 시간 (AST)</div>', unsafe_allow_html=True)

    if not hourly.empty:
        # 피봇으로 히트맵 데이터 구성
        heatmap_data = hourly.pivot_table(
            index="restroom", columns="hour_label", values="total_ast_min", fill_value=0
        )
        fig = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns.tolist(),
            y=heatmap_data.index.tolist(),
            color_continuous_scale="YlOrRd",
            labels=dict(x="시간", y="화장실", color="AST (분)"),
            aspect="auto",
        )
        fig.update_layout(**make_plotly_layout("AST 히트맵 (분)", height=250))
        st.plotly_chart(fig, use_container_width=True)

    # ─── 누적 체류 시간 추이 (AST Timeline) ────────────────
    st.markdown('<div class="section-header">누적 체류 시간 추이 (AST)</div>', unsafe_allow_html=True)

    if not visits.empty:
        # 시간대별 AST 계산: 각 시간대에 시작된 방문의 체류시간 합산 (누적)
        ast_rows = []
        for _, v in visits.iterrows():
            ast_rows.append({
                "time_bin": (int(v["start_ti"]) // 6) * 6,  # 1분 bin
                "restroom": v["restroom"],
                "duration_min": v["duration_min"],
            })
        ast_df = pd.DataFrame(ast_rows)
        ast_by_time = (
            ast_df.groupby(["time_bin", "restroom"])["duration_min"]
            .sum()
            .reset_index()
            .sort_values("time_bin")
        )

        # 누적합 계산
        fig = go.Figure()
        for restroom in ["남자화장실", "여자화장실"]:
            r_data = ast_by_time[ast_by_time["restroom"] == restroom].copy()
            if r_data.empty:
                continue
            r_data["cumulative_ast"] = r_data["duration_min"].cumsum()
            time_labels = r_data["time_bin"].apply(ti_to_hhmm)
            fig.add_trace(go.Scatter(
                x=r_data["time_bin"],
                y=r_data["cumulative_ast"],
                mode="lines",
                name=restroom,
                line=dict(color=COLORS[restroom], width=2.5),
                customdata=time_labels,
                hovertemplate="%{customdata} | %{y:.1f}분<extra>%{fullData.name}</extra>",
            ))

        tick_vals = [h * 360 for h in range(time_range[0], time_range[1] + 1)]
        tick_text = [f"{h:02d}:00" for h in range(time_range[0], time_range[1] + 1)]
        fig.update_layout(**make_plotly_layout("시간에 따른 누적 체류 시간 (분)", height=380))
        fig.update_xaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_text)
        fig.update_yaxes(title_text="누적 AST (분)")
        st.plotly_chart(fig, use_container_width=True)

    # ─── 분류 신뢰도 ──────────────────────────────────────
    st.markdown('<div class="section-header">남녀 분류 신뢰도</div>', unsafe_allow_html=True)

    if "win_rate" in visits.columns:
        col_conf1, col_conf2 = st.columns(2)
        with col_conf1:
            # win_rate 분포 히스토그램
            fig = px.histogram(
                visits, x="win_rate", color="restroom", nbins=20,
                color_discrete_map=COLORS, barmode="overlay", opacity=0.7,
                labels={"win_rate": "Head-to-Head 승률", "count": "방문 수", "restroom": ""},
            )
            fig.update_layout(**make_plotly_layout("분류 승률 분포 (높을수록 확실)", height=300))
            st.plotly_chart(fig, use_container_width=True)

        with col_conf2:
            # 분류 방법 비율
            if "classify_method" in visits.columns:
                method_counts = visits["classify_method"].value_counts()
                method_labels = {
                    "h2h": "Head-to-Head (양쪽 동시 감지)",
                    "count_only": "감지 횟수 비교",
                    "single": "한쪽만 감지",
                }
                fig = go.Figure(go.Pie(
                    labels=[method_labels.get(m, m) for m in method_counts.index],
                    values=method_counts.values,
                    hole=0.4,
                    marker=dict(colors=["#64ffda", "#ffd166", "#a78bfa"]),
                    textinfo="label+percent",
                    textfont=dict(size=11),
                ))
                layout = make_plotly_layout("분류 방법 비율", height=300)
                layout.pop("xaxis", None)
                layout.pop("yaxis", None)
                fig.update_layout(**layout, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # 평균 신뢰도 테이블
            conf_rows = []
            for restroom in ["남자화장실", "여자화장실"]:
                rv = visits[visits["restroom"] == restroom]
                if not rv.empty:
                    conf_rows.append({
                        "화장실": restroom,
                        "평균 승률": f"{rv['win_rate'].mean():.0%}",
                        "승률>=70%": f"{(rv['win_rate']>=0.7).sum():,} ({(rv['win_rate']>=0.7).mean()*100:.0f}%)",
                        "방문 수": f"{len(rv):,}",
                    })
            if conf_rows:
                st.dataframe(pd.DataFrame(conf_rows), hide_index=True, use_container_width=True)

    # ─── 상세 데이터 ────────────────────────────────────
    with st.expander("방문 상세 데이터"):
        display_cols = ["restroom", "start_time", "end_time", "duration_min", "win_rate", "classify_method", "mac_address"]
        display_cols = [c for c in display_cols if c in visits.columns]
        st.dataframe(
            visits[display_cols].sort_values("start_time"),
            use_container_width=True,
            height=400,
        )

# ═══════════════════════════════════════════════════════════
#  날짜 비교 모드
# ═══════════════════════════════════════════════════════════
else:
    st.markdown("## 날짜별 비교 분석")

    all_visits = {}
    all_occupancy = {}
    all_foot_traffic = {}
    for date in dates:
        v, o, ft = load_and_process(date)
        # 시간대 필터
        ti_start = time_range[0] * 360
        ti_end = time_range[1] * 360
        if not v.empty:
            v = v[(v["start_ti"] >= ti_start) & (v["start_ti"] < ti_end)]
        if not o.empty:
            o = o[(o["time_bin"] >= ti_start) & (o["time_bin"] < ti_end)]
        all_visits[date] = v
        all_occupancy[date] = o
        all_foot_traffic[date] = ft

    # 날짜별 비교 테이블
    comparison = compute_daily_comparison(all_visits)
    if comparison.empty:
        st.warning("비교할 데이터가 없습니다.")
        st.stop()

    st.markdown('<div class="section-header">날짜별 요약</div>', unsafe_allow_html=True)
    st.dataframe(
        comparison.rename(columns={
            "date": "날짜", "restroom": "화장실",
            "total_visits": "총 방문", "unique_devices": "고유 디바이스",
            "avg_duration_min": "평균 체류(분)", "total_ast_hours": "AST(시간)",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # 날짜별 방문 추이 비교
    st.markdown('<div class="section-header">날짜별 방문 추이</div>', unsafe_allow_html=True)

    fig = go.Figure()
    date_colors = px.colors.qualitative.Set2
    for i, (date, v) in enumerate(all_visits.items()):
        if v.empty:
            continue
        hourly = v.groupby("start_hour").size().reset_index(name="count")
        fig.add_trace(go.Scatter(
            x=hourly["start_hour"].apply(lambda h: f"{int(h):02d}:00"),
            y=hourly["count"],
            mode="lines+markers",
            name=date,
            line=dict(color=date_colors[i % len(date_colors)], width=2),
        ))
    fig.update_layout(**make_plotly_layout("날짜별 시간대 방문 비교"))
    st.plotly_chart(fig, use_container_width=True)

    # 남녀 비율 비교
    st.markdown('<div class="section-header">날짜별 남녀 방문 비율</div>', unsafe_allow_html=True)

    ratio_data = []
    for date, v in all_visits.items():
        if v.empty:
            continue
        for restroom in ["남자화장실", "여자화장실"]:
            cnt = len(v[v["restroom"] == restroom])
            ratio_data.append({"date": date, "restroom": restroom, "count": cnt})

    if ratio_data:
        rdf = pd.DataFrame(ratio_data)
        fig = px.bar(
            rdf, x="date", y="count", color="restroom",
            barmode="group", color_discrete_map=COLORS,
            labels={"date": "날짜", "count": "방문 횟수", "restroom": ""},
        )
        fig.update_layout(**make_plotly_layout("남녀 방문 비교", height=350))
        st.plotly_chart(fig, use_container_width=True)

    # 체류 시간 박스플롯 비교
    st.markdown('<div class="section-header">날짜별 체류 시간 분포</div>', unsafe_allow_html=True)

    all_v = pd.concat(
        [v.assign(date=d) for d, v in all_visits.items() if not v.empty],
        ignore_index=True,
    )
    if not all_v.empty:
        fig = px.box(
            all_v, x="date", y="duration_min", color="restroom",
            color_discrete_map=COLORS,
            labels={"date": "날짜", "duration_min": "체류 시간(분)", "restroom": ""},
        )
        fig.update_layout(**make_plotly_layout("체류 시간 분포 비교", height=400))
        st.plotly_chart(fig, use_container_width=True)
