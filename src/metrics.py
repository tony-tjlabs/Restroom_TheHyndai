"""화장실 이용 메트릭 계산."""

import pandas as pd
import numpy as np


def compute_summary(visits: pd.DataFrame, occupancy: pd.DataFrame) -> dict:
    """전체 요약 메트릭 계산."""
    if visits.empty:
        return {
            "total_visits": 0,
            "unique_devices": 0,
            "avg_duration_min": 0,
            "median_duration_min": 0,
            "max_duration_min": 0,
            "peak_hour": 0,
            "peak_hour_visits": 0,
            "total_ast_hours": 0,
        }

    # 시간대별 방문 수
    hourly = visits.groupby("start_hour").size()
    peak_hour = int(hourly.idxmax()) if not hourly.empty else 0
    peak_hour_visits = int(hourly.max()) if not hourly.empty else 0

    # AST: 전체 체류 시간 합산 (시간 단위)
    total_ast_hours = round(visits["duration_sec"].sum() / 3600, 1)

    return {
        "total_visits": len(visits),
        "unique_devices": visits["mac_address"].nunique(),
        "avg_duration_min": round(visits["duration_min"].mean(), 1),
        "median_duration_min": round(visits["duration_min"].median(), 1),
        "max_duration_min": round(visits["duration_min"].max(), 1),
        "peak_hour": peak_hour,
        "peak_hour_visits": peak_hour_visits,
        "total_ast_hours": total_ast_hours,
    }


def compute_hourly_stats(visits: pd.DataFrame) -> pd.DataFrame:
    """시간대별 통계."""
    if visits.empty:
        return pd.DataFrame()

    stats = (
        visits.groupby(["start_hour", "restroom"])
        .agg(
            visit_count=("mac_address", "size"),
            unique_devices=("mac_address", "nunique"),
            avg_duration_min=("duration_min", "mean"),
            total_ast_min=("duration_min", "sum"),
        )
        .reset_index()
    )
    stats["avg_duration_min"] = stats["avg_duration_min"].round(1)
    stats["total_ast_min"] = stats["total_ast_min"].round(1)
    stats["hour_label"] = stats["start_hour"].apply(lambda h: f"{int(h):02d}:00")
    return stats


def compute_duration_distribution(visits: pd.DataFrame) -> pd.DataFrame:
    """체류 시간 분포 (구간별)."""
    if visits.empty:
        return pd.DataFrame()

    bins = [0, 1, 2, 3, 5, 10, 20, 60, float("inf")]
    labels = ["~1분", "1~2분", "2~3분", "3~5분", "5~10분", "10~20분", "20~60분", "60분+"]

    visits = visits.copy()
    visits["duration_bin"] = pd.cut(visits["duration_min"], bins=bins, labels=labels, right=False)

    dist = (
        visits.groupby(["duration_bin", "restroom"], observed=True)
        .size()
        .reset_index(name="count")
    )
    return dist


def compute_peak_analysis(visits: pd.DataFrame) -> pd.DataFrame:
    """30분 단위 피크 분석."""
    if visits.empty:
        return pd.DataFrame()

    visits = visits.copy()
    # 30분 단위 bin
    visits["half_hour"] = (visits["start_ti"] // 180) * 180  # 180 = 30min * 6
    visits["half_hour_label"] = visits["half_hour"].apply(
        lambda ti: f"{(ti*10)//3600:02d}:{((ti*10)%3600)//60:02d}"
    )

    peak = (
        visits.groupby(["half_hour", "half_hour_label", "restroom"])
        .agg(
            visit_count=("mac_address", "size"),
            unique_devices=("mac_address", "nunique"),
            avg_duration_min=("duration_min", "mean"),
        )
        .reset_index()
    )
    peak["avg_duration_min"] = peak["avg_duration_min"].round(1)
    return peak


def compute_daily_comparison(all_visits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """날짜별 비교 테이블."""
    rows = []
    for date_str, visits in all_visits.items():
        if visits.empty:
            continue
        for restroom in visits["restroom"].unique():
            rv = visits[visits["restroom"] == restroom]
            rows.append({
                "date": date_str,
                "restroom": restroom,
                "total_visits": len(rv),
                "unique_devices": rv["mac_address"].nunique(),
                "avg_duration_min": round(rv["duration_min"].mean(), 1),
                "total_ast_hours": round(rv["duration_sec"].sum() / 3600, 1),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()
