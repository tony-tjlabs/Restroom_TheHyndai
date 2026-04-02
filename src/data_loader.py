"""캐시 기반 데이터 로더 (배포 전용).

원본 raw 데이터와 탐지 로직은 포함하지 않음.
사전 생성된 parquet/json 캐시에서 로드만 수행.
"""

import json
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache"

SWARD_MAP = {
    "210002D5": "남자화장실",
    "210003C6": "여자화장실",
}


def get_available_dates() -> list[str]:
    """캐시된 날짜 목록 반환."""
    files = sorted(CACHE_DIR.glob("*_visits.parquet"))
    return [f.stem.replace("_visits", "") for f in files]


def load_cached_data(date_str: str):
    """캐시에서 visits, occupancy, foot_traffic 로드."""
    visits_path = CACHE_DIR / f"{date_str}_visits.parquet"
    occ_path = CACHE_DIR / f"{date_str}_occupancy.parquet"
    ft_path = CACHE_DIR / f"{date_str}_foot_traffic.json"

    visits = pd.read_parquet(visits_path) if visits_path.exists() else pd.DataFrame()
    occupancy = pd.read_parquet(occ_path) if occ_path.exists() else pd.DataFrame()

    if ft_path.exists():
        with open(ft_path) as f:
            foot_traffic = json.load(f)
    else:
        foot_traffic = {"total_unique": 0, "by_type": {}}

    return visits, occupancy, foot_traffic
