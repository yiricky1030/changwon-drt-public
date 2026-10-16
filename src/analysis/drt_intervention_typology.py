"""Exploratory dong intervention typology layered on existing DRT results.

The scores in this module are definition-match diagnostics. They are not DRT
suitability, demand forecasts, or replacements for the released scenario ranks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


PERIOD_COLUMNS = {
    "morning": "morning_ridership",
    "daytime": "daytime_ridership",
    "evening": "evening_17_23_ridership",
    "late_night": "late_night_ridership",
}
PERIOD_LABELS = {
    "morning": "오전 집중시간대(06~08시)",
    "daytime": "주간(09~15시)",
    "evening": "저녁·야간(16~23시)",
    "late_night": "심야(00~05시)",
    "unavailable": "시간대 확인 불가",
}
TYPE_ORDER = ["복합 취약 검토형", "생활이동 지원 검토형", "기존 노선 보완 검토형", "접근성 보완 검토형", "시간대 집중 대응 검토형"]


@dataclass(frozen=True)
class TypologyResult:
    rows: pd.DataFrame
    thresholds: dict[str, dict[str, float | int]]
    definitions: dict[str, Any]


def percentile(series: pd.Series, *, higher_is_signal: bool = True) -> pd.Series:
    """Average-tie percentile on observed values; preserve missing values."""
    values = pd.to_numeric(series, errors="coerce")
    ranked = values.rank(method="average", pct=True) * 100
    return ranked if higher_is_signal else 100 - ranked


def aggregate_accessibility(network: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed stop-level walk accessibility without imputing NULLs."""
    source = network.copy()
    source["nearest_walk_pct"] = percentile(source["nearest_stop_walk_distance_m"])
    metrics = {
        "network_isolation_pct": ("walk_isolation_pct_citywide", "median"),
        "local_stop_walk_scarcity_pct": ("local_stop_walk_scarcity_pct_citywide", "median"),
        "local_route_walk_scarcity_pct": ("local_route_walk_scarcity_pct_citywide", "median"),
        "nearest_walk_distance_pct": ("nearest_walk_pct", "median"),
        "network_reachable_stop_count": ("network_reachable_flag", "sum"),
        "network_stop_count": ("stop_id", "count"),
    }
    return source.groupby(["adm_cd", "adm_nm"], as_index=False).agg(**metrics)


def _thresholds(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | int]]:
    result = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        result[column] = {
            "count": int(values.size), "missing": int(len(frame) - values.size),
            "q25": float(values.quantile(.25)), "median": float(values.quantile(.5)),
            "q75": float(values.quantile(.75)),
        }
    return result


def _scenario(primary: str, period: str) -> str:
    period_text = {
        "morning": "오전 집중시간대", "daytime": "주간", "evening": "저녁 집중시간대",
        "late_night": "심야 시간대", "unavailable": "시간대 추가 확인 후",
    }[period]
    mapping = {
        "생활이동 지원 검토형": f"{period_text} 생활이동 지원형 DRT 검토",
        "기존 노선 보완 검토형": f"{period_text} 기존 노선 연계형 DRT 검토",
        "접근성 보완 검토형": "Anchor 연계형 접근성 보완 DRT 검토",
        "시간대 집중 대응 검토형": f"{period_text} 한정 운영 검토",
        "복합 취약 검토형": "복수 문제 구조와 시간대를 함께 고려하는 우선 현장검토",
        "상대적 우선 검토 낮음": "현재 지표상 추가 현장 확인 중심의 DRT 검토",
    }
    return mapping[primary]


def classify(dong: pd.DataFrame, network: pd.DataFrame) -> TypologyResult:
    frame = dong.copy().merge(aggregate_accessibility(network), on=["adm_cd", "adm_nm"], how="left", validate="one_to_one")
    frame["elderly_signal"] = frame["elderly_ratio_pct"]
    frame["stop_scarcity_signal"] = frame["stop_supply_scarcity_pct"]
    frame["route_scarcity_signal"] = frame["route_supply_scarcity_pct"]
    frame["demand_signal"] = frame["ridership_per_1000_population_pct"]
    frame["observed_stop_demand_signal"] = frame["ridership_per_observed_stop_pct"]
    frame["daytime_signal"] = percentile(frame["daytime_share"])
    frame["temporal_signal"] = percentile(frame["top3_hour_share"])
    frame["access_signal"] = frame[["network_isolation_pct", "local_stop_walk_scarcity_pct", "local_route_walk_scarcity_pct", "nearest_walk_distance_pct"]].mean(axis=1, skipna=False)
    frame["supply_signal"] = frame[["stop_scarcity_signal", "route_scarcity_signal"]].mean(axis=1)
    frame["combined_demand_signal"] = frame[["demand_signal", "observed_stop_demand_signal"]].mean(axis=1, skipna=False)

    periods = list(PERIOD_COLUMNS)
    frame["evening_17_23_ridership"] = pd.to_numeric(frame["evening_ridership"], errors="coerce") + pd.to_numeric(frame["night_ridership"], errors="coerce")
    period_values = frame[[PERIOD_COLUMNS[p] for p in periods]].apply(pd.to_numeric, errors="coerce")
    period_total = period_values.sum(axis=1, min_count=1)
    dominant_column = period_values.fillna(-np.inf).idxmax(axis=1)
    frame["dominant_period"] = dominant_column.map({v: k for k, v in PERIOD_COLUMNS.items()}).fillna("unavailable")
    frame.loc[period_total.le(0) | period_total.isna(), "dominant_period"] = "unavailable"
    frame["dominant_period_share"] = period_values.max(axis=1, skipna=True).div(period_total.where(period_total.gt(0)))
    def second_period(row: pd.Series) -> str | None:
        available = row.dropna().sort_values(ascending=False)
        return ({v: k for k, v in PERIOD_COLUMNS.items()}.get(available.index[1]) if len(available) > 1 else None)
    frame["second_period"] = period_values.apply(second_period, axis=1)
    frame["concentration_index"] = frame["dominant_period_share"]

    evidence = pd.DataFrame(index=frame.index)
    evidence["생활이동 지원 검토형"] = frame[["elderly_signal", "supply_signal", "daytime_signal"]].mean(axis=1, skipna=False)
    evidence["기존 노선 보완 검토형"] = frame[["combined_demand_signal", "route_scarcity_signal", "stop_scarcity_signal"]].mean(axis=1, skipna=False)
    evidence["접근성 보완 검토형"] = frame["access_signal"]
    evidence["시간대 집중 대응 검토형"] = frame[["temporal_signal", "combined_demand_signal"]].mean(axis=1, skipna=False)
    domains = frame[["elderly_signal", "supply_signal", "combined_demand_signal", "access_signal", "temporal_signal"]]
    frame["high_risk_domain_count"] = domains.ge(75).sum(axis=1)
    evidence["복합 취약 검토형"] = domains.apply(lambda r: r.nlargest(3).mean(), axis=1)

    specialized = evidence.drop(columns="복합 취약 검토형")
    frame["primary_type"] = specialized.idxmax(axis=1)
    frame.loc[frame["high_risk_domain_count"].ge(3), "primary_type"] = "복합 취약 검토형"
    frame.loc[specialized.max(axis=1).lt(45) & frame["high_risk_domain_count"].eq(0), "primary_type"] = "상대적 우선 검토 낮음"
    frame["secondary_type"] = specialized.apply(lambda r: r.sort_values(ascending=False).index[1], axis=1)
    frame.loc[frame["primary_type"].eq("복합 취약 검토형"), "secondary_type"] = specialized.idxmax(axis=1)
    frame["type_evidence_score"] = [round(float(evidence.loc[i, p]), 1) if p in evidence else round(float(specialized.loc[i].max()), 1) for i, p in frame["primary_type"].items()]

    reason_maps = {
        "elderly_signal": "고령인구 비율이 55개 동에서 상대적으로 높음",
        "supply_signal": "인구 대비 정류소·노선 공급 취약 신호가 큼",
        "combined_demand_signal": "관측 이용 신호가 55개 동에서 상대적으로 높음",
        "access_signal": "보행 네트워크 접근성 취약 신호가 큼",
        "temporal_signal": "이용이 일부 시간대에 상대적으로 집중됨",
        "daytime_signal": "주간 이용 비중이 상대적으로 높음",
    }
    reason_columns = list(reason_maps)
    for idx, row in frame.iterrows():
        ranked = sorted(reason_columns, key=lambda c: (-float(row[c]) if pd.notna(row[c]) else np.inf, c))[:3]
        for n, column in enumerate(ranked, 1):
            frame.loc[idx, f"reason_{n}"] = reason_maps[column]
    frame["operation_review_scenario"] = [_scenario(t, p) for t, p in zip(frame.primary_type, frame.dominant_period)]
    frame["analysis_limitations"] = "실제 DRT 호출 OD·이동 목적·차량 운영조건이 없고 이용량은 8일 관측이므로 탐색적 정책 검토에 한정"

    definitions = {
        "priority_rule": "3개 이상 high-risk domain이면 복합 취약 검토형; 그 외 유형 evidence 최대; 모든 전문 유형 evidence 45 미만이면 상대적 우선 검토 낮음",
        "evidence": "각 유형 정의에 포함된 무가중 percentile 신호의 단순 평균; DRT 필요도 또는 적합도 점수가 아님",
        "high": ">= 75th percentile", "medium": "25th to 75th percentile", "low": "<= 25th percentile",
        "temporal": {"buckets": {k: PERIOD_LABELS[k] for k in periods}, "concentration": "dominant_period_share = max period ridership / four-period ridership"},
    }
    threshold_columns = ["elderly_ratio", "stops_per_1000_population", "routes_per_1000_population", "ridership_per_1000_population", "ridership_per_observed_stop", "daytime_share", "top3_hour_share", "network_isolation_pct", "nearest_walk_distance_pct"]
    return TypologyResult(frame, _thresholds(frame, threshold_columns), definitions)
