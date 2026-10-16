"""Feedback 9 strict re-audit of service-frequency grain and time units.

This analysis remains separate from every canonical score, rank and weight.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVES = ["회원2동", "병암동", "양덕2동", "합성1동", "월영동"]
SUPPLY = "estimated_route_trips_serving_dong_weekday"


def pair(frame: pd.DataFrame, x: str, y: str) -> dict:
    sub = frame[[x, y]].dropna()
    pr, sr = stats.pearsonr(sub[x], sub[y]), stats.spearmanr(sub[x], sub[y])
    return {"n": int(len(sub)), "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
            "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue)}


def ols(y: np.ndarray, X: np.ndarray, names: list[str]) -> dict:
    X1 = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    residual = y - X1 @ beta
    n, k = X1.shape; dof = n - k
    sigma2 = float(residual @ residual / dof)
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X1.T @ X1)))
    t = beta / se; p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    ss_total = float(((y - y.mean()) ** 2).sum()); r2 = 1 - float(residual @ residual) / ss_total
    return {"n": int(n), "r_squared": r2, "adj_r_squared": 1 - (1-r2)*(n-1)/dof,
            "coefficients": {name: {"coef": float(b), "se": float(s), "t": float(tv), "p": float(pv)}
                             for name,b,s,tv,pv in zip(["intercept", *names], beta, se, t, p)}}


def actual_weekday_average() -> tuple[pd.DataFrame, list[str]]:
    facts = pd.read_parquet(ROOT / "data/release/fact_stop_ridership.parquet")
    profiles = pd.read_parquet(ROOT / "data/release/changwon_transit_stop_profile.parquet", columns=["stop_id", "is_analysis_ready"])
    spatial = pd.read_parquet(ROOT / "data/processed/stop_dong_spatial_join_qa.parquet", columns=["stop_id", "dong_code"])
    ready = profiles.loc[profiles.is_analysis_ready, ["stop_id"]].copy(); ready["stop_id"] = ready.stop_id.astype(str)
    inside = spatial.loc[spatial.dong_code.notna(), ["stop_id", "dong_code"]].rename(columns={"dong_code": "adm_cd"})
    inside["stop_id"] = inside.stop_id.astype(str)
    selected = facts.assign(stop_id=facts.stop_id.astype(str)).merge(ready, on="stop_id", validate="many_to_one").merge(inside, on="stop_id", validate="many_to_one")
    selected["service_date"] = pd.to_datetime(selected.service_date)
    weekdays = selected.loc[selected.service_date.dt.dayofweek.lt(5)]
    dates = sorted(weekdays.service_date.dt.date.astype(str).unique().tolist())
    daily = weekdays.groupby(["adm_cd", "service_date"], observed=True).ridership.sum(min_count=1).reset_index()
    result = daily.groupby("adm_cd", observed=True).agg(
        observed_weekday_count=("service_date", "nunique"),
        observed_weekday_ridership_total=("ridership", lambda s: s.sum(min_count=1)),
        average_observed_weekday_ridership=("ridership", "mean"),
    ).reset_index()
    return result, dates


def grain_audit() -> tuple[dict, pd.DataFrame]:
    bridge = pd.read_parquet(ROOT / "data/release/changwon_transit_route_stop.parquet")
    stops = pd.read_parquet(ROOT / "data/release/changwon_drt_stop_features.parquet", columns=["stop_id", "adm_cd"])
    schedule = pd.read_parquet(ROOT / "data/release/changwon_transit_route_schedule.parquet", columns=["route_id", "estimated_scheduled_trips_weekday"])
    matched = bridge.loc[bridge.is_matched.fillna(False), ["route_id", "stop_id"]].drop_duplicates()
    matched["stop_id"] = matched.stop_id.astype(str); stops["stop_id"] = stops.stop_id.astype(str)
    route_stop_dong = matched.merge(stops, on="stop_id", how="inner")[["route_id", "stop_id", "adm_cd"]]
    route_dong = route_stop_dong[["route_id", "adm_cd"]].drop_duplicates()
    dedup = route_dong.merge(schedule, on="route_id", validate="many_to_one").groupby("adm_cd").estimated_scheduled_trips_weekday.sum(min_count=1)
    stop_events = route_stop_dong.merge(schedule, on="route_id", validate="many_to_one").groupby("adm_cd").estimated_scheduled_trips_weekday.sum(min_count=1)
    multiplier = stop_events / dedup
    report = {"source_schedule_grain": "one row per route_id", "source_route_id_unique": bool(schedule.route_id.is_unique),
              "route_stop_dong_rows": int(len(route_stop_dong)), "distinct_route_dong_pairs": int(len(route_dong)),
              "duplicate_route_stop_dong_excess_removed": int(len(route_stop_dong) - len(route_dong)),
              "production_aggregation_grain": "route_id × adm_cd, one route estimate per dong",
              "route_stop_duplicated_in_production": False,
              "counterfactual_stop_event_overstatement_multiplier": {"mean": float(multiplier.mean()), "median": float(multiplier.median()), "max": float(multiplier.max())},
              "semantic_label": "2026-09 TAGO snapshot based estimated route trips serving dong per representative weekday",
              "not_measured_vehicle_runs": True, "not_scheduled_stop_events": True}
    return report, dedup.rename(SUPPLY).reset_index()


def regression(frame: pd.DataFrame, ridership: str) -> dict:
    f = frame.dropna(subset=[ridership, SUPPLY, "population_total", "elderly_ratio", "density_per_km2"])
    f = f[(f[ridership] > 0) & (f[SUPPLY] > 0) & (f.population_total > 0) & (f.density_per_km2 > 0)]
    result = ols(np.log(f[ridership].to_numpy()), np.column_stack([np.log(f[SUPPLY]), np.log(f.population_total), f.elderly_ratio, np.log(f.density_per_km2)]),
                 ["log_estimated_route_trips_serving_dong_weekday", "log_population_total", "elderly_ratio", "log_density_per_km2"])
    result["dependent_variable"] = f"log({ridership})"
    return result


def stratified(frame: pd.DataFrame, ridership_per_1000: str) -> list[dict]:
    f = frame.dropna(subset=[SUPPLY, ridership_per_1000]).copy()
    f["supply_tercile"] = pd.qcut(f[SUPPLY], 3, labels=["LOW", "MID", "HIGH"], duplicates="drop")
    out = f.groupby("supply_tercile", observed=True).agg(
        n=("adm_nm", "count"), supply_min=(SUPPLY, "min"), supply_max=(SUPPLY, "max"),
        ridership_per_1000_mean=(ridership_per_1000, "mean"), ridership_per_1000_std=(ridership_per_1000, "std"),
        ridership_per_1000_min=(ridership_per_1000, "min"), ridership_per_1000_max=(ridership_per_1000, "max")).reset_index()
    out["ridership_per_1000_cv"] = out.ridership_per_1000_std / out.ridership_per_1000_mean
    return out.to_dict("records")


def ratio_summary(frame: pd.DataFrame, numerator: str) -> dict:
    ratio = (frame[numerator] / frame[SUPPLY]).replace([np.inf, -np.inf], np.nan)
    return {"n_valid": int(ratio.notna().sum()), "mean": float(ratio.mean()), "median": float(ratio.median()),
            "min": float(ratio.min()), "max": float(ratio.max())}


def main() -> int:
    grain, rebuilt_supply = grain_audit()
    intensity = pd.read_parquet(ROOT / "data/release/changwon_transit_service_intensity_by_dong.parquet")
    assert SUPPLY in intensity and "sum_estimated_scheduled_trips_weekday" not in intensity
    pd.testing.assert_series_equal(intensity.set_index("adm_cd")[SUPPLY].sort_index(), rebuilt_supply.set_index("adm_cd")[SUPPLY].sort_index(), check_names=False)
    dong = pd.read_parquet(ROOT / "data/release/changwon_drt_dong_features.parquet", columns=["adm_cd", "adm_nm", "area_km2", "population_total", "elderly_ratio", "ridership_total", "ridership_per_1000_population", "weekday_daily_avg_ridership"])
    weekday, dates = actual_weekday_average()
    frame = dong.merge(intensity.drop(columns="adm_nm"), on="adm_cd", validate="one_to_one").merge(weekday, on="adm_cd", how="left", validate="one_to_one")
    frame["density_per_km2"] = frame.population_total / frame.area_km2
    frame["average_observed_weekday_ridership_per_1000_population"] = frame.average_observed_weekday_ridership / frame.population_total * 1000
    np.testing.assert_allclose(frame.weekday_daily_avg_ridership.astype(float), frame.average_observed_weekday_ridership.astype(float), equal_nan=True)
    observed = frame.loc[frame.ridership_total.notna()].copy()
    before = {"numerator": "ridership_total, eight observed dates including weekend", "denominator": SUPPLY + ", representative weekday one-day estimate",
              "time_units_aligned": False, "key_pair": pair(observed, "ridership_total", SUPPLY), "conditional_regression": regression(observed, "ridership_total"),
              "stratified": stratified(observed, "ridership_per_1000_population"), "ratio": ratio_summary(observed, "ridership_total")}
    after = {"numerator": "average_observed_weekday_ridership, six actual weekdays", "denominator": SUPPLY + ", representative weekday one-day estimate",
             "time_units_aligned": True, "key_pair": pair(observed, "average_observed_weekday_ridership", SUPPLY), "conditional_regression": regression(observed, "average_observed_weekday_ridership"),
             "stratified": stratified(observed, "average_observed_weekday_ridership_per_1000_population"), "ratio": ratio_summary(observed, "average_observed_weekday_ridership")}
    reps = []
    for row in frame.loc[frame.adm_nm.isin(REPRESENTATIVES)].sort_values("adm_nm").itertuples():
        supply = float(getattr(row, SUPPLY))
        reps.append({"adm_nm": row.adm_nm, "routes_serving_dong": int(row.routes_serving_dong), SUPPLY: supply,
                     "ridership_total_8day": None if pd.isna(row.ridership_total) else float(row.ridership_total),
                     "average_observed_weekday_ridership": None if pd.isna(row.average_observed_weekday_ridership) else float(row.average_observed_weekday_ridership),
                     "before_misaligned_ratio": None if pd.isna(row.ridership_total) else float(row.ridership_total / supply),
                     "after_time_aligned_ratio": None if pd.isna(row.average_observed_weekday_ridership) else float(row.average_observed_weekday_ridership / supply)})
    result = {"status": "PARTIALLY_RESOLVED", "aggregation_grain_audit": grain,
              "temporal_alignment": {"ridership_observation_range": ["2025-08-18", "2025-08-25"], "observed_weekday_dates": dates, "observed_weekday_count": len(dates),
                                     "schedule_snapshot_reference_date": "2026-09-12", "cross_snapshot_comparison": True,
                                     "limitation": "Time units now match at representative weekday/day, but reference dates differ by about 13 months; the schedule estimate is not 2025 supply."},
              "before_time_misaligned": before, "after_time_aligned": after, "representative_dong": reps,
              "canonical_analysis_changed": False,
              "conclusion": "Aggregation was already route×dong deduplicated. The time-misaligned ratio was invalid and is replaced by average observed weekday ridership per estimated route trip. Reference-date mismatch and endogeneity remain, so Feedback 9 stays PARTIALLY_RESOLVED."}
    (ROOT / "reports/observed_use_supply_dependence_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": result["status"], "grain": grain, "before": before["key_pair"], "after": after["key_pair"], "representatives": reps}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
