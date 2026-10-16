"""Aggregate the official 창원시 횡단보도 현황 (crosswalk inventory) file to dong level.

Source: 경상남도 창원시_창원시 횡단보도 현황_20211231 (data.go.kr dataset 15106905),
downloaded via the portal's public, login-free file-download endpoint (no API
key required for this file dataset). Raw CSV (CP949, unmodified) is kept at
data/raw/road_pedestrian/changwon_crosswalk_20211231.csv.

IMPORTANT LIMITATION: this source file has NO coordinates (no lat/lon,
no geometry) — only a free-text "행정읍면동" (district + dong name) column.
Per the project's matching rules (official ID / spatial join preferred,
no name-only matching for stops/Anchors), this file is therefore usable
ONLY as a dong-level administrative context layer, never for Anchor- or
stop-level physical-condition claims. Rows whose dong label does not exact-
match the current 55-dong canonical list are reported as UNMATCHED, not
guessed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_PATH = ROOT / "data/raw/road_pedestrian/changwon_crosswalk_20211231.csv"
DISTRICT_PREFIX = re.compile(r"^[가-힣]+구\s*")
KNOWN_LEGACY_SPELLING = {"팔용동": "팔룡동"}  # 용/룡 한자 이표기, 원본 값은 보존하고 매칭에만 사용


def main() -> int:
    raw = pd.read_csv(RAW_PATH, encoding="cp949")
    dong = pd.read_parquet(ROOT / "data/release/changwon_drt_dong_features.parquet", columns=["adm_cd", "adm_nm"])
    canonical_names = set(dong.adm_nm)

    raw["dong_label_raw"] = raw["행정읍면동"].fillna("")
    raw["dong_label_normalized"] = raw.dong_label_raw.str.replace(DISTRICT_PREFIX, "", regex=True).str.strip()
    raw["dong_label_normalized"] = raw.dong_label_normalized.replace(KNOWN_LEGACY_SPELLING)
    raw["is_matched"] = raw.dong_label_normalized.isin(canonical_names)

    matched = raw[raw.is_matched].copy()
    unmatched_labels = sorted(raw.loc[~raw.is_matched, "dong_label_raw"].unique().tolist())

    def yn(series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().isin(["있음", "예", "Y", "y", "유"])

    matched["exists"] = yn(matched["존재유무"])
    matched["curb_cut"] = yn(matched["보도턱낮춤유무"])
    matched["accessible_facility"] = yn(matched["장애시설유무"])

    agg = matched.groupby("dong_label_normalized").agg(
        crosswalk_count=("FID", "count"),
        crosswalk_exists_count=("exists", "sum"),
        curb_cut_count=("curb_cut", "sum"),
        accessible_facility_count=("accessible_facility", "sum"),
        mean_crosswalk_width_m=("횡단보도폭", "mean"),
        mean_crosswalk_length_m=("횡단보도길이", "mean"),
    ).reset_index().rename(columns={"dong_label_normalized": "adm_nm"})
    agg["curb_cut_share"] = agg.curb_cut_count / agg.crosswalk_count
    agg["accessible_facility_share"] = agg.accessible_facility_count / agg.crosswalk_count

    result = dong.merge(agg, on="adm_nm", how="left")
    result["has_crosswalk_data"] = result.crosswalk_count.notna()

    release_path = ROOT / "data/release/changwon_pedestrian_context_by_dong.parquet"
    result.to_parquet(release_path, index=False)

    qa = {
        "source_raw_rows": int(len(raw)),
        "matched_rows": int(len(matched)),
        "unmatched_rows": int((~raw.is_matched).sum()),
        "unmatched_labels_raw": unmatched_labels,
        "canonical_dong_count": int(len(dong)),
        "dong_with_crosswalk_data": int(result.has_crosswalk_data.sum()),
        "dong_without_crosswalk_data": sorted(result.loc[~result.has_crosswalk_data, "adm_nm"].tolist()),
        "known_legacy_spelling_applied": KNOWN_LEGACY_SPELLING,
        "matching_method": "EXACT_NAME_AFTER_DISTRICT_PREFIX_STRIP (좌표 없음 — Anchor/정류소 단위 매칭에는 사용하지 않음, 행정동 수준 context 전용)",
        "limitation": "원본 파일 기준일(2021-12-31)과 현재 canonical 행정동 경계 기준일이 달라, 이후 행정동 명칭 변경/분동(예: 반월동→반월중앙동, 노산동/동서동/태평동 등 통합)으로 매칭되지 않는 라벨이 존재한다. 이 행은 추정하여 채우지 않고 UNMATCHED로 남긴다.",
    }
    (ROOT / "reports/pedestrian_context_quality_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
