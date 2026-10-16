"""Fetch and validate official SGIS population data against the released boundary."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://sgisapi.mods.go.kr/OpenAPI3"


def request_json(path: str, params: dict[str, str]) -> dict:
    """Call SGIS and require its application-level success response."""
    with urlopen(f"{BASE}/{path}?{urlencode(params)}", timeout=120) as response:
        payload = json.load(response)
    if str(payload.get("errCd")) != "0":
        raise RuntimeError(f"SGIS error {payload.get('errCd')}: {payload.get('errMsg')}")
    return payload


def main() -> int:
    load_dotenv(ROOT / ".env")
    key = os.getenv("SGIS_CONSUMER_KEY")
    secret = os.getenv("SGIS_CONSUMER_SECRET")
    if not key or not secret:
        raise RuntimeError("SGIS_CONSUMER_KEY and SGIS_CONSUMER_SECRET are required")
    auth = request_json("auth/authentication.json", {"consumer_key": key, "consumer_secret": secret})
    token = auth["result"]["accessToken"]

    boundary = gpd.read_parquet(ROOT / "data/release/changwon_transit_context_by_dong.parquet")
    prefixes = sorted(boundary.dong_code.astype(str).str[:5].unique())
    total_rows: list[dict] = []
    elderly_rows: list[dict] = []
    raw_total: list[dict] = []
    raw_elderly: list[dict] = []
    for prefix in prefixes:
        common = {"accessToken": token, "year": "2024", "adm_cd": prefix, "low_search": "1"}
        total = request_json("stats/population.json", common)
        elderly = request_json("stats/searchpopulation.json", {**common, "gender": "0", "age_type": "24"})
        raw_total.append(total)
        raw_elderly.append(elderly)
        total_rows.extend(total.get("result", []))
        elderly_rows.extend(elderly.get("result", []))

    total_frame = pd.DataFrame(total_rows).rename(columns={"tot_ppltn": "population_total", "adm_nm": "total_adm_nm"})
    elderly_frame = pd.DataFrame(elderly_rows).rename(columns={"population": "population_65_plus", "adm_nm": "elderly_adm_nm"})
    total_frame["adm_cd"] = total_frame.adm_cd.astype("string")
    elderly_frame["adm_cd"] = elderly_frame.adm_cd.astype("string")
    total_frame["population_total"] = pd.to_numeric(total_frame.population_total, errors="coerce")
    elderly_frame["population_65_plus"] = pd.to_numeric(elderly_frame.population_65_plus, errors="coerce")
    population = total_frame[["adm_cd", "total_adm_nm", "population_total"]].merge(
        elderly_frame[["adm_cd", "elderly_adm_nm", "population_65_plus"]], on="adm_cd", how="outer", validate="one_to_one"
    )
    codes = set(boundary.dong_code.astype(str))
    population = population.loc[population.adm_cd.isin(codes)].copy()
    joined = boundary[["dong_code", "dong_name"]].merge(population, left_on="dong_code", right_on="adm_cd", how="outer", indicator=True)
    report = {
        "auth": {"token_success": True},
        "api": {
            "total_response_fields": sorted(total_frame.columns.tolist()),
            "elderly_response_fields": sorted(elderly_frame.columns.tolist()),
            "total_rows": len(total_frame),
            "elderly_rows": len(elderly_frame),
            "reference_year": 2024,
            "adm_cd_lengths": sorted(population.adm_cd.str.len().dropna().astype(int).unique().tolist()),
        },
        "code_match": {
            "boundary": len(boundary), "population": len(population),
            "matched": int(joined._merge.eq("both").sum()),
            "unmatched_boundary": int(joined._merge.eq("left_only").sum()),
            "unmatched_population": int(joined._merge.eq("right_only").sum()),
            "duplicate_adm_cd": int(population.adm_cd.duplicated(keep=False).sum()),
            "name_mismatch": int((joined.dong_name.notna() & joined.total_adm_nm.notna() & joined.dong_name.ne(joined.total_adm_nm)).sum()),
        },
        "quality": {
            "negative_total": int((population.population_total < 0).sum()),
            "negative_elderly": int((population.population_65_plus < 0).sum()),
            "elderly_gt_total": int((population.population_65_plus > population.population_total).sum()),
            "null_population": int(population[["population_total", "population_65_plus"]].isna().any(axis=1).sum()),
        },
    }
    success = (
        report["code_match"] == {"boundary": 55, "population": 55, "matched": 55, "unmatched_boundary": 0, "unmatched_population": 0, "duplicate_adm_cd": 0, "name_mismatch": report["code_match"]["name_mismatch"]}
        and not any(report["quality"].values())
        and report["api"]["adm_cd_lengths"] == [8]
    )
    report["status"] = "PASS" if success else "FAIL"
    output = ROOT / "reports/sgis_population_validation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if success:
        raw_dir = ROOT / "data/raw/population/sgis"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "population_2024.json").write_text(json.dumps(raw_total, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (raw_dir / "population_65_plus_2024.json").write_text(json.dumps(raw_elderly, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        population.to_parquet(ROOT / "data/processed/sgis_population_2024.parquet", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
