# Data acquisition and publication status

이 폴더는 안내문만 포함합니다. 원본·중간·생산 데이터는 이 공개 저장소에 없습니다. 아래 경로에
직접 데이터를 배치해야 코드가 동작합니다. 정확한 컬럼·스키마는 `metadata/*.yaml`을 참고하세요.

2026-09-21 권리 감사(Publication Rights Audit) 결론을 기준으로 합니다. 공식 포털의 이용허락범위
문구를 확인했다고 해서 이 저장소가 만든 혼합·파생 산출물 전체의 재배포가 자동으로 허가되는 것은
아닙니다. 최종 공개 판단은 [출처와 조건](../docs/data-attribution.md)을 따릅니다.

| Dataset | Provider | Expected local path | Status |
|---|---|---|---|
| 버스정류소 Master | 경상남도 창원시 | `data/raw/stops/stops.csv` | SOURCE_LINK_ONLY — 로컬 스냅샷(2024-12-31)과 현재 공식 배포본(2025-12-31) 연결 증빙 필요 |
| 정류소 시간대별 이용량 | 미확정(STCIS 경유 여부 확인 필요) | `data/raw/ridership/ridership.csv` | RIGHTS_REVIEW — 2025-08-18~25 snapshot의 전달·재배포 권한 미확정 |
| TAGO 버스노선·배차 | 국토교통부 TAGO | `data/raw/route/`, `data/raw/transit_service_frequency/` | SOURCE_LINK_ONLY — 공식 포털 이용허락범위 제한 없음, API 인증키는 재배포 대상 아님 |
| 행정동 경계 | VWorld/국토교통부 | `data/raw/BND_ADM_DONG_PG.zip` | RIGHTS_REVIEW — 자료별 다운로드 조건 미확인 |
| SGIS 인구(2024) | 국가데이터처 SGIS | `data/raw/population/`, `data/processed/sgis_population_2024.parquet` | RIGHTS_REVIEW — API 사이트 활용과 정적 재배포 허가 구분 필요 |
| NGII 인구격자 | 국토지리정보원 | `data/raw/population_grid/` | PUBLIC (공공누리 제1유형, 출처표시 조건) — 원본 파일 자체는 용량상 미포함 |
| HIRA 병원정보(POI) | 건강보험심사평가원 | 기존 metadata 지정 cache | PUBLIC (공공누리 제1유형, 출처표시 조건) |
| 행정기관 목록(POI) | 행정안전부 | 기존 metadata 지정 cache | PUBLIC (이용허락범위 제한 없음, 좌표 보완분은 OSM 조건 별도) |
| 횡단보도 현황 | 창원시 | `data/raw/` | PUBLIC (이용허락범위 제한 없음) |
| 도로현황(도로폭) | 창원시 | `data/raw/` | PUBLIC (이용허락범위 제한 없음, 현재 분석에 미채택) |
| OSM 보행 네트워크 | OpenStreetMap contributors | `data/raw/osm/`, `data/interim/` | NOT_REDISTRIBUTED — ODbL 1.0, [출처와 조건](../docs/data-attribution.md) 참고 |
| 생산·파생 분석 테이블 | 본 프로젝트 산출 | `data/processed/`, `data/release/` | NOT_REDISTRIBUTED — 원천 데이터 권리가 해결되기 전에는 파생 테이블도 공개하지 않음 |

## 공개하지 않은 이유

원본 CSV·API 응답·Parquet·Shapefile은 이 저장소에 없습니다. 공식 포털에 "이용허락범위 제한 없음"
문구가 있다는 사실만으로 이 프로젝트가 여러 출처를 결합·가공한 결과물 전체의 재배포가 허가되는 것은
아니라고 판단해 데이터 자체는 보류했습니다. Interactive Dashboard와 화면 캡처도 같은 이유로
공개하지 않았습니다. 코드·스키마·검증 방법론만 공개합니다.

## 재현하려면

1. 위 표의 공식 링크에서 동일한 종류의 데이터를 직접 획득합니다 (동일 snapshot을 보장하지 않습니다).
2. 기재된 로컬 경로에 배치합니다.
3. `.env.example`을 `.env`로 복사하고 필요한 API 키를 채웁니다. 키를 커밋하지 마세요.
4. [재현 안내](../docs/reproduction.md)를 따릅니다.

데이터 없이도 `tests/`의 순수 단위 테스트는 실행할 수 있습니다.
