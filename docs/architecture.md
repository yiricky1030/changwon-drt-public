# Architecture

입력 → 손실 없는 ETL → PostGIS/파일 출력 → 행정동·정류소 분석 → 불확실성 진단이 공개 코드의 흐름입니다.

`src/ingest`, `transform`, `quality`, `export`는 읽기·변환·검증·출력을 나눕니다. `src/database`는 `sql/`과 환경변수로 PostGIS를 구성합니다. `src/analysis`는 시나리오·Anchor·POI·보행망·공간 진단 함수를 제공합니다.

`scripts/`에는 기본 ETL, TAGO/SGIS 수집·검증, 특징 생성, 시나리오·Anchor·유형·보행망·POI 분석과 관측/기여도 진단 CLI가 있습니다. 이들은 실제 입력이나 선행 산출물이 필요한 코드입니다. 원본 분석 알고리즘은 변경하지 않았으며 입력 파일명과 경로만 일반화했습니다.

`data/processed`, `data/release`, `reports`는 로컬 실행 중 사용하는 입력·출력 경로입니다. 실제 데이터·과거 결과·보호 hash는 배포하지 않습니다. `metadata/`에는 기본 스키마와 실행에 필요한 보행망 기준만 남겼습니다. 결과 분포·감사 통계가 섞인 metadata는 제외했습니다.

개발 프로젝트의 대시보드·브라우저 QA·일회성 감사 및 보고서 생성기는 이 공개본 범위에 포함하지 않습니다. 외부 데이터가 필요한 통합 테스트도 기본 테스트에 포함되지 않습니다.

[실행 단계](reproduction.md) · [입력 안내](../data/README.md) · [공개 목록](../PUBLICATION_MANIFEST.md)
