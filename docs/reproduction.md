# Reproduction

## 데이터 없이 검증하기

Python 3.12 환경에서 저장소 루트를 작업 디렉터리로 사용합니다.

```sh
python -m pip install -r requirements-docker.txt -c requirements-docker.lock
python -m pytest -q -p no:cacheprovider
```

`tests/`는 외부 데이터·API·데이터베이스 없이 실행하는 단위 테스트입니다. 실제 데이터 기반 통합검사와 대시보드 브라우저 검사는 이 공개본에 포함하지 않았습니다. 공개본에서 실행한 결과는 [Public QA](public-qa.md)에 기록합니다.

## 실제 입력 확보 후 실행하기

[데이터 안내](../data/README.md)에 따라 자료를 확보하고 `config/settings.yaml`의 상대경로에 배치합니다. 원본 이용량·정류소·경계 외에도 ETL이 읽는 `data/processed/sgis_population_2024.parquet`가 필요합니다. 인구 자료의 확보·정규화는 별도 준비 단계입니다.

```sh
cp .env.example .env
# 필요한 API 인증값과 로컬 DB 접속값을 .env에 입력
python scripts/run_pipeline.py --skip-db
```

이 명령은 필수 입력이 준비된 경우에만 실행할 수 있습니다. 후속 분석은 기본 release 테이블 → 노선·정류소/행정동 특징 → 시나리오·Anchor 분석 순서로 선행 산출물이 필요합니다. `scripts/`의 경로 상수를 확인하세요. 모든 분석을 한 번에 수행하는 실행기는 제공하지 않습니다.

보행망·POI 분석에는 별도 네트워크·공식 POI 입력이 필요하며, API 호출에는 인증과 네트워크 연결이 필요합니다. 기본 단위 테스트는 이를 호출하지 않습니다.

## Docker와 PostGIS

`.env`의 `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`를 채운 뒤:

```sh
docker compose config --quiet
docker compose build
docker compose up -d
docker compose exec -T analysis python -m pytest -q -p no:cacheprovider
# 필수 데이터 준비 후 실행. --skip-db를 생략하면 로컬 PostGIS에도 적재
docker compose exec -T analysis python scripts/run_pipeline.py
```

Docker 설정은 제공하지만 이번 공개본 검사에서 이미지 빌드나 실제 PostGIS 적재는 검증하지 않았습니다. 데이터 확보와 선행 테이블 생성 없이 분석 명령의 성공을 보장하지 않습니다.
