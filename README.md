# Changwon Transit GIS Screening

창원시 대중교통의 관측 이용·공급·인구·공간 조건을 분석해, 추가 현장검토가 필요한 지역과 판단의 한계를 함께 확인하는 GIS 프로젝트입니다.

공모전 팀 프로젝트의 DRT 후보지 분석에서 출발했으며, 이후 개인 확장에서 데이터 파이프라인과 공간·불확실성 진단을 보완했습니다. 팀 결과 전체를 개인 성과로 주장하지 않습니다.

이 저장소는 **코드·단위 테스트·방법론을 공개하는 버전**입니다. 실제 데이터, 분석 결과표, 데이터가 내장된 대시보드와 화면 캡처는 포함하지 않습니다.

## 주요 기능

- 원본 행·중복 관측·NULL을 보존하는 이용량 변환과 정류소 매칭
- 행정동·정류소별 교통·인구 지표 생성 및 PostGIS 적재·검증
- 정책 가중치별 탐색적 상대점수와 순위 민감도 분석
- Anchor·POI·보행 네트워크 접근성 분석
- 관측 coverage, 지표별 기여, 공간 자기상관·결측 민감성 진단

정책 담당자·GIS 실무자가 지역을 좁힌 뒤 정류소·노선·관측 근거를 확인하고 현장조사로 연결하는 흐름을 지원합니다. 분석 점수만으로 DRT 도입을 결정하지 않습니다.

## 구성

```text
src/          수집·변환·분석·검증·출력 함수
scripts/      데이터 확보 후 실행하는 CLI
tests/        외부 데이터 없이 실행하는 단위 테스트
config/       상대경로와 분석 설정
sql/          PostGIS 테이블·인덱스·품질 쿼리
metadata/     기본 출력 스키마·보행망 설정
data/         데이터 획득 안내
docs/         구조·방법론·재현·출처·검증 범위
```

## 실행

Python 3.12 환경에서 의존성을 설치하고 단위 테스트를 실행합니다.

```sh
python -m pip install -r requirements-docker.txt -c requirements-docker.lock
python -m pytest -q -p no:cacheprovider
```

Source datasets are not redistributed. Reproduction requires obtaining the listed datasets from their original providers and placing them in the documented paths.

실제 분석 CLI는 [데이터 안내](data/README.md)에 따른 입력과 선행 산출물이 필요합니다. 공개 폴더만으로 전체 분석·대시보드를 재현할 수는 없습니다. [실행 단계](docs/reproduction.md)와 [구조](docs/architecture.md)를 참고하세요.

## 해석 범위

- 2025-08-18~25의 8일 관측과 2026년 공급 snapshot 사이에 시점 차이가 있습니다.
- 관측 이용은 잠재 수요가 아니며, 미관측을 이용량 0으로 읽지 않습니다.
- 가중치는 정책적 선호입니다. 높은 점수·안정성이 서비스 필요도나 정책 타당성을 입증하지 않습니다.
- Anchor는 후속 현장조사 시작점이며, Service Zone은 실제 주민 보행권을 보장하지 않습니다.
- 목적지별 실제 통행시간·환승, 현장·이용자 검증과 운영 실행 가능성 검증은 완료되지 않았습니다.

[방법론](docs/methodology.md) · [검증 범위](docs/validation.md) · [공개본 검사 결과](docs/public-qa.md)

## 데이터와 권리

데이터 출처와 재배포 조건은 [출처 안내](docs/data-attribution.md)에 기록했습니다. 해당 문서의 권리 상태는 기존 검토 기록이며 이번 정리에서 새로 확인한 허가가 아닙니다. 프로젝트 코드 라이선스는 아직 지정하지 않았습니다(`LICENSE_PENDING`).

포함 파일과 제외 기준은 [공개 목록](PUBLICATION_MANIFEST.md)을 참고하세요.
