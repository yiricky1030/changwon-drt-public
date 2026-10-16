# Publication manifest

검사일: 2026-10-16. 상태: **PUBLIC_TREE_READY** — 코드·문서 공개용 폴더 기준입니다. 데이터·대시보드는 공개 범위에 포함하지 않습니다.

## INCLUDED

| 경로 | 파일 수 | 용도 |
|---|---:|---|
| `config` | 1 | 상대경로 설정 |
| `data` | 1 | 획득 안내문만 포함 |
| `docs` | 7 | 구조·방법론·재현·출처·검증 |
| `metadata` | 3 | 기본 스키마·보행망 기준 |
| `root` | 10 | README·환경 예제·의존성·Docker·제외 정책 |
| `scripts` | 24 | 실제 입력이 필요한 CLI |
| `sql` | 5 | PostGIS 스키마·검증 |
| `src` | 50 | 재사용할 분석·ETL 함수 |
| `tests` | 8 | 외부 데이터 없는 단위 테스트 |

## EXCLUDED

- `reports/`: 중복 감사 문서, 피드백 이력, 임시 작업 목록, 결과표, 로그·JUnit XML·화면 캡처
- `data/raw`, `data/interim`, `data/processed`, `data/release`: 원천·중간·생산 데이터 전체
- `preview/`와 대시보드 생성·브라우저 QA 코드: 데이터 내장 화면과 개발 환경에 종속된 도구
- 결과 분포·집계 통계가 섞인 metadata, 실데이터 기반 통합 테스트와 일회성 보고서 생성기
- 실행하지 않은 현장·사용자 평가 템플릿, 개인 면접 메모, 기존 공개 준비 감사 자료
- `.git`, `.env`, 가상환경, 캐시, bytecode, IDE 설정, 백업·압축 파일

원본의 `reports/final_non_gis_usability_audit.md`도 내부 검토 기록으로 제외했습니다. 실제 사용자 검증 결과로 공개하지 않습니다.

## DATA_NOT_REDISTRIBUTED / RIGHTS_REVIEW

공식 데이터도 이번 폴더에는 재배포하지 않습니다. [데이터 안내](data/README.md)와 [출처·기존 권리 검토](docs/data-attribution.md)를 참고하세요. 권리 조건을 이번 정리에서 새로 판정하지 않았습니다.

## DASHBOARD_WITHHELD

이용량·인구·경계의 파생 값과 좌표가 내장된 대시보드·화면 캡처는 포함하지 않습니다. 공개 코드로 전체 대시보드를 재현할 수 있다는 주장을 하지 않습니다.

## LICENSE_PENDING

프로젝트 코드 라이선스는 권리자가 지정하기 전까지 추가하지 않습니다. 출처 안내는 사용 허가를 대신하지 않습니다. 팀 프로젝트와 이후 개인 확장 기여는 [README](README.md)에서 구분합니다.

## 검사와 실행 범위

[공개본 검사 결과](docs/public-qa.md) · [실행 안내](docs/reproduction.md)

원본 저장소는 수정하지 않았으며 Git 이력·commit·remote를 복사하지 않았습니다. 이 폴더의 내용만 새 저장소의 루트에 올리는 용도입니다.
