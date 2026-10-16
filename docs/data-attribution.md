# Data attribution and unresolved permissions

2026-09-21 Publication Rights Audit 결론을 요약합니다. 상세 근거(원문 조항, 확인 일자, 데이터셋별
현재 노출 범위)는 개발 저장소 내부 감사 문서에 있으며 이 공개 저장소에는 포함하지 않았습니다. 이
문서는 확인한 출처를 기록하는 것이며, 미확인 자료의 공개 허가서가 아닙니다.

## 확인된 출처와 조건

| 자료 | 출처·기준시점 | 확인한 조건 |
|---|---|---|
| 인구 격자 | 국토지리정보원 [통계지도 인구](https://www.data.go.kr/data/15059921/fileData.do); 총인구 500m 2024-10, 총인구 100m·65세 이상 100m/500m 2025-10 | [공공누리 제1유형](https://www.kogl.or.kr/info/licenseType1.do). 출처 링크와 기관·자료명·연도 표시. 집계/공간결합은 본 프로젝트의 처리이며 기관의 공식 검증·추천을 뜻하지 않음 |
| 의료기관 POI | 건강보험심사평가원 [병원정보서비스](https://www.data.go.kr/data/15001698/openapi.do) | 공공누리 제1유형. 이름·좌표 선별 및 공간결합은 프로젝트의 가공 |
| 버스노선·배차 | 국토교통부 [TAGO 버스노선정보](https://www.data.go.kr/data/15098529/openapi.do) | 현재 포털은 이용허락범위 제한 없음. 인증키는 배포 대상이 아님 |
| 행정기관 | 행정안전부 [읍면동 하부행정기관 현황](https://www.data.go.kr/data/15059715/fileData.do) | 이용허락범위 제한 없음. 기관명·주소와 OSM 보완 좌표의 출처를 분리 |
| 횡단보도 | 창원시 [횡단보도 현황](https://www.data.go.kr/data/15106905/fileData.do) | 이용허락범위 제한 없음 |
| 도로현황(도로폭) | 창원시 [도로현황](https://www.data.go.kr/data/15063846/fileData.do) | 이용허락범위 제한 없음. 현재 active 분석에는 미채택 |

공공누리 제1유형은 출처 표시를 조건으로 공유·변경·영리 이용을 허용합니다. 표시에는 출처 페이지
링크를 포함하고, 기관이 프로젝트를 후원하거나 결과를 인증하는 듯한 표현은 쓰지 않습니다. 위 조건은
해당 데이터에만 적용되며, 다른 출처와 결합한 모든 결과에 일괄 적용되지 않습니다.

## OpenStreetMap

보행망과 일부 행정기관 지오코딩: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
원천 데이터는 [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/) 조건을
따릅니다. 데이터베이스 파생물에 해당하는 경우 동일조건·접근 제공 의무를 별도로 검토해야 합니다.
프로젝트 코드 전체나 다른 제공기관 데이터를 ODbL로 재허락한다는 뜻은 아닙니다.

이 공개 저장소에는 Dashboard(지도 UI)가 포함되지 않으므로 온라인/오프라인 배경지도 자체는 노출되지
않습니다. 개발 저장소의 Dashboard는 온라인 배경지도에 OpenStreetMap contributors와 CARTO(타일
제공자) 출처표시를 함께 표시하고, 오프라인 행정경계 배경에도 별도 출처표시를 유지합니다.

## 허가 확인 전 공개 보류

- **이용량 CSV(2025-08):** 실제 전달·다운로드 조건 및 권리자 확인이 필요합니다.
- **정류소 master:** 로컬 스냅샷(2024-12-31)과 공식 현재 배포본(2025-12-31)의 연결 증빙이 필요합니다.
- **SGIS 2024 인구:** API 사이트 활용과 정적 JSON/DB 재배포 허가는 구분됩니다. 수집 당시 적용 약관과
  정적 배포 범위를 확인해야 합니다.
- **VWorld 행정동 경계:** 자료별 다운로드 조건을 확보해야 합니다. 좌표변환·단순화·병합이 원천 조건을
  없애지 않습니다.

위 4개 자료가 해결되기 전까지 이들에서 파생된 값(정류소·행정동 지오메트리, 인구 지표, 이용량,
Dashboard 전체)은 이 공개 저장소에 포함하지 않습니다.

## 코드/라이브러리

이 공개본에는 Dashboard와 함께 벤더링된 프론트엔드 자산(Leaflet 등)이 포함되지 않습니다. Python
의존성의 라이선스는 각 패키지의 PyPI 배포 조건을 따르며 `requirements*.txt`에 버전을 명시합니다.
프로젝트 코드 자체의 LICENSE는 아직 선택되지 않았습니다 — 자세한 내용은
[PUBLICATION_MANIFEST.md](../PUBLICATION_MANIFEST.md)의 LICENSE_PENDING 항목을 참고하세요.
