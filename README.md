# Smart Construction Safety

YOLOv8 기반 건설 현장 위험 요소 탐지 프로젝트입니다. 산업 현장 이미지와 영상에서 개구부, 불량 덮개, 작업자 및 중장비 주변 상황을 확인하는 PC용 탐지 프로그램을 구현했습니다.

2025 탄소중립 INNOVATION ACADEMY 5기 최우수상 수상작입니다.

## 담당 작업

- 프로젝트 구조 설계와 안전 이미지 데이터셋 탐색 및 구성
- 데이터 라벨링, YOLOv8 탐지·세그멘테이션 모델 학습 및 튜닝
- 학습 모델을 이미지와 영상에 적용하는 프로그램 구현 및 결과 확인
- Orion Board용 RKNN 모델 변환과 보드 탑재 시도

발표자료 제작과 최종 발표를 제외한 기술 작업을 담당했습니다.

## 저장소 구성

| 경로 | 내용 |
| --- | --- |
| [`src/`](src/) | 위험 구역, 개구부, 인원 및 중장비 관련 PC 추론 코드 |
| [`models/`](models/) | 학습 모델과 ONNX/RKNN 변환 파일 |
| [`training_results/`](training_results/) | 학습 로그 및 라벨 분포 |
| [`runs/segment/`](runs/segment/) | 이미지·영상 탐지 결과 |
| [`videos_input/`](videos_input/) | 테스트 입력 영상 |
| [`videos_output/`](videos_output/) | PC 추론 결과 영상 |
| [`orion_deploy/`](orion_deploy/) | 보드 탑재를 위해 준비한 RKNN 파일 |

## 확인 가능한 결과

- [위험 요소 탐지 이미지](runs/segment/predict3/H-220805_A26_N-03_001_0001.jpg)
- [라벨 분포](training_results/labels.jpg)
- [PC 영상 추론 출력](videos_output/output_hazard.mp4)
- [학습 결과 CSV](training_results/results.csv)

## 실행 범위

`src/`의 스크립트는 PC 환경에서 작성되었으며 모델·영상 경로가 당시 로컬 경로로 지정되어 있습니다. 다른 컴퓨터에서 실행하려면 각 스크립트의 경로를 현재 저장소 위치에 맞게 변경해야 합니다. 공개 저장소에는 전체 학습 데이터셋이 포함되어 있지 않습니다.

`.rknn` 파일은 변환 산출물이지만, Orion Board에서의 최종 실행·FPS·전력 측정 완료를 뜻하지 않습니다. 온디바이스 동작은 탑재를 시도한 단계로 구분합니다.
