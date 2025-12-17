🚧 Smart Construction Safety Monitoring System
YOLOv8 & Edge AI(Orion Board) 기반 건설 현장 안전 관제 시스템

📂 폴더 구조 및 설명 (Directory Structure)
프로젝트의 전체 폴더 구조와 각 폴더의 역할입니다.

Plaintext

yolo/
│
├── 📁 src/                 # [핵심] 파이썬 소스 코드 모음
│   ├── danger_zone.py      # 🏗️ 스마트 회전 반경 (굴착기 작업 반경 감지)
│   ├── hazard_detection.py # 🕳️ 환경적 추락 위험 (개구부/난간 미설치 감지)
│   ├── people_count.py     # 👥 작업자/신호수 통합 인원 카운팅
│   ├── custom_zone.py      # 🖱️ 관리자 지정 위험 구역 (마우스로 그리기)
│   └── fall_detection.py   # 📉 전도(넘어짐) 감지 (넘어진걸 인식 못해서 무쓸모)
│
├── 📁 models/              # AI 모델 파일 (.pt, .rknn 등)
│   ├── best.pt             # PC 실행용 YOLO 모델
│   ├── best.rknn           # 오리온 보드(NPU) 실행용 모델
│   └── best.onnx           # 변환 중간 파일
│
├── 📁 videos_input/        # [입력] 테스트할 원본 동영상 (.mp4)
│   ├── Construction_...    # 현장 CCTV 시뮬레이션 영상들
│   └── ...
│
├── 📁 videos_output/       # [출력] AI 감지 결과 영상 (자동 저장됨)
│   ├── output_danger...    # 결과 영상들이 여기에 쌓임
│   └── ...
│
├── 📁 orion_deploy/        # 🚀 오리온 보드 배포용 폴더
│   ├── best.rknn           # NPU 모델
│   ├── test_npu.py         # 보드 테스트 코드
│   └── (보드로 옮길 파일들)
│
├── 📁 training_results/    # 학습 결과 리포트
│   ├── results.csv         # 학습 손실/정확도 그래프 데이터
│   ├── labels.jpg          # 데이터셋 라벨 분포도
│   └── ...
│
├── 📁 images/              # 데이터셋 원본 이미지 및 테스트 이미지
├── 📁 runs/                # YOLO 자동 생성 로그 (학습 가중치, 테스트 결과 이미지 등)
├── 📁 rknn-toolkit2/       # RKNN 모델 변환 툴킷 (라이브러리)
└── 📄 args.yaml            # 학습 설정 파일
💻 기능별 실행 방법 (PC 환경)
src 폴더 내의 코드를 실행하면 videos_input의 영상을 읽어 videos_output에 결과를 저장합니다.

1. 🏗️ 스마트 회전 반경 (Smart Swing Radius)
중장비(굴착기)의 형태를 분석하여 회전 반경을 계산하고, 작업자가 접근하면 경고합니다.

Bash

python src/danger_zone.py
2. 🕳️ 환경적 추락 감지 (Hazard Detection)
현장의 **개구부(Open Hole)**나 난간 미설치(Missing Guardrail) 구역을 찾아내어 시각화합니다.

Bash

python src/hazard_detection.py
3. 👥 통합 인원 카운팅 (People Counting)
작업자와 신호수를 구분하지 않고 전체 인원을 실시간으로 파악합니다.

Bash

python src/people_count.py
4. 🚧 사용자 정의 위험 구역 (Custom Zone)
관리자가 마우스로 직접 위험 구역을 설정합니다.

조작법: 왼쪽 클릭(점 찍기) → s키(시작) → r키(초기화)

⚠️ 참고 사항
입력 영상: 테스트할 영상은 반드시 videos_input 폴더에 있어야 합니다.

결과 확인: 실행 후 videos_output 폴더를 확인하세요. 파일명 앞에 output_이 붙어 저장됩니다.

모델 경로: 모든 코드는 models/best.pt를 기본으로 로드합니다.

네, 원하시는 대로 라벨 리스트를 포함한 통합 To-Do List입니다.

✅ 최종 프로젝트 To-Do List
1. 오리온 보드 이식

[ ] PC의 orion_deploy 폴더(모델, 영상, 코드)를 보드로 전송

[ ] test_npu.py 실행 후 Shape 확인하여 코드 수정

[ ] 보드에서 실행 테스트 완료

2. 성능 측정

[ ] FPS: 보드 실행 시 초당 프레임 측정

[ ] 메모리: 실행 중 RAM 점유율 확인

[ ] 전력 소모: 약 5W 기재 (PC 대비 절감 수치 포함)

3. PPT 제작 및 시연 영상 선별

[ ] 가장 인식 잘 된 Best 영상 3개 선별 (회전반경, 위험구역, 카운팅)

[ ] 발표 자료(PPT) 제작

4. 라벨 종류 전체 정리 (PPT 기재용)

사람 (Person)

0: Worker (작업자 - 안전모 착용)

1: Signal_man (신호수 - 형광 조끼/봉)

중장비 (Heavy Equipment)

2: Excavator (굴착기)

3: Dump Truck (덤프 트럭)

4: Concrete Mixer (레미콘)

5: Road Roller (로드 롤러)

6: Forklift (지게차)

7: Mobile Crane (이동식 크레인)

8: Truck (기타 트럭)

환경적 위험 요소 (Hazards)

11: No Railing (난간 미설치)

13: Opened Hatch (해치 열림)

14: Bad Cover (덮개 불량)

15: Open Hole (개구부/구멍)

16: Bad Board (발판 불량)