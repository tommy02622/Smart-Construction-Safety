import cv2
import numpy as np
import math
import os
from ultralytics import YOLO

# 1. 모델 설정
model = YOLO(r"C:\Users\gunhu\dev\yolo\models\best.pt")

# ★ 입력 비디오 경로 (여기만 바꾸면 저장 파일명도 알아서 바뀜!)
video_path = r"C:\Users\gunhu\dev\yolo\videos_input\Construction_Site_Accident_Video.mp4"

# 파일 경로 확인
if not os.path.exists(video_path):
    print(f"❌ 에러: 입력 파일을 찾을 수 없습니다 -> {video_path}")
    exit()

# --- ★ 핵심: 파일명 자동 생성 로직 ★ ---
# 1) 입력 파일명 추출
input_filename = os.path.basename(video_path)

# 2) 확장자 제거
file_name_only = os.path.splitext(input_filename)[0]

# 3) 'output_radius_' 접두사 붙여서 새 이름 생성
output_filename = f"output_radius_{file_name_only}.mp4"

# 4) 저장 경로 합치기
output_dir = r"C:\Users\gunhu\dev\yolo\videos_output"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, output_filename)
# --------------------------------------

cap = cv2.VideoCapture(video_path)

# 2. 설정값 (굴착기, 트럭 등)
# 모델마다 번호가 다를 수 있으니 model.names로 꼭 확인하세요!
HEAVY_MACHINES = [2, 3, 4, 5, 6, 7, 8] 
CLASS_WORKER = 0 # 작업자

# 결과 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 영상 크기 가져오기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))

print("🚀 스마트 회전 반경 감지 시스템 시작...")
print(f"📂 입력: {input_filename}")
print(f"💾 저장: {output_filename}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 추론 실행
    results = model(frame, conf=0.25, verbose=False)
    
    danger_zones = [] 

    # --- 1단계: 중장비 모양 분석해서 '회전 반경' 계산 ---
    if results[0].masks is not None:
        masks = results[0].masks.xy
        boxes = results[0].boxes

        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            if cls in HEAVY_MACHINES:
                # 1. 마스크(윤곽선) 가져오기
                contour = np.array(masks[i], dtype=np.int32)
                
                # 2. 장비의 무게중심(Center) 구하기
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # 3. 중심에서 가장 먼 점(Max Distance) 찾기 = 팔 길이
                    max_dist = 0
                    for point in contour:
                        # [수정된 부분] point[0]이 아니라 point 자체를 가져옴
                        px, py = point 
                        dist = math.sqrt((px - cx)**2 + (py - cy)**2)
                        if dist > max_dist:
                            max_dist = dist
                    
                    # 4. 여유 버퍼 살짝 줘서 반지름 확정
                    radius = int(max_dist + 20)
                    danger_zones.append((cx, cy, radius))

                    # --- 시각화 ---
                    # 위험 반경 (빨간색 투명 원)
                    overlay = frame.copy()
                    cv2.circle(overlay, (cx, cy), radius, (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                    
                    # 테두리 및 중심점
                    cv2.circle(frame, (cx, cy), radius, (0, 0, 255), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                    
                    label_name = model.names[cls]
                    cv2.putText(frame, f"{label_name} Radius", (cx - 40, cy - radius - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- 2단계: 작업자 침범 확인 ---
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == CLASS_WORKER:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w_cx, w_cy = (x1 + x2) // 2, y2 # 작업자 발 위치

                is_danger = False
                for (d_cx, d_cy, radius) in danger_zones:
                    dist_to_machine = math.sqrt((w_cx - d_cx)**2 + (w_cy - d_cy)**2)
                    
                    if dist_to_machine < radius:
                        is_danger = True
                        break
                
                if is_danger:
                    # 🚨 위험 경고
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "DANGER!", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    # ✅ 안전
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Smart Swing Radius', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ 완료! 파일이 여기에 저장되었습니다: {save_path}")