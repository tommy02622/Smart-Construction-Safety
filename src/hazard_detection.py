import cv2
import numpy as np
import os
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO(r"C:\Users\gunhu\dev\yolo\models\best.pt")

# 2. 입력 동영상 (여기만 바꾸면 저장 파일명도 알아서 바뀜!)
video_path = r"C:\Users\gunhu\dev\yolo\videos_input\Construction_Site_Danger_Revealed.mp4"

# 파일 경로가 맞는지 확인
if not os.path.exists(video_path):
    print(f"❌ 에러: 입력 파일을 찾을 수 없습니다 -> {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)

# --- ★ 핵심: 파일명 자동 생성 로직 ★ ---
# 1) 입력 파일 경로에서 '파일명'만 떼어냄 (예: Construction_Site_Hole_Revealed.mp4)
input_filename = os.path.basename(video_path)

# 2) 확장자(.mp4)를 떼어냄 (예: Construction_Site_Hole_Revealed)
file_name_only = os.path.splitext(input_filename)[0]

# 3) 앞에 'output_'을 붙여서 새로운 이름 생성
output_filename = f"output_{file_name_only}.mp4"

# 4) 저장 폴더와 합치기
output_dir = r"C:\Users\gunhu\dev\yolo\videos_output"
save_path = os.path.join(output_dir, output_filename)
# --------------------------------------

# 감지할 클래스 (11:난간없음, 15:개구부 등)
HAZARD_CLASSES = [11, 13, 14, 15, 16]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 영상 크기 가져오기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 저장 경로(save_path)로 설정
out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))

print(f"🚀 위험 요소 감지 시작...")
print(f"📂 입력: {input_filename}")
print(f"💾 저장: {output_filename} (덮어쓰기 방지됨)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    results = model(frame, conf=0.25, verbose=False)
    
    if results[0].masks is not None:
        masks = results[0].masks.xy
        boxes = results[0].boxes

        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            
            if cls in HAZARD_CLASSES:
                contour = np.array(masks[i], dtype=np.int32)
                
                overlay = frame.copy()
                cv2.drawContours(overlay, [contour], -1, (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                
                cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
                
                label_name = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.putText(frame, f"HAZARD: {label_name}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('Open Hole & Railing Detection', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ 완료! 파일이 여기에 저장되었습니다: {save_path}")