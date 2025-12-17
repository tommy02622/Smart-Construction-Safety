import cv2
import os
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO(r"C:\Users\gunhu\dev\yolo\models\best.pt")

# 2. 입력 동영상 (여기만 바꾸면 저장 파일명도 알아서 바뀜!)
video_path = r"C:\Users\gunhu\dev\yolo\videos_input\14911573_3840_2160_60fps.mp4"

# 파일 경로 확인
if not os.path.exists(video_path):
    print(f"❌ 에러: 입력 파일을 찾을 수 없습니다 -> {video_path}")
    exit()

# --- ★ 핵심: 파일명 자동 생성 로직 ★ ---
# 1) 입력 파일명 추출
input_filename = os.path.basename(video_path)

# 2) 확장자 제거
file_name_only = os.path.splitext(input_filename)[0]

# 3) 'output_people_' 접두사 붙여서 새 이름 생성
output_filename = f"output_people_{file_name_only}.mp4"

# 4) 저장 경로 합치기
output_dir = r"C:\Users\gunhu\dev\yolo\videos_output"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, output_filename)
# --------------------------------------

cap = cv2.VideoCapture(video_path)

# 감지할 대상 (0: worker, 1: signal_man)
PERSON_CLASSES = [0, 1] 

# 화면 크기 조정
NEW_WIDTH = 1280
NEW_HEIGHT = 720

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(save_path, fourcc, 30.0, (NEW_WIDTH, NEW_HEIGHT))

print(f"🚀 인원 통합 표시 시작...")
print(f"📂 입력: {input_filename}")
print(f"💾 저장: {output_filename}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 크기 조정
    frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT))

    results = model(frame, conf=0.25, verbose=False)
    
    total_people = 0
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            
            if cls in PERSON_CLASSES:
                total_people += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 라벨 통일
                label = "Person"
                color = (0, 255, 0) 
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                
                cv2.putText(frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # UI 표시
    cv2.rectangle(frame, (20, 20), (350, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Total Person: {total_people}", (35, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow('Unified People Counting', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ 완료! 파일이 여기에 저장되었습니다: {save_path}")