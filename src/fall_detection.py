import cv2
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO(r"C:\Users\gunhu\dev\yolo\best.pt")

# ★ 넘어짐 영상 경로 고정
video_path = r"C:\Users\gunhu\dev\yolo\Construction_Worker_Slips_and_Falls.mp4"
cap = cv2.VideoCapture(video_path)

# 2. 감도 설정 (극한의 감도)
FALL_RATIO_THRESHOLD = 0.8  # 이 비율보다 가로가 조금만 길어도 넘어짐으로 간주
WORKER_CLASSES = [0, 1]     # worker, signal_man

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_fall_final.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

print("🚀 넘어짐 감지 최종 테스트 (Augment ON, Conf 0.1)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # ★ 핵심: augment=True (누운 사람 찾기 위해 내부적으로 이미지를 돌려봄 - 속도는 느려짐)
    # ★ 핵심: conf=0.1 (10%만 확신해도 박스 그림)
    results = model(frame, conf=0.1, augment=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            
            if cls in WORKER_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                
                if height == 0: continue
                aspect_ratio = width / height

                # 상태 판단
                if aspect_ratio > FALL_RATIO_THRESHOLD:
                    color = (0, 0, 255) # 빨강 (넘어짐)
                    status = f"FALL! ({aspect_ratio:.2f})"
                    # 시각 효과
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                else:
                    color = (0, 255, 0) # 초록 (정상)
                    status = f"Normal ({aspect_ratio:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, status, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Final Fall Check', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ 넘어짐 테스트 완료.")