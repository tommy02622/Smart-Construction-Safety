import cv2
import numpy as np
import os
from ultralytics import YOLO

# --- 1. 경로 설정 ---
# 모델 경로
model_path = r"C:\Users\gunhu\dev\yolo\models\best.pt"

# 입력 동영상 (여기만 바꾸면 저장 파일명도 알아서 바뀜!)
video_path = r"C:\Users\gunhu\dev\yolo\videos_input\Construction_Site_CCTV_Video_Generation.mp4"

# 파일 경로가 맞는지 확인
if not os.path.exists(video_path):
    print(f"❌ 에러: 입력 파일을 찾을 수 없습니다 -> {video_path}")
    exit()

# --- ★ 핵심: 파일명 자동 생성 로직 ★ ---
# 1) 입력 파일 경로에서 '파일명'만 떼어냄
input_filename = os.path.basename(video_path)

# 2) 확장자(.mp4)를 떼어냄
file_name_only = os.path.splitext(input_filename)[0]

# 3) 앞에 'output_zone_'을 붙여서 새로운 이름 생성 (구분하기 쉽게 zone 추가)
output_filename = f"output_zone_{file_name_only}.mp4"

# 4) 저장 폴더와 합치기
output_dir = r"C:\Users\gunhu\dev\yolo\videos_output"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, output_filename)
# --------------------------------------

# 2. 모델 로드
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ 영상을 열 수 없습니다.")
    exit()

# 화면 크기 설정 (HD)
width, height = 1280, 720
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))

# --- 3. 마우스 이벤트 설정 (그리기 로직) ---
points = [] # 클릭한 좌표들을 저장할 리스트

def draw_polygon(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN: # 왼쪽 클릭: 점 추가
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN: # 오른쪽 클릭: 초기화
        points = []
        print("🔄 구역 초기화됨. 다시 그리세요.")

# 윈도우 생성 및 콜백 함수 연결
cv2.namedWindow("Set Danger Zone")
cv2.setMouseCallback("Set Danger Zone", draw_polygon)

print("🎨 [설정 모드] 위험 구역을 마우스로 클릭해서 그리세요.")
print("   - 왼쪽 클릭: 점 추가")
print("   - 오른쪽 클릭: 다시 그리기")
print("   - 's' 키: 설정 완료 및 감지 시작")
print(f"💾 저장 예정: {output_filename}")

# 첫 프레임 읽기 (구역 설정을 위해)
ret, first_frame = cap.read()
if not ret:
    print("❌ 영상을 읽을 수 없습니다.")
    exit()

first_frame = cv2.resize(first_frame, (width, height))

# --- 4. 구역 설정 루프 (s키 누를 때까지 대기) ---
zone_polygon = []
while True:
    temp_frame = first_frame.copy()
    
    # 찍은 점들을 잇는 선 그리기
    if len(points) > 0:
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        # 다각형 그리기 (아직 확정 전이라 노란색)
        cv2.polylines(temp_frame, [pts], True, (0, 255, 255), 2)
        
        # 각 점 표시
        for p in points:
            cv2.circle(temp_frame, p, 5, (0, 0, 255), -1)

    # 안내 문구
    cv2.putText(temp_frame, "Click points to define ZONE. Press 's' to START.", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Set Danger Zone", temp_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'): # Start
        if len(points) > 2: # 점이 최소 3개는 있어야 면적이 됨
            zone_polygon = np.array(points, np.int32)
            print("✅ 구역 설정 완료! 감지를 시작합니다.")
            break
        else:
            print("⚠️ 점을 3개 이상 찍어야 합니다!")
    elif key == ord('q'):
        exit()

cv2.destroyWindow("Set Danger Zone")

# --- 5. 실시간 감지 루프 ---
print("🚀 실시간 감시 중...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.resize(frame, (width, height))
    
    # YOLO 추론
    results = model(frame, conf=0.25, verbose=False)
    
    intrusion_detected = False
    
    # 구역 그리기 (평소엔 초록색, 침입 시 빨간색)
    zone_color = (0, 255, 0) 
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            
            # 사람(0, 1)만 감시
            if cls in [0, 1]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # ★ 핵심 로직: 사람의 '발' 위치 계산 ★
                feet_x = (x1 + x2) // 2
                feet_y = y2 
                
                # 점이 다각형 안에 있는지 검사
                result = cv2.pointPolygonTest(zone_polygon, (feet_x, feet_y), False)
                
                if result >= 0: # 내부에 있음! (침입)
                    intrusion_detected = True
                    # 사람 박스 빨간색
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "WARNING!", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # 안전하면 초록색
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 침입 발생 시 구역 색상 변경 및 경고창
    if intrusion_detected:
        zone_color = (0, 0, 255) # 빨강
        cv2.putText(frame, "DANGER ZONE INTRUSION!", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        # 테두리 굵게
        cv2.polylines(frame, [zone_polygon], True, zone_color, 5)
        
        # 내부를 반투명하게 칠하기
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_polygon], zone_color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
    else:
        # 평소엔 얇은 초록 테두리
        cv2.polylines(frame, [zone_polygon], True, zone_color, 2)
        cv2.putText(frame, "SAFE ZONE MONITORING", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Custom Zone Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ 저장 완료: {save_path}")