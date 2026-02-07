import cv2
from ultralytics import YOLO

# ====================================================
# 1. ตั้งค่า Path
# ====================================================
# ลองเปลี่ยน Path กลับไปใช้ตัวที่แม่นที่สุด (Run 4) หรือตัวปัจจุบันก็ได้
MODEL_PATH = r'C:\Users\Admin\Downloads\finetune_info_v2-20260204T064628Z-3-001\runs\weights\best.pt'

# ใส่ Path วิดีโอ
VIDEO_PATH = r'E:\Project\CE\video_test.mp4' 

# ====================================================
# 2. เริ่มทำงาน (Debug Preview Mode)
# ====================================================

print(f"🔄 Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Error: เปิดไฟล์วิดีโอไม่ได้! ({VIDEO_PATH})")
    exit()

print("🎬 Start Debug Preview... (กด 'q' เพื่อออก)")

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("✅ จบวิดีโอ")
        break
    
    frame_count += 1

    # -------------------------------------------------
    # 🧠 ให้ AI ทำนาย (ปรับจูนตรงนี้)
    # -------------------------------------------------
    # 1. conf=0.15 : ลดความมั่นใจลง (ให้มันจับง่ายขึ้น)
    # 2. imgsz=640 : บังคับสเกลภาพให้มาตรฐาน (แก้ปัญหาคลิป 4K แล้ว AI เอ๋อ)
    results = model(frame, verbose=False, conf=0.15, imgsz=640) 

    # เช็ค Log: ถ้าเจอมันจะบอกจำนวน
    det_count = len(results[0].boxes)
    if det_count > 0:
        print(f"Frame {frame_count}: 👀 เจอ {det_count} จุด")

    # -------------------------------------------------
    # 🎨 วาดกรอบ
    # -------------------------------------------------
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls_name = model.names[int(box.cls[0].item())]

            # วาดสี่เหลี่ยม (สีเขียวสะท้อนแสง)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # ตกแต่งป้ายชื่อ
            label = f"{cls_name} {conf:.2f}"
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 5
            
            # พื้นหลังป้ายชื่อ (สีดำโปร่งแสงนิดๆ จะได้อ่านง่าย)
            cv2.rectangle(frame, (x1, y1), c2, (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 0, 0.6, (0, 0, 0), 2)

    # -------------------------------------------------
    # 📺 แสดงผล (ปรับขนาดจอให้พอดีตา)
    # -------------------------------------------------
    # ย่อภาพลงหน่อย เพราะคลิปคุณน่าจะ 4K (มันใหญ่ล้นจอ)
    display_frame = cv2.resize(frame, (1280, 720)) 

    cv2.imshow("AI Debug Preview", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 ปิดโปรแกรม")