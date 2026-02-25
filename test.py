import cv2
import os
import time
from ultralytics import YOLO

# 1. แก้ปัญหา GUI บน Raspberry Pi
os.environ["QT_QPA_PLATFORM"] = "xcb"

# 2. โหลดโมเดล OpenVINO (แนะนำให้ใช้โมเดลที่ export แบบ imgsz=320)
model_path = '/home/sunlight-lnwza007/Project/model/weights/best_openvino_model'
model = YOLO(model_path, task='detect')

# 3. ตั้งค่ากล้องให้เบาที่สุด
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # ลดความหน่วงของภาพ
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # ภาพบนจอขนาดกำลังดี
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)            # ล็อค FPS กล้อง

print("--- เริ่มการทำงานเวอร์ชันลดภาระ CPU ---")

while cap.isOpened():
    start_time = time.time()
    
    success, frame = cap.read()
    if not success:
        break

    # 4. ส่งประมวลผล (ไม่มีการจำ ID)
    # ใช้ conf=0.4 (ปรับลด/เพิ่มได้) เพื่อลดการกระพริบ ถ้ามั่นใจน้อยกว่านี้จะไม่แสดงผล
    results = model(frame, imgsz=320, conf=0.4, verbose=False)

    # 5. เทคนิคเร่งความเร็ว: วาดกรอบเองด้วย OpenCV (เร็วกว่า r.plot() มาก)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # ดึงพิกัดกรอบ (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # ดึงค่าความมั่นใจ
            conf = float(box.conf[0])
            
            # วาดกรอบสี่เหลี่ยม (สีเขียว, เส้นหนา 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # ใส่ข้อความ "Plate" (ไม่ใส่ ID)
            cv2.putText(frame, f"Plate {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 6. คำนวณและแสดง FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Project Presentation - LPR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()