from ultralytics import YOLO
import torch
import torch_directml # Library สำคัญสำหรับ AMD บน Windows
import requests
import datetime

# --- [ตั้งค่า Discord Webhook] ---
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1467567408293085214/GNb0F3RawobteeefpTiNauM01w7TCoQklYKwgksNRzEGOCrmggTxe_7JwhrsL62o80_W"

def send_discord_embed(title, description, color, fields=None):
    try:
        data = {
            "embeds": [{
                "title": title,
                "description": description,
                "color": color,
                "fields": fields if fields else [],
                "footer": {"text": "YOLOv26 AMD Trainer 🔴"},
                "timestamp": datetime.datetime.now().astimezone().isoformat()
            }]
        }
        requests.post(DISCORD_WEBHOOK_URL, json=data)
    except: pass

if __name__ == '__main__':
    # 1. เช็คความพร้อมของ AMD GPU ผ่าน DirectML
    dml = torch_directml.device()
    print(f"✅ AMD GPU Detected (DirectML Mode)")
    
    send_discord_embed(
        "🔴 AMD Training Started", 
        "ระบบกำลังใช้ AMD GPU ผ่าน DirectML ในการเทรน YOLOv26n", 
        0xff0000 # สีแดง AMD
    )

    # 2. โหลดโมเดล
    model = YOLO('yolo26n.pt') 

    # 3. เริ่มเทรน
    # หมายเหตุ: สำหรับ DirectML ในปัจจุบัน บางครั้งการระบุ device='dml' 
    # อาจต้องรันผ่านสคริปต์ที่จัดการ backend หรือใช้ device=0 หากระบบมองเห็นเป็นลอจิคัล
    results = model.train(
        data=r'C:\Users\Admin\Downloads\LPR plate.v1i.yolo26\data.yaml', 
        epochs=100,
        imgsz=640,
        batch=16,
        device='cpu',  # <--- ใช้ตัวแปร dml ที่สร้างจาก torch_directml
        name='yolov26_lpr_amd',
        amp=False, 
        workers=0,   # สำหรับ AMD บน Windows แนะนำให้ใช้ 0 เพื่อลดปัญหา Multiprocessing Error
        exist_ok=True
    )

    # 4. Export
    model.export(format='ncnn')
    send_discord_embed("📦 Export Done", "แปลงไฟล์เป็น NCNN เรียบร้อย!", 0x9b59b6)