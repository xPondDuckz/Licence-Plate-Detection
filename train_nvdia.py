import os
import time
import torch
import platform
import getpass
import requests
import datetime
from ultralytics import YOLO

# =========================================================
# 1. ตั้งค่าพื้นฐาน (Configuration)
# =========================================================
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1467567408293085214/GNb0F3RawobteeefpTiNauM01w7TCoQklYKwgksNRzEGOCrmggTxe_7JwhrsL62o80_W"
MODEL_VARIANT = 'yolo26n.pt'  # ใช้ YOLOv26 Nano เป็นหลัก
PROJECT_NAME = 'yolov26_lpr_advanced'

# สีสำหรับ Discord Embed
COLOR_START   = 0x2ecc71  # เขียว
COLOR_UPDATE  = 0x3498db  # ฟ้า
COLOR_FINISH  = 0xf1c40f  # เหลือง
COLOR_SUCCESS = 0x9b59b6  # ม่วง
COLOR_ERROR   = 0xe74c3c  # แดง

# ตัวแปรสำหรับคำนวณเวลา
start_training_time = None

# =========================================================
# 2. ฟังก์ชันเสริม (Helper Functions)
# =========================================================
def get_system_info():
    """ดึงข้อมูล User และชื่อเครื่อง"""
    return f"{getpass.getuser()}@{platform.node()}"

def get_training_status(mAP50):
    """วิเคราะห์ระดับความแม่นยำเพื่อบอกสถานะ"""
    if mAP50 < 0.3: return "🔴 กำลังเรียนรู้พื้นฐาน (Low)"
    if mAP50 < 0.6: return "🟡 เริ่มแยกแยะได้แล้ว (Medium)"
    if mAP50 < 0.8: return "🟢 แม่นยำสูง (High)"
    return "🔥 พร้อมรบ! (Ready for Deployment)"

def send_discord_embed(title, description, color, fields=None):
    """ส่งข้อความแจ้งเตือนเข้า Discord"""
    try:
        payload = {
            "embeds": [{
                "title": title,
                "description": description,
                "color": color,
                "fields": fields if fields else [],
                "footer": {"text": f"User: {get_system_info()} | YOLOv26n System 🚀"},
                "timestamp": datetime.datetime.now().astimezone().isoformat()
            }]
        }
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10).raise_for_status()
    except Exception as e:
        print(f"Discord Error: {e}")

# =========================================================
# 3. ระบบ Callbacks (ติดตามการเทรน)
# =========================================================
def on_train_start(trainer):
    global start_training_time
    start_training_time = time.time()
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    fields = [
        {"name": "🖥️ Device", "value": f"`{gpu_name}`", "inline": True},
        {"name": "📁 Project", "value": f"`{PROJECT_NAME}`", "inline": True},
        {"name": "📦 Batch Size", "value": f"`{trainer.args.batch}`", "inline": True}
    ]
    send_discord_embed("🚀 Training Started", "เริ่มกระบวนการเทรน YOLOv26n", COLOR_START, fields)

def on_train_epoch_end(trainer):
    epoch = trainer.epoch + 1
    total_epochs = trainer.args.epochs
    
    # คำนวณเวลาที่เหลือ (ETA)
    elapsed = time.time() - start_training_time
    avg_time = elapsed / epoch
    eta = str(datetime.timedelta(seconds=int((total_epochs - epoch) * avg_time)))

    # ดึงค่าความแม่นยำ mAP
    metrics = trainer.metrics
    map50 = metrics.get('metrics/mAP50(B)', 0.0)
    map95 = metrics.get('metrics/mAP50-95(B)', 0.0)
    
    # ดึงค่า Loss
    loss_box = trainer.loss_items[0] if trainer.loss_items is not None else 0.0
    loss_cls = trainer.loss_items[1] if trainer.loss_items is not None else 0.0

    status = get_training_status(map50)

    fields = [
        {"name": "🔄 Epoch", "value": f"`{epoch}/{total_epochs}`", "inline": True},
        {"name": "⏳ เวลาที่เหลือ (ETA)", "value": f"`{eta}`", "inline": True},
        {"name": "📊 สถานะปัจจุบัน", "value": f"**{status}**", "inline": False},
        {"name": "🎯 mAP50 (แม่นยำ)", "value": f"`{map50:.4f}`", "inline": True},
        {"name": "📐 mAP50-95 (เป๊ะ)", "value": f"`{map95:.4f}`", "inline": True},
        {"name": "📉 Box Loss", "value": f"`{loss_box:.4f}`", "inline": True},
        {"name": "📉 Class Loss", "value": f"`{loss_cls:.4f}`", "inline": True}
    ]
    
    # แจ้งเตือนทุกรอบ (ถี่ตามต้องการ)
    send_discord_embed("📊 Progress Update", "วิเคราะห์ผลการเรียนรู้รายรอบ", COLOR_UPDATE, fields)

def on_train_end(trainer):
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_training_time)))
    send_discord_embed("✅ Training Finished!", f"เทรนเสร็จสิ้นเรียบร้อย!\nใช้เวลาทั้งหมด: `{total_time}`", COLOR_FINISH)

# =========================================================
# 4. ส่วนการทำงานหลัก (Main Process)
# =========================================================
if __name__ == '__main__':
    # ตรวจสอบ GPU
    if torch.cuda.is_available():
        print(f"🔥 Found NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ ไม่พบ NVIDIA GPU! โปรดลง CUDA และ PyTorch GPU Version")
        send_discord_embed("❌ Error", "ระบบไม่สามารถหา NVIDIA GPU เจอ", COLOR_ERROR)
        exit()

    # โหลดโมเดล
    print(f"--- Loading {MODEL_VARIANT} ---")
    model = YOLO(MODEL_VARIANT)

    # ลงทะเบียน Callbacks
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    # เริ่มการเทรน
    try:
        model.train(
            data=r'G:\Project\Data\pengsiri\data.yaml',
            epochs=100,
            imgsz=640,
            batch=16,
            device=0,
            name=PROJECT_NAME,
            exist_ok=True,
            workers=4
        )
    except Exception as e:
        send_discord_embed("❌ Training Crashed!", f"เกิดข้อผิดพลาดระหว่างเทรน:\n`{str(e)}`", COLOR_ERROR)
        exit()

    # การ Export ไปยัง Raspberry Pi (NCNN)
    print("--- Exporting to NCNN ---")
    try:
        # บันทึกโมเดลเป็น NCNN
        export_path = model.export(format='ncnn')
        send_discord_embed("📦 Export Successful!", f"แปลงไฟล์เป็น NCNN เรียบร้อย!\nพร้อมนำไปใช้บน Raspberry Pi แล้ว 🎉", COLOR_SUCCESS)
        print(f"Export Success: {export_path}")
    except Exception as e:
        send_discord_embed("❌ Export Failed", f"เกิดข้อผิดพลาดขณะ Export NCNN:\n`{str(e)}`", COLOR_ERROR)

    print("!!! All Done !!!")