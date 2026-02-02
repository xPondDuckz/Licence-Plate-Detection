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
DISCORD_WEBHOOK_URL = "ใส่_WEBHOOK_URL_ของคุณที่นี่"
DATA_YAML_PATH = r'G:\Project\Data\pengsiri\data.yaml'

# Path ไฟล์เก่า (ถ้ามี)
OLD_WEIGHTS_PATH = r'G:\Project\Licence-Plate-Detection\runs\detect\yolov26_lpr_smart_guard\weights\best.pt'
DEFAULT_MODEL = 'yolo26n.pt'
PROJECT_NAME = 'yolov26_lpr_auto_pilot' # เปลี่ยนชื่อให้สื่อความหมาย

# สีสถานะ Discord
COLOR_HEALTHY = 0x2ecc71  # เขียว (ปกติ)
COLOR_WARNING = 0xf1c40f  # เหลือง (เริ่มนิ่ง)
COLOR_DANGER  = 0xe74c3c  # แดง (Overfitting)
COLOR_FINISH  = 0x9b59b6  # ม่วง (เสร็จแล้ว)

start_training_time = None

# =========================================================
# 2. ฟังก์ชันแจ้งเตือน (Notification System)
# =========================================================
def send_discord_image(file_path, caption=""):
    if not os.path.exists(file_path): return
    try:
        with open(file_path, 'rb') as f:
            requests.post(DISCORD_WEBHOOK_URL, data={"content": caption}, files={"file": f}, timeout=20)
    except: pass

def send_discord_embed(title, description, color, fields=None):
    try:
        payload = {
            "embeds": [{
                "title": title,
                "description": description,
                "color": color,
                "fields": fields if fields else [],
                "footer": {"text": f"AutoPilot AI | {getpass.getuser()}@{platform.node()}"},
                "timestamp": datetime.datetime.now().astimezone().isoformat()
            }]
        }
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    except: pass

# =========================================================
# 3. ระบบ Callbacks (Smart Monitor + Time Calculation)
# =========================================================
def on_train_start(trainer):
    global start_training_time
    start_training_time = time.time()
    
    model_name = "Old best.pt (Fine-tuning)" if os.path.exists(OLD_WEIGHTS_PATH) else "New YOLO26n"
    
    send_discord_embed(
        "🚀 Auto-Pilot Started", 
        f"Base Model: **{model_name}**\nโหมด: **ปล่อยเทรนยาว (Auto-Stop on Overfit)**", 
        COLOR_HEALTHY
    )

def on_train_epoch_end(trainer):
    epoch = trainer.epoch + 1
    total_epochs = trainer.args.epochs
    
    # --- [ส่วนคำนวณเวลาแบบใหม่] ---
    elapsed_seconds = time.time() - start_training_time
    avg_time_per_epoch = elapsed_seconds / epoch
    remaining_epochs = total_epochs - epoch
    eta_seconds = remaining_epochs * avg_time_per_epoch
    
    # แปลงเป็นเวลาที่เหลือ (เช่น 1:30:00)
    eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
    
    # คำนวณ "เวลาที่จะเสร็จ" (Timestamp)
    finish_time = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)
    finish_time_str = finish_time.strftime("%H:%M น.") # เช่น 14:30 น.
    # -----------------------------

    metrics = trainer.metrics
    map50 = metrics.get('metrics/mAP50(B)', 0.0)
    train_box_loss = trainer.loss_items[0].item()

    # Logic ตรวจสุขภาพโมเดล (เหมือนเดิม)
    status_msg = "🟢 ปกติ (Healthy)"
    status_color = COLOR_HEALTHY
    if epoch > 50 and map50 < 0.5 and train_box_loss < 0.5:
        status_msg = "🔴 เริ่มจำข้อสอบ (Overfitting Alert)"
        status_color = COLOR_DANGER
    elif map50 < 0.3 and epoch > 30:
        status_msg = "🟡 เรียนรู้ช้า (Underfitting)"
        status_color = COLOR_WARNING

    fields = [
        {"name": "🔄 Epoch", "value": f"`{epoch}/{total_epochs}`", "inline": True},
        {"name": "🎯 mAP50", "value": f"`{map50:.4f}`", "inline": True},
        {"name": "📉 Train Loss", "value": f"`{train_box_loss:.4f}`", "inline": True},
        
        # ช่องใหม่: เวลาที่คาดว่าจะเสร็จ
        {"name": "⏰ คาดว่าจะเสร็จตอน", "value": f"**`{finish_time_str}`** (อีก `{eta_str}`)", "inline": False},
        
        {"name": "🩺 Health", "value": f"{status_msg}", "inline": False}
    ]
    
    send_discord_embed("📊 Progress Update", "สถานะการเทรนล่าสุด", status_color, fields)

    if epoch % 20 == 0:
        val_plot = os.path.join(trainer.save_dir, 'val_batch0_pred.jpg')
        time.sleep(1)
        send_discord_image(val_plot, f"🖼️ **ตัวอย่างผลการทำนายที่ Epoch {epoch}**")

def on_train_end(trainer):
    time.sleep(2)
    result_plot = os.path.join(trainer.save_dir, 'results.png')
    send_discord_image(result_plot, "📈 **สรุปผลการเทรนทั้งหมด**")
    
    stop_reason = "ครบกำหนด 100 รอบ"
    # เช็คว่าจบก่อนกำหนดไหม (Early Stopping)
    if trainer.epoch + 1 < trainer.args.epochs:
        stop_reason = "หยุดอัตโนมัติ (Early Stopping) เพื่อกัน Overfit"

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_training_time)))
    send_discord_embed("✅ Mission Complete!", f"จบงานแล้วครับลูกพี่!\nสาเหตุ: **{stop_reason}**\nเวลาที่ใช้ไป: `{total_time}`", COLOR_FINISH)

# =========================================================
# 4. Main Process
# =========================================================
if __name__ == '__main__':
    
    if os.path.exists(OLD_WEIGHTS_PATH):
        print("--- Fine-tuning Mode ---")
        model = YOLO(OLD_WEIGHTS_PATH)
    else:
        print("--- New Training Mode ---")
        model = YOLO(DEFAULT_MODEL)

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    # *** ตั้งค่าสำหรับคนไม่อยู่หน้าจอ ***
    model.train(
        data=DATA_YAML_PATH,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        name=PROJECT_NAME,
        exist_ok=True,
        workers=4,
        
        # --- ระบบกันตาย (Smart Guard) ---
        patience=20,      # <--- พระเอกของเรา: ถ้าไม่เก่งขึ้นใน 20 รอบ จะหยุดเองทันที (คุณกลับมาจะได้ไม่เสียเวลาเปล่า)
        dropout=0.15,     # กันท่องจำ
        
        # --- เพิ่มความโหดให้เก่งจริง ---
        mosaic=1.0, 
        mixup=0.15,
        degrees=15.0,
        perspective=0.0005,
        copy_paste=0.1
    )

    # *** Auto Export: เอาไปลง Pi ได้เลย ***
    try:
        print("--- Exporting Best Model to NCNN ---")
        # YOLO จะเลือก best.pt ให้อัตโนมัติหลังเทรนเสร็จ
        exported_file = model.export(format='ncnn') 
        send_discord_embed("📦 Ready for Raspberry Pi", f"ไฟล์ NCNN ถูกสร้างเรียบร้อยแล้ว!\nตำแหน่ง: `{exported_file}`", COLOR_FINISH)
    except Exception as e:
        send_discord_embed("❌ Export Failed", f"Error: {e}", COLOR_DANGER)

    print("!!! All Done. You can close this window now. !!!")