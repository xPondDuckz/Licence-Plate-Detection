from ultralytics import YOLO
import torch
import requests
import datetime

# --- [ตั้งค่า Discord Webhook] ---
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1467567408293085214/GNb0F3RawobteeefpTiNauM01w7TCoQklYKwgksNRzEGOCrmggTxe_7JwhrsL62o80_W"

# --- [สีสำหรับ Embeds] ---
COLOR_START   = 0x2ecc71  # เขียว
COLOR_UPDATE  = 0x3498db  # ฟ้า
COLOR_FINISH  = 0xf1c40f  # เหลือง
COLOR_SUCCESS = 0x9b59b6  # ม่วง
COLOR_ERROR   = 0xe74c3c  # แดง

def send_discord_embed(title, description, color, fields=None):
    try:
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "footer": {"text": "YOLOv26 Training System 🚀"},
            "timestamp": datetime.datetime.now().astimezone().isoformat()
        }
        if fields: embed["fields"] = fields
        data = {"embeds": [embed]}
        requests.post(DISCORD_WEBHOOK_URL, json=data).raise_for_status()
    except Exception as e:
        print(f"Discord Notify Error: {e}")

# --- [Callbacks สำหรับติดตามการเทรน] ---
def on_train_epoch_end(trainer):
    # ส่งแจ้งเตือนทุกๆ 1 Epoch (ปรับเลขได้)
    epoch = trainer.epoch + 1
    if epoch % 1 == 0:
        fields = [
            {"name": "📦 Model", "value": "`YOLOv26n`", "inline": True},
            {"name": "🔄 Epoch", "value": f"`{epoch}/{trainer.args.epochs}`", "inline": True},
            {"name": "📈 Fitness (mAP)", "value": f"`{trainer.fitness:.4f}`", "inline": False}
        ]
        send_discord_embed("📊 Training Progress Update", "กำลังประมวลผลข้อมูลป้ายทะเบียน...", COLOR_UPDATE, fields)

def on_train_end(trainer):
    send_discord_embed("✅ Training Finished!", "เทรนเสร็จสมบูรณ์แล้ว! กำลังเข้าสู่ขั้นตอนการ Export ไฟล์สำหรับ Raspberry Pi...", COLOR_FINISH)

if __name__ == '__main__':
    # 1. เช็คความพร้อมของ GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🔥 NVIDIA GPU Active: {gpu_name}")
        send_discord_embed("🚀 System Ready", f"ตรวจพบ GPU: `{gpu_name}`\nเริ่มทำการเทรน **YOLOv26n**", COLOR_START)
    else:
        print("⚠️ ไม่พบ NVIDIA GPU! โปรดตรวจสอบไดรเวอร์ CUDA")
        send_discord_embed("❌ Error", "ไม่พบ NVIDIA GPU ในระบบ", COLOR_ERROR)
        exit()

    # 2. โหลดโมเดล YOLOv26 Nano
    print("--- Loading YOLOv26n ---")
    model = YOLO('yolov26n.pt') # <--- ใช้ v26n เป็นหลักตามต้องการ

    # 3. ลงทะเบียน Callbacks
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    # 4. เริ่มการเทรน (Full Power NVIDIA GPU)
    print("--- Training Started ---")
    results = model.train(
        data=r'C:\Users\Admin\Downloads\LPR plate.v1i.yolov8\data.yaml', 
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,      # บังคับใช้ NVIDIA GPU ตัวที่ 0
        name='yolov26_lpr_training', 
        workers=4,
        exist_ok=True
    )

    # 5. Export ไปเป็น NCNN สำหรับ Raspberry Pi
    print("--- Exporting to NCNN ---")
    try:
        model.export(format='ncnn')
        send_discord_embed("📦 Export Successful!", "ไฟล์โมเดล YOLOv26n ถูกแปลงเป็น NCNN เรียบร้อย! พร้อมย้ายลง Raspberry Pi แล้ว 🎉", COLOR_SUCCESS)
    except Exception as e:
        send_discord_embed("❌ Export Failed", f"เกิดข้อผิดพลาด: {str(e)}", COLOR_ERROR)

    print("!!! All Processes Done !!!")