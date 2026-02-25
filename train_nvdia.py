import os
import time
import random
import cv2
import torch
import platform
import getpass
import requests
import datetime
from ultralytics import YOLO

# =========================================================
# 1. ตั้งค่า (Configuration)
# =========================================================
DISCORD_WEBHOOK_URL = "ใส่_WEBHOOK_URL_ของคุณที่นี่"
DATA_YAML_PATH = r'G:\Project\Data\pengsiri\data.yaml'

# --- [ตั้งค่าไฟล์วิดีโอสำหรับทดสอบ] ---
# ⚠️ สำคัญ: ใส่ Path ไฟล์วิดีโอที่มีอยู่ในเครื่องตรงนี้
TEST_VIDEO_PATH = r'G:\Project\Data\pengsiri\test_video.mp4' 

# --- [Phase 1: คัดเลือกตัวนักกีฬา] ---
SEARCH_ROUNDS = 3       # แข่งกี่รอบ
SEARCH_EPOCHS = 50      # รอบคัดตัว

# --- [Phase 2: ปั้นแชมป์] ---
FINAL_EPOCHS = 100
PROJECT_NAME = 'yolov26_auto_champion_video'

# สี Discord
COLOR_INFO = 0x3498db
COLOR_WIN  = 0xf1c40f
COLOR_FINAL= 0x9b59b6

# =========================================================
# 2. ระบบ Discord (แจ้งเตือน & ส่งไฟล์)
# =========================================================
def send_discord_file(file_path, caption=""):
    """ฟังก์ชันส่งไฟล์ (รูปหรือวิดีโอ) เข้า Discord"""
    if not os.path.exists(file_path): return
    try:
        with open(file_path, 'rb') as f:
            requests.post(DISCORD_WEBHOOK_URL, data={"content": caption}, files={"file": f}, timeout=30)
    except Exception as e:
        print(f"Error sending file: {e}")

def send_discord_embed(title, description, color, fields=None):
    try:
        payload = {
            "embeds": [{
                "title": title,
                "description": description,
                "color": color,
                "fields": fields if fields else [],
                "footer": {"text": f"AutoChampion AI | {getpass.getuser()}"},
                "timestamp": datetime.datetime.now().astimezone().isoformat()
            }]
        }
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    except: pass

# =========================================================
# 3. ฟังก์ชันสร้างคลิปทดสอบ (Video Test)
# =========================================================
def run_local_video_test(model_path, source_video, duration_sec=10):
    if not os.path.exists(source_video):
        send_discord_embed("⚠️ Video Not Found", f"หาไฟล์วิดีโอไม่เจอ: `{source_video}`", 0xe74c3c)
        return

    print(f"🎬 Creating Test Video from: {os.path.basename(source_video)}...")
    output_video = "test_result_preview.mp4"
    if os.path.exists(output_video): os.remove(output_video)

    # Setup Video
    cap = cv2.VideoCapture(source_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Resize ลงถ้าวิดีโอใหญ่เกิน (เพื่อลดขนาดไฟล์ส่ง Discord)
    if width > 1280:
        scale = 1280 / width
        width = int(width * scale)
        height = int(height * scale)

    # ใช้ mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    model = YOLO(model_path)
    max_frames = int(fps * duration_sec)
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success: break
        
        # Resize frame ก่อนส่งเข้าโมเดล (ถ้าจำเป็น)
        if frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))

        # ให้ AI จับภาพ (conf=0.4 คือต้องมั่นใจ 40% ถึงจะวาด)
        results = model(frame, verbose=False, conf=0.4)
        annotated_frame = results[0].plot()
        
        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()
    
    # ส่งเข้า Discord
    file_size_mb = os.path.getsize(output_video) / (1024 * 1024)
    if file_size_mb < 25: # เช็คขนาดไฟล์ (Discord Limit)
        send_discord_file(output_video, f"🎥 **คลิปทดสอบ (Round 1 Result)**\nModel: `{os.path.basename(model_path)}`")
        print("   ✅ Video sent to Discord!")
    else:
        send_discord_embed("⚠️ File Too Large", f"คลิปใหญ่เกินไป ({file_size_mb:.2f} MB)", 0xe74c3c)

# =========================================================
# 4. Callbacks (รอบชิง)
# =========================================================
start_final_time = None
def on_final_start(trainer):
    global start_final_time
    start_final_time = time.time()

def on_final_epoch_end(trainer):
    epoch = trainer.epoch + 1
    if epoch % 20 == 0:
        metrics = trainer.metrics
        map50 = metrics.get('metrics/mAP50(B)', 0.0)
        elapsed = time.time() - start_final_time
        avg_time = elapsed / epoch
        eta = str(datetime.timedelta(seconds=int((trainer.args.epochs - epoch) * avg_time)))
        
        fields = [
            {"name": "🔥 Final Training", "value": f"Epoch `{epoch}/{trainer.args.epochs}`", "inline": True},
            {"name": "🎯 mAP50", "value": f"`{map50:.4f}`", "inline": True},
            {"name": "⏳ ETA", "value": f"`{eta}`", "inline": True}
        ]
        send_discord_embed("📊 Champion Progress", "กำลังเทรนโมเดลผู้ชนะ...", COLOR_FINAL, fields)

def on_final_end(trainer):
    time.sleep(2)
    send_discord_file(os.path.join(trainer.save_dir, 'results.png'), "📈 **สรุปผลงานระดับแชมป์**")

# =========================================================
# 5. Main Process
# =========================================================
if __name__ == '__main__':
    
    # --- PHASE 1: Audition ---
    print(f"🎬 PHASE 1: Searching ({SEARCH_ROUNDS} Rounds)...")
    send_discord_embed("🎬 Phase 1 Start", f"เริ่มคัดตัว {SEARCH_ROUNDS} รอบ...", COLOR_INFO)
    
    results_log = []
    
    for r in range(1, SEARCH_ROUNDS + 1):
        seed = random.randint(1, 9999)
        run_name = f"audition_round_{r}"
        print(f"   Running Round {r} (Seed {seed})")
        
        model = YOLO('yolo26n.pt')
        model.train(
            data=DATA_YAML_PATH,
            epochs=SEARCH_EPOCHS,
            imgsz=640,
            batch=16,
            device=0,
            project='runs/detect',
            name=run_name,
            exist_ok=True,
            seed=seed,
            verbose=False
        )
        
        score = model.trainer.metrics.get('metrics/mAP50(B)', 0.0)
        path = str(model.trainer.best)
        results_log.append({"round": r, "seed": seed, "score": score, "path": path})
        print(f"   -> Round {r} Score: {score:.4f}")
        
        # *** จุดเปลี่ยนสำคัญ: ถ้าเป็นรอบที่ 1 ให้ส่งคลิป ***
        if r == 1:
            print("   🎥 Generating Test Video for Round 1...")
            send_discord_embed("🎥 Generating Video", "กำลังสร้างคลิปทดสอบจากผลลัพธ์รอบแรก...", 0xf1c40f)
            run_local_video_test(path, TEST_VIDEO_PATH, duration_sec=10)
        # **********************************************

    # --- DECISION ---
    results_log.sort(key=lambda x: x['score'], reverse=True)
    winner = results_log[0]
    
    fields = [
        {"name": "🏆 The Winner", "value": f"**Round {winner['round']}**", "inline": False},
        {"name": "🚀 Base mAP", "value": f"`{winner['score']:.4f}`", "inline": True}
    ]
    send_discord_embed("✨ ได้ผู้ชนะแล้ว!", "ระบบกำลังเริ่มเทรนต่อทันที...", COLOR_WIN, fields)
    
    # --- PHASE 2: Final ---
    print("\n🚀 PHASE 2: Final Training...")
    champion_model = YOLO(winner['path'])
    champion_model.add_callback("on_train_start", on_final_start)
    champion_model.add_callback("on_train_epoch_end", on_final_epoch_end)
    champion_model.add_callback("on_train_end", on_final_end)
    
    champion_model.train(
        data=DATA_YAML_PATH,
        epochs=FINAL_EPOCHS,
        imgsz=640,
        batch=16,
        device=0,
        project='runs/detect',
        name='CHAMPION_FINAL',
        exist_ok=True,
        patience=20,
        dropout=0.15,
        mosaic=0.5,
        degrees=10.0,
        workers=4
    )
    
    try:
        export_path = champion_model.export(format='ncnn')
        send_discord_embed("📦 Mission Complete", f"ไฟล์พร้อมใช้งานบน Pi:\n`{export_path}`", 0x2ecc71)
    except: pass
    
    print("!!! ALL DONE !!!")