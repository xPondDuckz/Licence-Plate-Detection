import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
from ultralytics import YOLO
import easyocr
import time
from datetime import datetime
import pytz
import os

class ALPRSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("KMITL ALPR Dashboard")
        
        # บังคับ Full Screen สีขาวสะอาดตา
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#FFFFFF")
        
        # --- Config & Assets ---
        self.kmitl_orange = "#FF6600"
        self.bg_color = "#FFFFFF"
        self.card_color = "#F8F9FA"
        self.text_dark = "#2D3436"
        self.f_family = "IBM Plex Sans Thai Looped"
        
        # Paths
        self.logo_path = "/home/sunlight-lnwza007/Downloads/logo.png"
        self.video_test_path = "/home/sunlight-lnwza007/Project/video/video_test.mp4"
        self.yolo_path = "/home/sunlight-lnwza007/Project/model/weights/best.pt"
        
        print("กำลังโหลดโมเดลและเตรียมระบบ...")
        self.model = YOLO(self.yolo_path)
        self.reader = easyocr.Reader(['th', 'en'], gpu=False)

        self.setup_ui()
        self.setup_video_source()
        
        self.running = True
        self.last_plate = "---"
        
        self.update_clock()
        threading.Thread(target=self.video_stream, daemon=True).start()

    def setup_video_source(self):
        """ตรวจสอบกล้อง ถ้าไม่มีให้ใช้คลิปวิดีโอ"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print(f"⚠️ ไม่พบกล้อง: สลับไปใช้ไฟล์วิดีโอทดสอบที่ {self.video_test_path}")
            self.cap = cv2.VideoCapture(self.video_test_path)
            self.is_video_file = True
        else:
            self.is_video_file = False

    def setup_ui(self):
        """Header ใหม่: โลโก้ซ้าย ข้อความกลาง"""
        # 1. Header Bar
        header = tk.Frame(self.root, bg=self.bg_color, height=100, highlightbackground="#EEEEEE", highlightthickness=1)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        # ฝั่งซ้าย: โลโก้
        logo_container = tk.Frame(header, bg=self.bg_color, width=250)
        logo_container.pack(side="left", fill="y", padx=20)
        
        if os.path.exists(self.logo_path):
            try:
                logo_img = Image.open(self.logo_path)
                logo_img = logo_img.resize((180, 70), Image.Resampling.LANCZOS)
                self.logo_tk = ImageTk.PhotoImage(logo_img)
                tk.Label(logo_container, image=self.logo_tk, bg=self.bg_color).pack(pady=15)
            except Exception as e:
                print(f"Logo Error: {e}")

        # ฝั่งขวา: เวลา
        time_container = tk.Frame(header, bg=self.bg_color, width=250)
        time_container.pack(side="right", fill="y", padx=20)
        self.clock_label = tk.Label(time_container, text="", font=(self.f_family, 18), bg=self.bg_color, fg=self.text_dark)
        self.clock_label.pack(pady=30)

        # ตรงกลาง: ข้อความหลัก
        tk.Label(header, text="KMITL AUTOMATIC LICENSE PLATE RECOGNITION", 
                 font=(self.f_family, 26, "bold"), bg=self.bg_color, fg=self.kmitl_orange).place(relx=0.5, rely=0.5, anchor="center")

        # 2. Main Content
        content = tk.Frame(self.root, bg=self.bg_color)
        content.pack(fill="both", expand=True, padx=25, pady=25)

        # ฝั่งซ้าย: หน้าจอวิดีโอ
        left_panel = tk.Frame(content, bg=self.bg_color)
        left_panel.pack(side="left", fill="both", expand=True)
        self.video_container = tk.Label(left_panel, bg="#F0F0F0")
        self.video_container.pack(fill="both", expand=True)

        # ฝั่งขวา: แผงควบคุมและผลลัพธ์
        right_panel = tk.Frame(content, bg=self.bg_color, width=450)
        right_panel.pack(side="right", fill="y", padx=(25, 0))
        right_panel.pack_propagate(False)

        res_card = tk.Frame(right_panel, bg=self.card_color, padx=25, pady=30, highlightbackground="#DDDDDD", highlightthickness=1)
        res_card.pack(fill="x", pady=(0, 25))
        tk.Label(res_card, text="ป้ายทะเบียนที่ตรวจพบล่าสุด", font=(self.f_family, 16), bg=self.card_color, fg="#636E72").pack(anchor="w")
        self.plate_display = tk.Label(res_card, text="---", font=(self.f_family, 80, "bold"), bg=self.card_color, fg=self.kmitl_orange)
        self.plate_display.pack(pady=25)

        st_card = tk.Frame(right_panel, bg=self.card_color, padx=25, pady=25, highlightbackground="#DDDDDD", highlightthickness=1)
        st_card.pack(fill="x")
        self.status_text = tk.Label(st_card, text="● ระบบกำลังวิเคราะห์", font=(self.f_family, 14), bg=self.card_color, fg="#27AE60")
        self.status_text.pack(anchor="w")

        exit_btn = tk.Button(right_panel, text="ปิดโปรแกรม (ESC)", command=self.on_exit, 
                             font=(self.f_family, 14), bg="#FF4757", fg="white", relief="flat", pady=15)
        exit_btn.pack(side="bottom", fill="x")

    def video_stream(self):
        """สตรีมวิดีโอพร้อม AI"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # วนคลิปใหม่ถ้าจบ
                    continue
                else:
                    break

            # YOLO Detection
            results = self.model(frame, conf=0.5, verbose=False)
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate = frame[y1:y2, x1:x2]
                    if plate.size > 0:
                        ocr_res = self.reader.readtext(plate)
                        if ocr_res:
                            text = ocr_res[0][1]
                            if text != self.last_plate:
                                self.last_plate = text
                                self.plate_display.config(text=text)

            self.update_video_view(results[0].plot())
            time.sleep(0.01)

    def update_video_view(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((1300, 850), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_container.configure(image=imgtk)
        self.video_container.image = imgtk

    def update_clock(self):
        now = datetime.now(pytz.timezone('Asia/Bangkok')).strftime("%H:%M:%S")
        self.clock_label.config(text=f"Time: {now}")
        self.root.after(1000, self.update_clock)

    def on_exit(self):
        self.running = False
        if self.cap.isOpened(): self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ALPRSystem(root)
    root.bind("<Escape>", lambda e: app.on_exit())
    root.mainloop()