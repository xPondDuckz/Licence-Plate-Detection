import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
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
        self.root.title("KMITL ALPR Dashboard - Precision Mode")
        
        # บังคับ Full Screen
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#FFFFFF")
        self.root.update()
        
        # --- Config ---
        self.kmitl_orange = "#FF6600"
        self.yolo_path = "E:/Project/CE/my_lpr_model3/weights/best.pt"
        self.video_test_path = "E:/Project/CE/video_test.mp4"
        self.logo_path = "/home/sunlight-lnwza007/Downloads/logo.png"
        self.f_family = "IBM Plex Sans Thai Looped"

        print("กำลังโหลดโมเดลประสิทธิภาพสูง...")
        self.model = YOLO(self.yolo_path)
        # โหลด OCR แบบเน้นภาษาไทย
        self.reader = easyocr.Reader(['th', 'en'], gpu=False)

        self.setup_ui()
        self.setup_video_source()
        
        self.running = True
        self.update_clock()
        threading.Thread(target=self.video_stream, daemon=True).start()

    def setup_video_source(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_test_path)
            self.is_video_file = True
        else:
            self.is_video_file = False

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#FFFFFF", height=100, highlightthickness=1, highlightbackground="#EEEEEE")
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        
        tk.Label(header, text="KMITL ALPR - PRECISION DETECTION", 
                 font=(self.f_family, 28, "bold"), bg="#FFFFFF", fg=self.kmitl_orange).place(relx=0.5, rely=0.5, anchor="center")
        
        self.clock_label = tk.Label(header, font=(self.f_family, 20), bg="#FFFFFF")
        self.clock_label.pack(side="right", padx=30)

        # Body
        content = tk.Frame(self.root, bg="#FFFFFF")
        content.pack(fill="both", expand=True, padx=30, pady=20)

        # จอวิดีโอ (ซ้าย)
        left_panel = tk.Frame(content, bg="#FFFFFF")
        left_panel.pack(side="left", fill="both", expand=True)
        self.video_container = tk.Label(left_panel, bg="#000000")
        self.video_container.pack(fill="both", expand=True)

        # Sidebar (ขวา)
        right_panel = tk.Frame(content, bg="#FFFFFF", width=500)
        right_panel.pack(side="right", fill="y", padx=(30, 0))
        right_panel.pack_propagate(False)

        tk.Label(right_panel, text="ภาพป้ายทะเบียนที่ตรวจพบ", font=(self.f_family, 16, "bold"), bg="#FFFFFF").pack(anchor="w", pady=(0, 10))
        self.crop_display = tk.Label(right_panel, bg="#E0E0E0", height=10)
        self.crop_display.pack(fill="x", pady=(0, 30))

        # ผลลัพธ์
        self.res_card = tk.Frame(right_panel, bg="#F8F9FA", padx=25, pady=30, highlightthickness=1, highlightbackground="#DDDDDD")
        self.res_card.pack(fill="x")
        
        tk.Label(self.res_card, text="หมายเลขทะเบียน:", font=(self.f_family, 14), bg="#F8F9FA").pack(anchor="w")
        self.lbl_num = tk.Label(self.res_card, text="---", font=(self.f_family, 48, "bold"), bg="#F8F9FA", fg=self.kmitl_orange)
        self.lbl_num.pack(anchor="w", pady=(5, 15))
        
        tk.Label(self.res_card, text="จังหวัด:", font=(self.f_family, 14), bg="#F8F9FA").pack(anchor="w")
        self.lbl_prov = tk.Label(self.res_card, text="---", font=(self.f_family, 26, "bold"), bg="#F8F9FA")
        self.lbl_prov.pack(anchor="w")

        self.exit_btn = tk.Button(right_panel, text="ปิดโปรแกรม (ESC)", command=self.on_exit, 
                             font=(self.f_family, 16, "bold"), bg="#FF4757", fg="white", relief="flat", pady=20)
        self.exit_btn.pack(side="bottom", fill="x")

    def improve_plate_quality(self, plate_img):
        """ปรับปรุงคุณภาพภาพป้ายทะเบียนก่อนส่งให้ OCR"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        # ขยายภาพให้ใหญ่ขึ้น 2 เท่าเพื่อให้ OCR อ่านง่ายขึ้น
        upscale = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # ปรับความคมชัด (Thresholding)
        thresh = cv2.threshold(upscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh

    def video_stream(self):
        frame_count = 0
        latest_annotated = None
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file: self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                break

            frame_count += 1
            # ตรวจจับทุก 2 เฟรมเพื่อความลื่นไหล
            if frame_count % 2 == 0:
                # เพิ่ม imgsz เป็น 640 เพื่อความแม่นยำสูงสุด
                results = self.model(frame, conf=0.5, imgsz=640, verbose=False)
                
                if results and len(results[0].boxes) > 0:
                    latest_annotated = results[0].plot()
                    
                    # ทำ OCR เมื่อเจอวัตถุที่มั่นใจ
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # ครอปภาพทะเบียนแบบขยายขอบเล็กน้อย
                        plate_crop = frame[max(0,y1-5):min(frame.shape[0],y2+5), 
                                           max(0,x1-5):min(frame.shape[1],x2+5)]
                        
                        if plate_crop.size > 0:
                            # ปรับปรุงภาพก่อนอ่าน
                            processed_plate = self.improve_plate_quality(plate_crop)
                            ocr_res = self.reader.readtext(processed_plate)
                            
                            if ocr_res:
                                # อัปเดต UI ทันทีที่อ่านได้
                                self.root.after(0, self.update_detection_ui, plate_crop, ocr_res)
                else:
                    latest_annotated = frame.copy()

            self.update_video_view(latest_annotated if latest_annotated is not None else frame)
            time.sleep(0.01)

    def update_detection_ui(self, plate_img, ocr_res):
        """แสดงผลรูปที่ครอปและข้อความป้ายทะเบียน"""
        # แสดงรูป Crop
        img = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
        img = img.resize((450, 180), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.crop_display.configure(image=imgtk)
        self.crop_display.image = imgtk

        # จัดการข้อความ OCR
        texts = [res[1] for res in ocr_res]
        if len(texts) >= 2:
            self.lbl_num.config(text=texts[0])
            self.lbl_prov.config(text=texts[1])
        elif len(texts) == 1:
            self.lbl_num.config(text=texts[0])
            self.lbl_prov.config(text="กำลังวิเคราะห์...")

    def update_video_view(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((1350, 850), Image.Resampling.NEAREST)
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