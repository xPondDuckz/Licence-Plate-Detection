#!/usr/bin/env python3
"""
License Plate Recognition GUI - OPTIMIZED for Raspberry Pi 4
Fullscreen GUI (1920x1080) with YOLOv8 detection
เวอร์ชันปรับแต่งให้ทำงานลื่นขึ้นบน Raspberry Pi
"""

import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import os
from collections import deque
import time


class LicensePlateDetectorGUI:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("ระบบตรวจจับป้ายทะเบียนรถ")
        
        # ตั้งค่า Fullscreen
        self.root.attributes('-fullscreen', True)
        self.root.geometry("1920x1080")
        self.root.configure(bg='#1e1e1e')
        
        # โหลดโมเดล YOLO
        print(f"กำลังโหลดโมเดล: {model_path}")
        self.model = YOLO(model_path)
        self.model.fuse()  # Optimize model for inference
        
        # ตัวแปรสำหรับกล้อง
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.detection_count = 0
        self.total_detections = 0
        
        # FPS tracking
        self.fps_queue = deque(maxlen=30)
        self.frame_time = time.time()
        
        # Frame skip สำหรับประสิทธิภาพ
        self.frame_skip = 2  # ประมวลผลทุก 2 เฟรม
        self.frame_counter = 0
        
        # สร้าง UI
        self.create_ui()
        
        # Bind keys
        self.root.bind('<Escape>', lambda e: self.toggle_fullscreen())
        self.root.bind('<q>', lambda e: self.quit_app())
        self.root.bind('<s>', lambda e: self.save_screenshot())
        
    def create_ui(self):
        """สร้างส่วนติดต่อผู้ใช้"""
        
        # Header
        header_frame = tk.Frame(self.root, bg='#2d2d2d', height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="🚗 ระบบตรวจจับป้ายทะเบียนรถ - YOLOv8 (Raspberry Pi)",
            font=('Arial', 26, 'bold'),
            bg='#2d2d2d',
            fg='#00ff00'
        )
        title_label.pack(pady=22)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Video display frame (ซ้าย)
        video_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Label สำหรับแสดงวิดีโอ
        self.video_label = tk.Label(video_frame, bg='#000000')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel (ขวา)
        control_frame = tk.Frame(main_frame, bg='#2d2d2d', width=400, relief=tk.RAISED, bd=2)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        control_frame.pack_propagate(False)
        
        # ชื่อแผง
        panel_title = tk.Label(
            control_frame,
            text="⚙️ แผงควบคุม",
            font=('Arial', 20, 'bold'),
            bg='#2d2d2d',
            fg='#ffffff'
        )
        panel_title.pack(pady=20)
        
        # ปุ่มควบคุม
        button_style = {
            'font': ('Arial', 13, 'bold'),
            'width': 25,
            'height': 2,
            'relief': tk.RAISED,
            'bd': 3
        }
        
        self.start_button = tk.Button(
            control_frame,
            text="▶ เริ่มตรวจจับ",
            command=self.start_detection,
            bg='#4CAF50',
            fg='white',
            activebackground='#45a049',
            cursor='hand2',
            **button_style
        )
        self.start_button.pack(pady=8)
        
        self.stop_button = tk.Button(
            control_frame,
            text="⏸ หยุดตรวจจับ",
            command=self.stop_detection,
            bg='#f44336',
            fg='white',
            activebackground='#da190b',
            cursor='hand2',
            state=tk.DISABLED,
            **button_style
        )
        self.stop_button.pack(pady=8)
        
        self.screenshot_button = tk.Button(
            control_frame,
            text="📷 จับภาพหน้าจอ",
            command=self.save_screenshot,
            bg='#2196F3',
            fg='white',
            activebackground='#0b7dda',
            cursor='hand2',
            state=tk.DISABLED,
            **button_style
        )
        self.screenshot_button.pack(pady=8)
        
        # Separator
        separator = ttk.Separator(control_frame, orient='horizontal')
        separator.pack(fill=tk.X, padx=20, pady=15)
        
        # สถิติ
        stats_label = tk.Label(
            control_frame,
            text="📊 สถิติการตรวจจับ",
            font=('Arial', 16, 'bold'),
            bg='#2d2d2d',
            fg='#00bfff'
        )
        stats_label.pack(pady=10)
        
        # จำนวนการตรวจจับในเฟรมปัจจุบัน
        self.current_count_label = tk.Label(
            control_frame,
            text="ตรวจพบในเฟรม: 0",
            font=('Arial', 13),
            bg='#2d2d2d',
            fg='#ffeb3b'
        )
        self.current_count_label.pack(pady=3)
        
        # จำนวนรวมทั้งหมด
        self.total_count_label = tk.Label(
            control_frame,
            text="จำนวนรวมทั้งหมด: 0",
            font=('Arial', 13),
            bg='#2d2d2d',
            fg='#ffffff'
        )
        self.total_count_label.pack(pady=3)
        
        # FPS
        self.fps_label = tk.Label(
            control_frame,
            text="FPS: 0.0",
            font=('Arial', 13),
            bg='#2d2d2d',
            fg='#00ff00'
        )
        self.fps_label.pack(pady=3)
        
        # ความละเอียด
        self.resolution_label = tk.Label(
            control_frame,
            text="ความละเอียด: N/A",
            font=('Arial', 13),
            bg='#2d2d2d',
            fg='#ffffff'
        )
        self.resolution_label.pack(pady=3)
        
        # สถานะ
        self.status_label = tk.Label(
            control_frame,
            text="สถานะ: พร้อมใช้งาน ✓",
            font=('Arial', 14, 'bold'),
            bg='#2d2d2d',
            fg='#ffeb3b'
        )
        self.status_label.pack(pady=10)
        
        # Separator
        separator2 = ttk.Separator(control_frame, orient='horizontal')
        separator2.pack(fill=tk.X, padx=20, pady=15)
        
        # การตั้งค่า
        settings_label = tk.Label(
            control_frame,
            text="🔧 การตั้งค่า",
            font=('Arial', 16, 'bold'),
            bg='#2d2d2d',
            fg='#00bfff'
        )
        settings_label.pack(pady=10)
        
        # Confidence threshold
        conf_frame = tk.Frame(control_frame, bg='#2d2d2d')
        conf_frame.pack(pady=5)
        
        tk.Label(
            conf_frame,
            text="Confidence:",
            font=('Arial', 11),
            bg='#2d2d2d',
            fg='#ffffff'
        ).pack(side=tk.LEFT, padx=5)
        
        self.conf_var = tk.DoubleVar(value=0.5)
        self.conf_scale = tk.Scale(
            conf_frame,
            from_=0.1,
            to=0.9,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.conf_var,
            bg='#2d2d2d',
            fg='#ffffff',
            highlightthickness=0,
            length=150
        )
        self.conf_scale.pack(side=tk.LEFT)
        
        self.conf_value_label = tk.Label(
            conf_frame,
            text="0.5",
            font=('Arial', 11, 'bold'),
            bg='#2d2d2d',
            fg='#00ff00',
            width=4
        )
        self.conf_value_label.pack(side=tk.LEFT, padx=5)
        
        # Update label when scale changes
        self.conf_scale.config(command=self.update_conf_label)
        
        # Separator
        separator3 = ttk.Separator(control_frame, orient='horizontal')
        separator3.pack(fill=tk.X, padx=20, pady=15)
        
        # ปุ่มออก
        quit_button = tk.Button(
            control_frame,
            text="✖ ออกจากโปรแกรม",
            command=self.quit_app,
            bg='#607d8b',
            fg='white',
            activebackground='#546e7a',
            cursor='hand2',
            **button_style
        )
        quit_button.pack(pady=10, side=tk.BOTTOM, pady=15)
        
        # คำแนะนำ
        help_label = tk.Label(
            control_frame,
            text="⌨️  คีย์ลัด:\n"
                 "ESC - เข้า/ออก Fullscreen\n"
                 "Q - ปิดโปรแกรม\n"
                 "S - จับภาพหน้าจอ",
            font=('Arial', 10),
            bg='#2d2d2d',
            fg='#888888',
            justify=tk.LEFT
        )
        help_label.pack(pady=10, side=tk.BOTTOM)
        
    def update_conf_label(self, value):
        """อัพเดท label ของ confidence"""
        self.conf_value_label.config(text=f"{float(value):.1f}")
        
    def start_detection(self):
        """เริ่มการตรวจจับ"""
        if not self.is_running:
            # ลองเปิดกล้อง
            self.cap = cv2.VideoCapture(0)
            
            # ตั้งค่าความละเอียดกล้อง (ปรับให้เหมาะกับ RPi)
            # ใช้ความละเอียดต่ำกว่าเพื่อประสิทธิภาพที่ดีขึ้น
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                self.status_label.config(text="สถานะ: ❌ ไม่พบกล้อง!", fg='#ff0000')
                return
            
            # ดึงความละเอียดจริง
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.resolution_label.config(text=f"ความละเอียด: {actual_width}x{actual_height}")
            
            self.is_running = True
            self.detection_count = 0
            self.total_detections = 0
            self.frame_counter = 0
            
            # เปลี่ยนสถานะปุ่ม
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.screenshot_button.config(state=tk.NORMAL)
            self.status_label.config(text="สถานะ: ⚡ กำลังทำงาน...", fg='#00ff00')
            
            # เริ่ม thread สำหรับการประมวลผล
            self.detection_thread = threading.Thread(target=self.process_video, daemon=True)
            self.detection_thread.start()
            
    def stop_detection(self):
        """หยุดการตรวจจับ"""
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # เปลี่ยนสถานะปุ่ม
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.screenshot_button.config(state=tk.DISABLED)
        self.status_label.config(text="สถานะ: ⏸ หยุดทำงาน", fg='#ffeb3b')
        
    def process_video(self):
        """ประมวลผลวิดีโอและตรวจจับป้ายทะเบียน - เวอร์ชันปรับแต่ง"""
        
        while self.is_running:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            
            if not ret:
                print("ไม่สามารถอ่านเฟรมจากกล้อง")
                break
            
            self.frame_counter += 1
            
            # Frame skipping เพื่อประสิทธิภาพ
            if self.frame_counter % self.frame_skip == 0:
                # ตรวจจับด้วย YOLO
                conf_threshold = self.conf_var.get()
                results = self.model(frame, conf=conf_threshold, verbose=False, device='cpu')
                
                # วาดผลลัพธ์
                annotated_frame = results[0].plot()
                
                # นับจำนวนการตรวจพบ
                detections = len(results[0].boxes)
                self.detection_count = detections
                if detections > 0:
                    self.total_detections += detections
            else:
                # ใช้เฟรมเดิมถ้าข้าม
                annotated_frame = frame.copy()
                detections = 0
            
            # คำนวณ FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = 1.0 / elapsed
                self.fps_queue.append(fps)
                avg_fps = np.mean(self.fps_queue)
            else:
                avg_fps = 0
            
            # วาดข้อมูลบนเฟรม
            self.draw_info_on_frame(annotated_frame, avg_fps, detections)
            
            # เก็บเฟรมปัจจุบัน
            self.current_frame = annotated_frame.copy()
            
            # อัพเดท UI (ไม่บ่อยเกินไป)
            if self.frame_counter % 2 == 0:
                self.update_ui(annotated_frame, avg_fps, detections)
            
    def draw_info_on_frame(self, frame, fps, detections):
        """วาดข้อมูลบนเฟรม"""
        h, w = frame.shape[:2]
        
        # พื้นหลังสำหรับข้อความ
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        # จำนวนที่ตรวจพบ
        color = (0, 255, 255) if detections > 0 else (100, 100, 100)
        cv2.putText(
            frame,
            f"Detected: {detections}",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
        
        # Confidence threshold
        cv2.putText(
            frame,
            f"Conf: {self.conf_var.get():.1f}",
            (15, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # เวลา
        current_time = datetime.now().strftime("%H:%M:%S")
        time_text = f"Time: {current_time}"
        text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(
            frame,
            time_text,
            (w - text_size[0] - 15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
    def update_ui(self, frame, fps, detections):
        """อัพเดทการแสดงผลบน UI"""
        # แปลงเฟรมเป็น RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ปรับขนาดเฟรมให้พอดีกับหน้าจอ
        height, width = frame_rgb.shape[:2]
        max_width = 1400
        max_height = 900
        
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # แปลงเป็น ImageTk
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # อัพเดทวิดีโอ
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        # อัพเดทสถิติ
        self.current_count_label.config(text=f"ตรวจพบในเฟรม: {detections}")
        self.total_count_label.config(text=f"จำนวนรวมทั้งหมด: {self.total_detections}")
        self.fps_label.config(text=f"FPS: {fps:.1f}")
        
    def save_screenshot(self):
        """บันทึกภาพหน้าจอ"""
        if self.current_frame is not None:
            # สร้างโฟลเดอร์ screenshots ถ้ายังไม่มี
            os.makedirs("screenshots", exist_ok=True)
            
            # สร้างชื่อไฟล์
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshots/lpr_{timestamp}.jpg"
            
            # บันทึกภาพ
            cv2.imwrite(filename, self.current_frame)
            
            # แสดงสถานะ
            self.status_label.config(text=f"✓ บันทึก: {filename}", fg='#00ff00')
            print(f"บันทึกภาพ: {filename}")
            
            # รีเซ็ตสถานะหลัง 3 วินาที
            self.root.after(3000, lambda: self.status_label.config(
                text="สถานะ: ⚡ กำลังทำงาน...", 
                fg='#00ff00'
            ))
        
    def toggle_fullscreen(self):
        """สลับโหมด fullscreen"""
        current = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current)
        
        if not current:
            self.status_label.config(text="สถานะ: 🖥️ Fullscreen", fg='#00bfff')
        else:
            self.status_label.config(text="สถานะ: 🪟 Windowed", fg='#00bfff')
        
    def quit_app(self):
        """ปิดโปรแกรม"""
        print("กำลังปิดโปรแกรม...")
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()


def main():
    """ฟังก์ชันหลัก"""
    print("=" * 60)
    print("ระบบตรวจจับป้ายทะเบียนรถ - YOLOv8")
    print("สำหรับ Raspberry Pi 4")
    print("=" * 60)
    
    # Path สำหรับ Raspberry Pi (Linux)
    model_path = "/home/pi/my_lpr_model3/weights/best.pt"
    
    # สำหรับการทดสอบ สามารถใช้ path อื่นได้
    # model_path = "best.pt"  # ถ้าไฟล์อยู่ในโฟลเดอร์เดียวกัน
    
    # ตรวจสอบว่าไฟล์โมเดลมีอยู่หรือไม่
    if not os.path.exists(model_path):
        print(f"❌ ไม่พบไฟล์โมเดล: {model_path}")
        print("\nกรุณา:")
        print("1. คัดลอกไฟล์ best.pt ไปยัง Raspberry Pi")
        print("2. วางไฟล์ที่: /home/pi/my_lpr_model3/weights/best.pt")
        print("3. หรือแก้ไข 'model_path' ในโค้ดให้ตรงกับที่ตั้งไฟล์")
        print("\nหรือใช้คำสั่ง:")
        print("  scp best.pt pi@<IP>:/home/pi/my_lpr_model3/weights/")
        return
    
    print(f"✓ พบไฟล์โมเดล: {model_path}")
    print("กำลังเริ่มต้นโปรแกรม...\n")
    
    # สร้างหน้าต่าง GUI
    root = tk.Tk()
    app = LicensePlateDetectorGUI(root, model_path)
    
    print("✓ โปรแกรมพร้อมใช้งาน")
    print("  กด ESC เพื่อสลับ Fullscreen")
    print("  กด Q เพื่อปิดโปรแกรม")
    print("  กด S เพื่อจับภาพหน้าจอ")
    print("=" * 60)
    
    root.mainloop()


if __name__ == "__main__":
    main()
