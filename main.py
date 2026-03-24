import os
import cv2
import time
import threading
import socket
import struct
import json
from datetime import datetime
import pytz
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

class ALPRSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("ตรวจจับป้ายทะเบียน KMITL PCC")
        
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#F0F2F5")
        
        self.kmitl_orange = "#FF6600"
        
        self.yolo_path = r"G:\Project\Licence-Plate-Detection\carplate-edit\weights\best.pt"
        self.logo_path = r"G:\Project\Licence-Plate-Detection\logo-kmitl.png"
        self.save_dir = r"G:\Project\Licence-Plate-Detection\plate-save"
        self.video_path = r"G:\Project\Video\2.mp4" 
        
        self.f_family = "sans-serif" 
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.save_to_disk = True
        self.full_screen_video = True

        self.latest_frame = None   
        self.latest_boxes = []     
        
        self.active_plate = None   
        # ขยายเวลาที่ยอมให้ AI มองไม่เห็นป้ายจาก 1.5 วิ เป็น 2.5 วิ (กัน AI ภาพกระพริบ)
        self.plate_timeout = 2.5   
        
        # ตัวแปรใหม่: ระบบ Cooldown ป้องกันการเซฟป้ายเดิมซ้ำ
        self.global_cooldown = 0.0 
        
        self.history_data = []
        
        self.time_offset = 0.0
        self.is_time_synced = False
        self.unsynced_counter = 1
        
        self.app_start_time = time.monotonic() 
        self.history_file = os.path.join(self.save_dir, "history_log.json")

        print("กำลังโหลดโมเดล...")
        self.model = YOLO(self.yolo_path, task='detect')

        self.main_container = tk.Frame(self.root, bg="#F0F2F5")
        self.main_container.pack(fill="both", expand=True)

        self.setup_ui()
        self.setup_video_source()
        self.load_history_from_disk()
        
        self.running = True
        
        threading.Thread(target=self.camera_thread, daemon=True).start()
        threading.Thread(target=self.ai_thread, daemon=True).start()
        threading.Thread(target=self.sync_time_thread, daemon=True).start()
        
        self.root.after(1000, self.update_gui_loop)

    def load_history_from_disk(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    saved_history = json.load(f)
                
                for record in saved_history:
                    img_path = os.path.join(self.save_dir, record['filename'])
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        if img is not None:
                            self.history_data.append({
                                "time": record['time'],
                                "image": img
                            })
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการโหลดประวัติเดิม: {e}")

    def save_history_record(self, time_str, filename):
        try:
            history_list = []
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history_list = json.load(f)
            
            history_list.append({"time": time_str, "filename": filename})
            
            if len(history_list) > 100:
                history_list = history_list[-100:]
                
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_list, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการบันทึกประวัติ: {e}")

    def setup_video_source(self):
        self.cap = cv2.VideoCapture(self.video_path)

    def setup_ui(self):
        # ขยายความสูง Header ให้ใหญ่ขึ้นรับกับโลโก้
        self.header_height = 150 
        self.right_panel_width = 450
        
        header = tk.Frame(self.main_container, bg=self.kmitl_orange, height=self.header_height)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        
        try:
            if os.path.exists(self.logo_path):
                logo_img = Image.open(self.logo_path)
                # ขยายโลโก้จาก 90x90 เป็น 120x120
                logo_img = logo_img.resize((143, 143), Image.LANCZOS)
                self.logo_tk = ImageTk.PhotoImage(logo_img)
                logo_label = tk.Label(header, image=self.logo_tk, bg=self.kmitl_orange)
                logo_label.pack(side="left", padx=40, pady=15)
            else:
                tk.Label(header, text="[ไม่มีโลโก้]", font=(self.f_family, 16), bg=self.kmitl_orange, fg="#FFFFFF").pack(side="left", padx=40, pady=40)
        except Exception as e:
            tk.Label(header, text="[Logo Error]", font=(self.f_family, 16), bg=self.kmitl_orange, fg="#FFFFFF").pack(side="left", padx=40, pady=40)
        
        tk.Label(header, text="ตรวจจับป้ายทะเบียน KMITL PCC", 
                 font=(self.f_family, 46, "bold"), bg=self.kmitl_orange, fg="#FFFFFF").place(relx=0.5, rely=0.5, anchor="center")

        content = tk.Frame(self.main_container, bg="#F0F2F5")
        content.pack(fill="both", expand=True, padx=20, pady=20)

        right_panel = tk.Frame(content, bg="#FFFFFF", width=self.right_panel_width, highlightthickness=1, highlightbackground="#DDDDDD")
        right_panel.pack(side="right", fill="y", padx=(20, 0))
        right_panel.pack_propagate(False)

        tk.Label(right_panel, text="* กดปุ่ม ESC บนคีย์บอร์ดเพื่อออกจากโปรแกรม", 
                 font=(self.f_family, 12), bg="#FFFFFF", fg="#AAAAAA").pack(side="bottom", pady=(10, 20))

        self.history_btn = tk.Button(right_panel, text="ดูประวัติย้อนหลัง (Logs)", command=self.show_history, 
                             font=(self.f_family, 18, "bold"), bg="#007BFF", fg="white", relief="flat", pady=15)
        self.history_btn.pack(side="bottom", fill="x", padx=25, pady=(0, 10))

        tk.Label(right_panel, text="ป้ายทะเบียนที่ตรวจพบล่าสุด", 
                 font=(self.f_family, 22, "bold"), bg="#FFFFFF", fg="#2C3E50").pack(side="top", pady=(30, 15))
        
        blank_img = Image.new('RGB', (400, 160), color='#E9ECEF')
        self.blank_tk = ImageTk.PhotoImage(blank_img)
        
        self.crop_display = tk.Label(right_panel, image=self.blank_tk, bg="#E9ECEF") 
        self.crop_display.pack(side="top", padx=25, pady=10)
        
        self.status_label = tk.Label(right_panel, text="กำลังรอรถผ่านกล้อง...", font=(self.f_family, 16), bg="#FFFFFF", fg="#888888")
        self.status_label.pack(side="top", pady=15)

        left_panel = tk.Frame(content, bg="#000000")
        left_panel.pack(side="left", fill="both", expand=True)
        self.video_container = tk.Label(left_panel, bg="#000000")
        self.video_container.place(relx=0.5, rely=0.5, anchor="center")

    def sync_time_thread(self):
        while self.running:
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                client.settimeout(5.0)
                client.sendto(b'\x1b' + 47 * b'\0', ('pool.ntp.org', 123))
                msg, address = client.recvfrom(1024)
                
                t = struct.unpack("!12I", msg)[10]
                ntp_time = t - 2208988800
                
                self.time_offset = ntp_time - time.time()
                self.is_time_synced = True
                
                time.sleep(3600)
            except Exception as e:
                self.is_time_synced = False
                time.sleep(10)

    def get_synced_datetime(self):
        real_timestamp = time.time() + self.time_offset
        return datetime.fromtimestamp(real_timestamp, pytz.timezone('Asia/Bangkok'))

    def format_thai_date(self, dt):
        thai_months = [
            "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
            "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"
        ]
        day = dt.day
        month = thai_months[dt.month - 1]
        year = dt.year + 543
        time_str = dt.strftime("%H:%M:%S")
        return f"{day} {month} {year} เวลา {time_str}"

    def measure_clarity(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def process_best_plate(self, plate_img):
        now_local = datetime.now()
        
        if self.is_time_synced:
            now = self.get_synced_datetime()
            timestamp_ui = self.format_thai_date(now)
            filename_str = f"plate_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        else:
            if now_local.year < 2024:
                elapsed = int(time.monotonic() - self.app_start_time)
                mins, secs = divmod(elapsed, 60)
                hours, mins = divmod(mins, 60)
                timestamp_ui = f"โหมดออฟไลน์ [{hours:02d}:{mins:02d}:{secs:02d}]"
                filename_str = f"plate_offline_{self.unsynced_counter:04d}.jpg"
            else:
                timestamp_ui = self.format_thai_date(now_local)
                filename_str = f"plate_{self.unsynced_counter:04d}_{now_local.strftime('%H%M%S')}.jpg"
                
            self.unsynced_counter += 1
        
        self.history_data.append({
            "time": timestamp_ui,
            "image": plate_img.copy()
        })

        if self.save_to_disk:
            filename = os.path.join(self.save_dir, filename_str)
            cv2.imwrite(filename, plate_img) 
            status_msg = f"ดึงภาพสำเร็จและบันทึกลงเครื่องแล้ว\nเมื่อ {timestamp_ui}"
            self.save_history_record(timestamp_ui, filename_str)
        else:
            status_msg = f"ดึงภาพเสร็จสิ้น\nเมื่อ {timestamp_ui}"
            
        self.root.after(0, self.update_detection_ui, plate_img, status_msg)

    def show_history(self):
        self.main_container.pack_forget()
        
        self.history_container = tk.Frame(self.root, bg="#F0F2F5")
        self.history_container.pack(fill="both", expand=True)
        
        hist_header = tk.Frame(self.history_container, bg=self.kmitl_orange, height=self.header_height)
        hist_header.pack(fill="x", side="top")
        hist_header.pack_propagate(False)
        
        back_btn = tk.Button(hist_header, text="< กลับหน้าหลัก", command=self.hide_history, 
                             font=(self.f_family, 18, "bold"), bg="#FFFFFF", fg=self.kmitl_orange, relief="flat", padx=20)
        back_btn.pack(side="left", padx=30, pady=25)
        
        tk.Label(hist_header, text="ประวัติการตรวจจับป้ายทะเบียน", font=(self.f_family, 36, "bold"), bg=self.kmitl_orange, fg="#FFFFFF").place(relx=0.5, rely=0.5, anchor="center")
        
        hist_content = tk.Frame(self.history_container, bg="#F0F2F5")
        hist_content.pack(fill="both", expand=True, padx=30, pady=30)
        
        list_frame = tk.Frame(hist_content, bg="#FFFFFF", width=350, highlightthickness=1, highlightbackground="#DDDDDD")
        list_frame.pack(side="left", fill="y", padx=(0, 20))
        list_frame.pack_propagate(False)
        
        tk.Label(list_frame, text="เวลาที่ตรวจพบ", font=(self.f_family, 18, "bold"), bg="#FFFFFF", fg="#333333").pack(pady=20)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.log_listbox = tk.Listbox(list_frame, font=(self.f_family, 14), yscrollcommand=scrollbar.set, 
                                      selectbackground=self.kmitl_orange, selectforeground="white", relief="flat", borderwidth=0)
        self.log_listbox.pack(side="left", fill="both", expand=True, padx=20, pady=(0, 20))
        scrollbar.config(command=self.log_listbox.yview)

        img_frame = tk.Frame(hist_content, bg="#FFFFFF", highlightthickness=1, highlightbackground="#DDDDDD")
        img_frame.pack(side="right", fill="both", expand=True)
        
        tk.Label(img_frame, text="รูปป้ายทะเบียน", font=(self.f_family, 22, "bold"), bg="#FFFFFF", fg="#333333").pack(pady=20)
        self.hist_img_label = tk.Label(img_frame, bg="#E9ECEF", text="คลิกที่เวลารายการด้านซ้าย\nเพื่อดูภาพป้ายทะเบียน", font=(self.f_family, 18), fg="#888888")
        self.hist_img_label.pack(expand=True, fill="both", padx=30, pady=(0, 30))

        for record in reversed(self.history_data):
            self.log_listbox.insert("end", record['time'])

        self.log_listbox.bind("<<ListboxSelect>>", self.on_history_select)

    def hide_history(self):
        self.history_container.destroy()
        self.main_container.pack(fill="both", expand=True)

    def on_history_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = len(self.history_data) - 1 - selection[0]
            record = self.history_data[index]
            
            img = record['image']
            plate_img_resized = cv2.resize(img, (600, 240), interpolation=cv2.INTER_LINEAR)
            img_pil = Image.fromarray(cv2.cvtColor(plate_img_resized, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img_pil)

            self.hist_img_label.configure(image=imgtk, text="")
            self.hist_img_label.image = imgtk

    def camera_thread(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            self.latest_frame = frame
            time.sleep(0.01)

    def ai_thread(self):
        while self.running:
            if self.latest_frame is None:
                time.sleep(0.01)
                continue

            ai_frame = self.latest_frame.copy()
            results = self.model(ai_frame, conf=0.45, imgsz=320, verbose=False)
            
            current_time = time.time()
            plate_detected_this_frame = False
            temp_boxes = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    temp_boxes.append((x1, y1, x2, y2, conf))

                    # 1. ถ้าระบบติด Cooldown อยู่ (เพิ่งเซฟไป) เราจะวาดกรอบเขียวเฉยๆ แต่ข้ามการเช็คเซฟซ้ำ
                    if current_time < self.global_cooldown:
                        continue

                    plate_detected_this_frame = True

                    y1_pad, y2_pad = max(0, y1-5), min(ai_frame.shape[0], y2+5)
                    x1_pad, x2_pad = max(0, x1-5), min(ai_frame.shape[1], x2+5)
                    plate_crop = ai_frame[y1_pad:y2_pad, x1_pad:x2_pad]

                    if plate_crop.size > 0:
                        sharpness = self.measure_clarity(plate_crop)
                        area = (x2_pad - x1_pad) * (y2_pad - y1_pad)
                        clarity_score = sharpness * area 

                        if self.active_plate is None:
                            self.active_plate = {'crop': plate_crop, 'score': clarity_score, 'last_seen': current_time, 'saved': False}
                            self.root.after(0, self.update_detection_ui, plate_crop, "พบป้ายทะเบียน กำลังประมวลผล...")
                        else:
                            self.active_plate['last_seen'] = current_time 
                            if clarity_score > self.active_plate['score']:
                                self.active_plate['crop'] = plate_crop
                                self.active_plate['score'] = clarity_score
                                self.root.after(0, self.update_detection_ui, plate_crop, "อัปเดตภาพที่ชัดเจนขึ้น...")

            # 2. ถ้ารถพ้นจอไปแล้ว หรือหมดเวลา Timeout ระบบถึงจะทำการเซฟภาพที่ดีที่สุด
            if not plate_detected_this_frame and self.active_plate is not None:
                if current_time - self.active_plate['last_seen'] > self.plate_timeout:
                    if not self.active_plate['saved']:
                        self.process_best_plate(self.active_plate['crop'])
                        self.active_plate['saved'] = True
                        
                        # 3. เซฟเสร็จ สั่งล็อค Cooldown 3.5 วินาทีเพื่อรอให้รถคันเดิมขับผ่านให้พ้นเฟรม
                        self.global_cooldown = current_time + 3.5
                    
                    self.active_plate = None 

            self.latest_boxes = temp_boxes
            time.sleep(0.01)

    def update_gui_loop(self):
        if self.running and self.latest_frame is not None:
            display_frame = self.latest_frame.copy()
            
            for (x1, y1, x2, y2, conf) in self.latest_boxes:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Plate {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if self.full_screen_video:
                target_w = self.screen_width - self.right_panel_width - 40 
                target_h = self.screen_height - self.header_height - 40
                if target_w > 0 and target_h > 0:
                    display_frame = cv2.resize(display_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img) 
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_container.configure(image=imgtk)
            self.video_container.image = imgtk

        if self.running:
            self.root.after(60, self.update_gui_loop)

    def update_detection_ui(self, plate_img, status_text):
        plate_img_resized = cv2.resize(plate_img, (400, 160), interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(cv2.cvtColor(plate_img_resized, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.crop_display.configure(image=imgtk)
        self.crop_display.image = imgtk
        
        if "สำเร็จ" in status_text or "เสร็จสิ้น" in status_text or "ล่าสุด" in status_text:
            self.status_label.config(text=status_text, fg="#28A745") 
        else:
            self.status_label.config(text=status_text, fg="#007BFF") 

    def on_exit(self):
        self.running = False
        if self.cap.isOpened(): self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ALPRSystem(root)
    root.bind("<Escape>", lambda e: app.on_exit())
    root.mainloop()