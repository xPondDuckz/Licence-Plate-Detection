import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import csv
from datetime import datetime
import pytz

class ALPRSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("ระบบตรวจจับป้ายทะเบียน - KMITL PCC")
        
        # บังคับขนาด 1920x1080 เต็มจอ
        self.root.geometry("1920x1080")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#FDFDFD")
        
        # --- การตั้งค่าฟอนต์และสี ---
        self.f_family = "IBM Plex Sans Thai Looped" 
        self.font_title = (self.f_family, 50, "bold")
        self.font_header = (self.f_family, 24, "bold")
        self.font_btn = (self.f_family, 20, "bold")
        self.font_plate = (self.f_family, 130, "bold")
        
        self.kmitl_orange = "#FF6600" 
        self.btn_green = "#27AE60"  # สีเขียวมรกต
        self.btn_red = "#C0392B"    # สีแดงเข้ม
        
        self.logo_path = "/home/sunlight-lnwza007/Downloads/logo.png" 
        self.log_file = "license_log.csv"

        # Main Container
        self.main_container = tk.Frame(self.root, bg="#FDFDFD")
        self.main_container.pack(fill="both", expand=True)

        self.frames = {}
        for F in (StartPage, LogPage):
            page_name = F.__name__
            frame = F(parent=self.main_container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        if page_name == "LogPage":
            frame.refresh_logs()

# --- หน้าหลัก ---
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#FDFDFD")
        self.controller = controller

        # 1. Header (ปรับขนาดให้กระชับขึ้นเพื่อเพิ่มพื้นที่หน้าต่างกลาง)
        self.header = tk.Frame(self, bg=controller.kmitl_orange, height=180)
        self.header.pack(fill="x", side="top")
        self.header.pack_propagate(False)

        # โลโก้ขนาดใหญ่
        try:
            load = Image.open(controller.logo_path)
            load = load.resize((160, 160), Image.Resampling.LANCZOS)
            render = ImageTk.PhotoImage(load)
            logo = tk.Label(self.header, image=render, bg=controller.kmitl_orange)
            logo.image = render
            logo.place(x=80, y=10)
        except: pass

        title = tk.Label(self.header, text="ระบบตรวจจับป้ายทะเบียน - KMITL PCC", 
                         font=controller.font_title, bg=controller.kmitl_orange, fg="white")
        title.pack(expand=True)

        # 2. Content Area (ขยายให้ใหญ่ขึ้นโดยลด Padding และเพิ่ม Weight)
        content_frame = tk.Frame(self, bg="#FDFDFD")
        content_frame.pack(fill="both", expand=True, padx=40, pady=10)
        
        content_frame.grid_columnconfigure(0, weight=1, uniform="group1")
        content_frame.grid_columnconfigure(1, weight=1, uniform="group1")
        content_frame.grid_rowconfigure(0, weight=1)

        # ฝั่งซ้าย: Camera Stream (ขยายขอบให้สุด)
        left_box = tk.LabelFrame(content_frame, text=" CAMERA STREAM ", 
                                font=controller.font_header, bg="white", fg="#2C3E50", bd=2)
        left_box.grid(row=0, column=0, padx=15, pady=10, sticky="nsew")
        
        self.video_label = tk.Label(left_box, bg="#000000")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # ฝั่งขวา: Recognition Panel (ขยายขอบให้สุด)
        right_box = tk.LabelFrame(content_frame, text=" RECOGNITION PANEL ", 
                                 font=controller.font_header, bg="white", fg="#2C3E50", bd=2)
        right_box.grid(row=0, column=1, padx=15, pady=10, sticky="nsew")

        # ส่วนรูปที่จับได้ (ขยายพื้นที่ภายใน)
        self.plate_img_label = tk.Label(right_box, text="AWAITING DETECTION", 
                                        font=controller.font_header, bg="#F8F9F9", fg="#BDC3C7")
        self.plate_img_label.pack(fill="both", expand=True, padx=25, pady=25)

        self.plate_text_var = tk.StringVar(value="---")
        plate_display = tk.Label(right_box, textvariable=self.plate_text_var, 
                                 font=controller.font_plate, bg="white", fg=controller.kmitl_orange)
        plate_display.pack(side="bottom", pady=50)

        # 3. Footer (ปรับขนาดปุ่มและลดระยะขอบลงเล็กน้อย)
        footer = tk.Frame(self, bg="#FDFDFD", height=130)
        footer.pack(fill="x", side="bottom", pady=30)

        btn_container = tk.Frame(footer, bg="#FDFDFD")
        btn_container.pack(expand=True)

        # ปุ่มประวัติ (สีเขียวตามสั่ง)
        btn_log = tk.Button(btn_container, text="ประวัติการบันทึกข้อมูล", 
                            command=lambda: controller.show_frame("LogPage"),
                            bg=controller.btn_green, fg="white", font=controller.font_btn,
                            width=20, pady=12, relief="flat", cursor="hand2")
        btn_log.pack(side="left", padx=50)

        # ปุ่มปิดระบบ (สีแดง)
        btn_exit = tk.Button(btn_container, text="ปิดระบบปฏิบัติการ", 
                             command=self.confirm_exit,
                             bg=controller.btn_red, fg="white", font=controller.font_btn,
                             width=20, pady=12, relief="flat", cursor="hand2")
        btn_exit.pack(side="left", padx=50)

        self.cap = cv2.VideoCapture(0)
        self.update_webcam()

    def confirm_exit(self):
        if messagebox.askokcancel("ยืนยัน", "คุณต้องการออกจากระบบหรือไม่?"):
            self.controller.root.destroy()

    def update_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            # ขยายขนาดภาพพรีวิวให้สัมพันธ์กับหน้าต่างที่ใหญ่ขึ้น
            display_frame = cv2.resize(frame, (1000, 600))
            cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.after(15, self.update_webcam)

# --- หน้าประวัติ ---
class LogPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="white")
        self.controller = controller
        
        header = tk.Frame(self, bg=controller.btn_green, height=180)
        header.pack(fill="x")
        tk.Label(header, text="Log History - ข้อมูลป้ายทะเบียน", font=controller.font_title, 
                 bg=controller.btn_green, fg="white").pack(expand=True)
        
        style = ttk.Style()
        style.configure("Treeview", font=(controller.f_family, 18), rowheight=50)
        style.configure("Treeview.Heading", font=(controller.f_family, 20, "bold"))

        self.tree = ttk.Treeview(self, columns=("Plate", "Date", "Time"), show='headings')
        for col in ("Plate", "Date", "Time"):
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=400)
        self.tree.pack(fill="both", expand=True, padx=100, pady=40)
        
        tk.Button(self, text="ย้อนกลับ", command=lambda: controller.show_frame("StartPage"),
                  font=controller.font_btn, bg=controller.btn_green, fg="white", 
                  width=15, pady=10, relief="flat").pack(pady=40)

    def refresh_logs(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        try:
            with open(self.controller.log_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    self.tree.insert("", 0, values=row)
        except: pass

if __name__ == "__main__":
    root = tk.Tk()
    app = ALPRSystem(root)
    root.mainloop()