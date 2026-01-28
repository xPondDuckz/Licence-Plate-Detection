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
        self.root.configure(bg="#F0F2F5")
        
        # --- การตั้งค่าฟอนต์และสี ---
        self.f_family = "IBM Plex Sans Thai Looped" 
        self.font_title = (self.f_family, 50, "bold")
        self.font_header = (self.f_family, 24, "bold")
        self.font_btn = (self.f_family, 20, "bold")
        self.font_plate = (self.f_family, 130, "bold")
        
        self.kmitl_orange = "#FF6600" 
        self.btn_green = "#27AE60"  
        self.btn_red = "#C0392B"    
        self.midnight_blue = "#2C3E50"
        
        self.logo_path = "/home/sunlight-lnwza007/Downloads/logo.png" 
        self.log_file = "license_log.csv"

        # Main Container
        self.main_container = tk.Frame(self.root, bg="#F0F2F5")
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

# --- หน้าหลัก (Dashboard) ---
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#F0F2F5")
        self.controller = controller

        # 1. Header (ขยายความสูงรองรับโลโก้ใหญ่)
        self.header = tk.Frame(self, bg=controller.kmitl_orange, height=220)
        self.header.pack(fill="x", side="top")
        self.header.pack_propagate(False)

        # โลโก้ขนาดใหญ่พิเศษ (200x200)
        try:
            load = Image.open(controller.logo_path)
            load = load.resize((200, 200), Image.Resampling.LANCZOS)
            render = ImageTk.PhotoImage(load)
            logo = tk.Label(self.header, image=render, bg=controller.kmitl_orange)
            logo.image = render
            logo.place(x=60, y=10)
        except: pass

        title = tk.Label(self.header, text="ระบบตรวจจับป้ายทะเบียน - KMITL PCC", 
                         font=controller.font_title, bg=controller.kmitl_orange, fg="white")
        title.pack(expand=True)

        # 2. Content Area (ปรับให้หน้าต่างกลางใหญ่ขึ้น)
        content_frame = tk.Frame(self, bg="#F0F2F5")
        content_frame.pack(fill="both", expand=True, padx=40, pady=15)
        
        content_frame.grid_columnconfigure(0, weight=1, uniform="group1")
        content_frame.grid_columnconfigure(1, weight=1, uniform="group1")
        content_frame.grid_rowconfigure(0, weight=1)

        # ฝั่งซ้าย: Camera Stream
        left_box = tk.LabelFrame(content_frame, text=" CAMERA STREAM ", 
                                font=controller.font_header, bg="white", fg="#34495E", bd=0)
        left_box.grid(row=0, column=0, padx=15, pady=10, sticky="nsew")
        
        self.video_label = tk.Label(left_box, bg="#1A1A1B")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # ฝั่งขวา: Recognition Panel
        right_box = tk.LabelFrame(content_frame, text=" RECOGNITION PANEL ", 
                                 font=controller.font_header, bg="white", fg="#34495E", bd=0)
        right_box.grid(row=0, column=1, padx=15, pady=10, sticky="nsew")

        self.plate_img_label = tk.Label(right_box, text="AWAITING DETECTION", 
                                        font=controller.font_header, bg="#F8F9F9", fg="#BDC3C7")
        self.plate_img_label.pack(fill="both", expand=True, padx=25, pady=25)

        self.plate_text_var = tk.StringVar(value="---")
        plate_display = tk.Label(right_box, textvariable=self.plate_text_var, 
                                 font=controller.font_plate, bg="white", fg=controller.kmitl_orange)
        plate_display.pack(side="bottom", pady=50)

        # 3. Footer
        footer = tk.Frame(self, bg="#F0F2F5", height=130)
        footer.pack(fill="x", side="bottom", pady=30)

        btn_container = tk.Frame(footer, bg="#F0F2F5")
        btn_container.pack(expand=True)

        btn_log = tk.Button(btn_container, text="ประวัติการบันทึกข้อมูล", 
                            command=lambda: controller.show_frame("LogPage"),
                            bg=controller.btn_green, fg="white", font=controller.font_btn,
                            width=22, pady=15, relief="flat", cursor="hand2")
        btn_log.pack(side="left", padx=50)

        btn_exit = tk.Button(btn_container, text="ปิดระบบปฏิบัติการ", 
                             command=self.confirm_exit,
                             bg=controller.btn_red, fg="white", font=controller.font_btn,
                             width=22, pady=15, relief="flat", cursor="hand2")
        btn_exit.pack(side="left", padx=50)

        self.cap = cv2.VideoCapture(0)
        self.update_webcam()

    def confirm_exit(self):
        if messagebox.askokcancel("ยืนยัน", "คุณต้องการออกจากระบบหรือไม่?"):
            self.controller.root.destroy()

    def update_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            display_frame = cv2.resize(frame, (1000, 600))
            cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.after(15, self.update_webcam)

# --- หน้าประวัติ (ดีไซน์ใหม่) ---
class LogPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#F0F2F5")
        self.controller = controller
        f_main = controller.f_family

        # Header หน้า Log แบบ Dashboard
        header_log = tk.Frame(self, bg=controller.midnight_blue, height=180)
        header_log.pack(fill="x")
        tk.Label(header_log, text="ประวัติข้อมูลป้ายทะเบียนรถยนต์", font=controller.font_title, 
                 bg=controller.midnight_blue, fg="white").pack(expand=True)

        # Content Area สำหรับตาราง
        content_log = tk.Frame(self, bg="#F0F2F5")
        content_log.pack(fill="both", expand=True, padx=80, pady=40)

        # ปรับแต่ง Style ของตารางให้ดูพรีเมียม (ไม่เอาแบบ Excel)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", 
                        background="#FFFFFF", 
                        foreground="#34495E", 
                        rowheight=60, 
                        font=(f_main, 18),
                        borderwidth=0)
        style.configure("Treeview.Heading", 
                        background="#ECF0F1", 
                        foreground=controller.midnight_blue, 
                        font=(f_main, 20, "bold"),
                        relief="flat")
        style.map("Treeview", background=[('selected', controller.kmitl_orange)])

        # สร้างตาราง
        self.tree = ttk.Treeview(content_log, columns=("Plate", "Date", "Time"), show='headings')
        self.tree.heading("Plate", text=" หมายเลขป้ายทะเบียน ")
        self.tree.heading("Date", text=" วันที่ตรวจพบ ")
        self.tree.heading("Time", text=" เวลาที่บันทึก ")
        
        self.tree.column("Plate", anchor="center", width=400)
        self.tree.column("Date", anchor="center", width=300)
        self.tree.column("Time", anchor="center", width=300)
        
        self.tree.pack(fill="both", expand=True)

        # ปุ่มกดย้อนกลับ
        footer_log = tk.Frame(self, bg="#F0F2F5", height=120)
        footer_log.pack(fill="x", side="bottom", pady=40)
        
        tk.Button(footer_log, text="ย้อนกลับสู่หน้าหลัก", 
                  command=lambda: controller.show_frame("StartPage"),
                  font=controller.font_btn, bg=controller.midnight_blue, fg="white", 
                  width=20, pady=12, relief="flat", cursor="hand2").pack(expand=True)

    def refresh_logs(self):
        # ล้างข้อมูลเดิม
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        try:
            if os.path.exists(self.controller.log_file):
                with open(self.controller.log_file, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    next(reader) # ข้าม header
                    
                    # อ่านข้อมูลมาใส่ และสลับสีแถวเพื่อให้ดูสวยงาม
                    rows = list(reader)
                    for i, row in enumerate(reversed(rows)): # เอาล่าสุดไว้บน
                        tag = 'even' if i % 2 == 0 else 'odd'
                        self.tree.insert("", "end", values=row, tags=(tag,))
                
                # กำหนดสีแถวสลับกัน (Striped effect)
                self.tree.tag_configure('even', background='#FFFFFF')
                self.tree.tag_configure('odd', background='#F9FAFB')
        except: pass

if __name__ == "__main__":
    root = tk.Tk()
    app = ALPRSystem(root)
    root.mainloop()