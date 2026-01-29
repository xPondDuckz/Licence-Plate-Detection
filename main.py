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
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#F0F2F5")
        
        # --- Config & Colors ---
        self.f_family = "IBM Plex Sans Thai Looped" 
        self.kmitl_orange = "#FF6600" 
        self.btn_green = "#27AE60"
        self.btn_green_hover = "#2ECC71"
        self.btn_red = "#C0392B"
        self.btn_red_hover = "#E74C3C"
        self.midnight_blue = "#2C3E50"
        self.tz = pytz.timezone('Asia/Bangkok') 
        
        # Paths (ตรวจสอบให้แน่ใจว่าไฟล์เหล่านี้มีอยู่จริง)
        self.logo_path = "/home/sunlight-lnwza007/Downloads/logo.png" 
        self.history_icon_path = "/home/sunlight-lnwza007/Downloads/history.png"
        self.exit_icon_path = "/home/sunlight-lnwza007/Downloads/exit.png"
        self.log_file = "license_log.csv"

        # --- Fonts ---
        self.font_title = (self.f_family, 48, "bold")
        self.font_header = (self.f_family, 24, "bold")
        self.font_btn = (self.f_family, 20, "bold")
        self.font_plate = (self.f_family, 130, "bold")
        self.font_time_header = (self.f_family, 20, "bold")

        # --- Load Icons ---
        self.icon_history = self.load_icon(self.history_icon_path, (40, 40))
        self.icon_exit = self.load_icon(self.exit_icon_path, (40, 40))

        # Main Container
        self.main_container = tk.Frame(self.root, bg="#F0F2F5")
        self.main_container.pack(fill="both", expand=True)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, LogPage):
            page_name = F.__name__
            frame = F(parent=self.main_container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def load_icon(self, path, size):
        try:
            if os.path.exists(path):
                return ImageTk.PhotoImage(Image.open(path).resize(size, Image.Resampling.LANCZOS))
        except: pass
        return None

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        if page_name == "LogPage":
            frame.refresh_logs()

# --- หน้าหลัก (StartPage) ---
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#F0F2F5")
        self.controller = controller

        # 1. Header (Logo | Title | Time)
        self.header = tk.Frame(self, bg=controller.kmitl_orange, height=220)
        self.header.pack(fill="x", side="top")
        self.header.pack_propagate(False)

        # Logo
        try:
            load = Image.open(controller.logo_path).resize((185, 185), Image.Resampling.LANCZOS)
            render = ImageTk.PhotoImage(load)
            logo = tk.Label(self.header, image=render, bg=controller.kmitl_orange)
            logo.image = render
            logo.place(x=50, rely=0.5, anchor="w")
        except: pass

        # Title
        tk.Label(self.header, text="ระบบตรวจจับป้ายทะเบียน - KMITL PCC", 
                 font=controller.font_title, bg=controller.kmitl_orange, fg="white").place(relx=0.5, rely=0.5, anchor="center")

        # Time (จัดตำแหน่งใหม่แบบ 2 บรรทัด)
        self.time_label = tk.Label(self.header, text="", font=controller.font_time_header, 
                                   bg=controller.kmitl_orange, fg="white", justify="right")
        self.time_label.place(relx=0.97, rely=0.5, anchor="e")

        # 2. Content Area
        content = tk.Frame(self, bg="#F0F2F5")
        content.pack(fill="both", expand=True, padx=50, pady=20)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=1)

        # Left: Live Stream
        left_card = tk.Frame(content, bg="white", highlightthickness=1, highlightbackground="#DCDDE1")
        left_card.grid(row=0, column=0, padx=15, pady=10, sticky="nsew")
        tk.Label(left_card, text="LIVE STREAM", font=controller.font_header, bg="white", fg=controller.midnight_blue).pack(anchor="w", padx=20, pady=10)
        self.video_label = tk.Label(left_card, bg="#1A1A1B")
        self.video_label.pack(fill="both", expand=True, padx=20, pady=20)

        # Right: Result
        right_card = tk.Frame(content, bg="white", highlightthickness=1, highlightbackground="#DCDDE1")
        right_card.grid(row=0, column=1, padx=15, pady=10, sticky="nsew")
        tk.Label(right_card, text="ผลการตรวจพบล่าสุด", font=controller.font_header, bg="white", fg=controller.midnight_blue).pack(anchor="w", padx=20, pady=10)
        self.plate_img_label = tk.Label(right_card, text="AWAITING DETECTION", font=controller.font_header, bg="#F8F9F9", fg="#BDC3C7")
        self.plate_img_label.pack(fill="both", expand=True, padx=25, pady=10)
        self.plate_text_var = tk.StringVar(value="---")
        tk.Label(right_card, textvariable=self.plate_text_var, font=controller.font_plate, bg="white", fg=controller.kmitl_orange).pack(side="bottom", pady=40)

        # 3. Footer Buttons
        footer = tk.Frame(self, bg="#F0F2F5", height=120)
        footer.pack(fill="x", side="bottom", pady=20)
        btn_box = tk.Frame(footer, bg="#F0F2F5")
        btn_box.pack(expand=True)

        self.btn_history = self.create_btn(btn_box, " ประวัติการบันทึก", controller.icon_history, 
                                           controller.btn_green, controller.btn_green_hover, 
                                           lambda: controller.show_frame("LogPage"), 320)
        self.btn_history.pack(side="left", padx=30)

        self.btn_exit = self.create_btn(btn_box, " ออกจากโปรแกรม", controller.icon_exit, 
                                        controller.btn_red, controller.btn_red_hover, 
                                        self.confirm_exit, 300)
        self.btn_exit.pack(side="left", padx=30)

        self.cap = cv2.VideoCapture(0)
        self.update_ui()

    def create_btn(self, parent, text, img, color, hover_color, cmd, w):
        btn = tk.Button(parent, text=text, image=img, compound="left", command=cmd,
                        bg=color, fg="white", font=self.controller.font_btn, 
                        bd=0, relief="flat", width=w, pady=15, cursor="hand2")
        btn.bind("<Enter>", lambda e: btn.config(bg=hover_color))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        return btn

    def confirm_exit(self):
        if messagebox.askokcancel("ยืนยัน", "คุณต้องการออกจากโปรแกรมหรือไม่?"):
            self.controller.root.destroy()

    def update_ui(self):
        ret, frame = self.cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)).resize((900, 550), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        # วันที่ภาษาไทย พ.ศ. 2569
        now = datetime.now(self.controller.tz)
        months_th = [
            "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
            "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"
        ]
        date_str = f"{now.day} {months_th[now.month-1]} {now.year + 543}"
        time_str = now.strftime("%H:%M:%S น.")
        self.time_label.config(text=f"{date_str}\n{time_str}")
        self.after(1000, self.update_ui)

# --- หน้าประวัติ (ดีไซน์ภาษาไทย + สถานะสีแดงเมื่อว่าง) ---
class LogPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#F0F2F5")
        self.controller = controller
        f_main = controller.f_family

        # 1. Header (ภาษาไทย)
        header_nav = tk.Frame(self, bg=controller.midnight_blue, height=120)
        header_nav.pack(fill="x")
        header_nav.pack_propagate(False)
        tk.Label(header_nav, text="บันทึกประวัติการเข้า-ออกยานพาหนะ", font=(f_main, 35, "bold"), 
                 bg=controller.midnight_blue, fg="white").pack(expand=True)

        # 2. Main Layout
        container = tk.Frame(self, bg="#F0F2F5")
        container.pack(fill="both", expand=True, padx=50, pady=30)
        container.grid_columnconfigure(0, weight=2)
        container.grid_columnconfigure(1, weight=1)
        container.grid_rowconfigure(0, weight=1)

        # LEFT: Table
        left_pane = tk.Frame(container, bg="white", highlightthickness=1, highlightbackground="#DCDDE1")
        left_pane.grid(row=0, column=0, padx=15, sticky="nsew")
        tk.Label(left_pane, text=" รายการที่บันทึกทั้งหมด", font=(f_main, 18, "bold"), bg="white", fg=controller.midnight_blue).pack(anchor="w", padx=20, pady=15)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", rowheight=55, font=(f_main, 15), borderwidth=0)
        style.configure("Treeview.Heading", background="#F8F9FA", font=(f_main, 16, "bold"), relief="flat")
        style.map("Treeview", background=[('selected', controller.kmitl_orange)])

        self.tree = ttk.Treeview(left_pane, columns=("Plate", "Date", "Time", "Status"), show='headings')
        self.tree.heading("Plate", text="ป้ายทะเบียน")
        self.tree.heading("Date", text="วันที่")
        self.tree.heading("Time", text="เวลา")
        self.tree.heading("Status", text="สถานะ")
        self.tree.column("Plate", anchor="center")
        self.tree.column("Date", anchor="center")
        self.tree.column("Time", anchor="center")
        self.tree.column("Status", anchor="center")
        self.tree.pack(fill="both", expand=True, padx=20, pady=20)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        # RIGHT: Preview Details (สถานะเริ่มต้นสีแดง)
        right_pane = tk.Frame(container, bg="white", highlightthickness=1, highlightbackground="#DCDDE1")
        right_pane.grid(row=0, column=1, padx=15, sticky="nsew")
        tk.Label(right_pane, text=" รายละเอียดประกอบ", font=(f_main, 18, "bold"), bg="white", fg=controller.midnight_blue).pack(anchor="w", padx=20, pady=15)
        
        self.img_preview = tk.Label(right_pane, text="ยังไม่พบไฟล์ภาพประกอบ", bg="#F8F9FA", fg="#BDC3C7", height=15)
        self.img_preview.pack(fill="x", padx=20, pady=10)

        self.lbl_plate = tk.Label(right_pane, text="ทะเบียน: ยังไม่พบข้อมูล", font=(f_main, 24, "bold"), bg="white", fg="#C0392B")
        self.lbl_plate.pack(pady=10)
        self.lbl_status = tk.Label(right_pane, text="สถานะ: ไม่มีข้อมูล", font=(f_main, 18), bg="white", fg="#C0392B")
        self.lbl_status.pack()

        # 3. Footer
        footer = tk.Frame(self, bg="#F0F2F5", height=100)
        footer.pack(fill="x", side="bottom")
        btn_back = tk.Button(footer, text=" ย้อนกลับสู่หน้าหลัก ", command=lambda: controller.show_frame("StartPage"),
                             font=controller.font_btn, bg=controller.midnight_blue, fg="white", 
                             bd=0, width=22, pady=12, cursor="hand2")
        btn_back.pack(pady=20)

    def on_select(self, event):
        selected = self.tree.focus()
        if not selected: return
        data = self.tree.item(selected, 'values')
        
        # เปลี่ยนเป็นสีส้ม KMITL เมื่อเลือก
        self.lbl_plate.config(text=f"ทะเบียน: {data[0]}", fg=self.controller.kmitl_orange)
        
        status_text = data[3]
        status_color = "#27AE60" if status_text == "เข้า" else "#C0392B"
        self.lbl_status.config(text=f"สถานะ: {status_text}", fg=status_color)

        img_path = f"images/{data[0]}.jpg"
        if os.path.exists(img_path):
            img = Image.open(img_path).resize((350, 250), Image.Resampling.LANCZOS)
            render = ImageTk.PhotoImage(img)
            self.img_preview.config(image=render, text="")
            self.img_preview.image = render
        else:
            self.img_preview.config(image="", text="ไม่พบไฟล์ภาพในระบบ")

    def refresh_logs(self):
        # รีเซ็ตเป็นสีแดงเมื่อโหลดหน้าใหม่
        self.lbl_plate.config(text="ทะเบียน: ยังไม่พบข้อมูล", fg="#C0392B")
        self.lbl_status.config(text="สถานะ: ไม่มีข้อมูล", fg="#C0392B")
        self.img_preview.config(image="", text="ยังไม่พบไฟล์ภาพประกอบ")
        
        for item in self.tree.get_children(): self.tree.delete(item)
        try:
            if os.path.exists(self.controller.log_file):
                with open(self.controller.log_file, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reversed(list(reader)):
                        clean_row = [str(c).replace("🚗", "").strip() for c in row]
                        self.tree.insert("", "end", values=clean_row)
        except: pass

if __name__ == "__main__":
    root = tk.Tk()
    app = ALPRSystem(root)
    root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
    root.mainloop()