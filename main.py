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
        
        # --- Config & Paths ---
        self.f_family = "IBM Plex Sans Thai Looped" 
        self.kmitl_orange = "#FF6600" 
        self.btn_green = "#27AE60"
        self.btn_red = "#C0392B"
        self.midnight_blue = "#2C3E50"
        self.tz = pytz.timezone('Asia/Bangkok') 
        
        self.logo_path = "/home/sunlight-lnwza007/Downloads/logo.png" 
        self.history_icon_path = "/home/sunlight-lnwza007/Downloads/history.png"
        self.exit_icon_path = "/home/sunlight-lnwza007/Downloads/exit.png"
        self.log_file = "license_log.csv" # ไฟล์ CSV ควรมีคอลัมน์: Plate, Date, Time, Status, Image_Path

        # --- Fonts ---
        self.font_title = (self.f_family, 52, "bold")
        self.font_header = (self.f_family, 22, "bold")
        self.font_btn = (self.f_family, 18, "bold")
        self.font_plate = (self.f_family, 120, "bold")

        # --- Load Icons ---
        self.icon_history = self.load_icon(self.history_icon_path, (40, 40))
        self.icon_exit = self.load_icon(self.exit_icon_path, (40, 40))

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

        # Header
        self.header = tk.Frame(self, bg=controller.kmitl_orange, height=200)
        self.header.pack(fill="x", side="top")
        self.header.pack_propagate(False)

        try:
            load = Image.open(controller.logo_path).resize((170, 170), Image.Resampling.LANCZOS)
            render = ImageTk.PhotoImage(load)
            logo = tk.Label(self.header, image=render, bg=controller.kmitl_orange)
            logo.image = render
            logo.place(x=50, rely=0.5, anchor="w")
        except: pass

        tk.Label(self.header, text="ระบบตรวจจับป้ายทะเบียน - KMITL PCC", 
                 font=controller.font_title, bg=controller.kmitl_orange, fg="white").place(relx=0.5, rely=0.5, anchor="center")

        self.time_label = tk.Label(self.header, text="", font=(controller.f_family, 18, "bold"), bg=controller.kmitl_orange, fg="white")
        self.time_label.place(relx=0.98, rely=0.15, anchor="ne")

        # Content
        content = tk.Frame(self, bg="#F0F2F5")
        content.pack(fill="both", expand=True, padx=50, pady=20)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=1)

        # Camera
        left_card = tk.Frame(content, bg="white", highlightthickness=1, highlightbackground="#DCDDE1")
        left_card.grid(row=0, column=0, padx=15, pady=10, sticky="nsew")
        self.video_label = tk.Label(left_card, bg="#1A1A1B")
        self.video_label.pack(fill="both", expand=True, padx=20, pady=20)

        # Plate Result
        right_card = tk.Frame(content, bg="white", highlightthickness=1, highlightbackground="#DCDDE1")
        right_card.grid(row=0, column=1, padx=15, pady=10, sticky="nsew")
        self.plate_text_var = tk.StringVar(value="---")
        tk.Label(right_card, textvariable=self.plate_text_var, font=controller.font_plate, bg="white", fg=controller.kmitl_orange).pack(expand=True)

        # Footer
        footer = tk.Frame(self, bg="#F0F2F5", height=100)
        footer.pack(fill="x", side="bottom", pady=20)
        btn_box = tk.Frame(footer, bg="#F0F2F5")
        btn_box.pack(expand=True)

        tk.Button(btn_box, text=" ประวัติการบันทึก", image=controller.icon_history, compound="left",
                  command=lambda: controller.show_frame("LogPage"), bg=controller.btn_green, fg="white", font=controller.font_btn, width=300, pady=12).pack(side="left", padx=20)
        
        tk.Button(btn_box, text=" ออกจากโปรแกรม", image=controller.icon_exit, compound="left",
                  command=lambda: self.controller.root.destroy(), bg=controller.btn_red, fg="white", font=controller.font_btn, width=280, pady=12).pack(side="left", padx=20)

        self.cap = cv2.VideoCapture(0)
        self.update_ui()

    def update_ui(self):
        ret, frame = self.cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)).resize((900, 550), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.time_label.config(text=datetime.now(self.controller.tz).strftime("%d/%m/%Y | %H:%M:%S"))
        self.after(20, self.update_ui)

# --- หน้าประวัติ (ดีไซน์ใหม่: Multi-Panel Log) ---
class LogPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#F0F2F5")
        self.controller = controller
        f_main = controller.f_family

        # Header Navigation
        nav = tk.Frame(self, bg=controller.midnight_blue, height=100)
        nav.pack(fill="x")
        tk.Label(nav, text="Vehicle Access History Log", font=(f_main, 28, "bold"), bg=controller.midnight_blue, fg="white").pack(expand=True)

        # Main Layout: Table (Left) | Details (Right)
        self.main_area = tk.Frame(self, bg="#F0F2F5")
        self.main_area.pack(fill="both", expand=True, padx=40, pady=30)
        self.main_area.grid_columnconfigure(0, weight=2) # Table area
        self.main_area.grid_columnconfigure(1, weight=1) # Preview area
        self.main_area.grid_rowconfigure(0, weight=1)

        # --- LEFT: Table Card ---
        table_card = tk.Frame(self.main_area, bg="white", highlightthickness=1, highlightbackground="#DCDDE1")
        table_card.grid(row=0, column=0, padx=15, sticky="nsew")

        # Table Header/Title
        tk.Label(table_card, text=" รายการทั้งหมด", font=(f_main, 18, "bold"), bg="white", fg=controller.midnight_blue).pack(anchor="w", padx=20, pady=15)

        style = ttk.Style()
        style.configure("Treeview", rowheight=50, font=(f_main, 14))
        style.configure("Treeview.Heading", font=(f_main, 16, "bold"))
        
        self.tree = ttk.Treeview(table_card, columns=("Plate", "Date", "Time", "Status"), show='headings')
        self.tree.heading("Plate", text="ป้ายทะเบียน")
        self.tree.heading("Date", text="วันที่")
        self.tree.heading("Time", text="เวลา")
        self.tree.heading("Status", text="สถานะ")
        
        self.tree.column("Plate", anchor="center", width=150)
        self.tree.column("Date", anchor="center", width=120)
        self.tree.column("Time", anchor="center", width=100)
        self.tree.column("Status", anchor="center", width=100)
        
        self.tree.pack(fill="both", expand=True, padx=20, pady=10)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        # --- RIGHT: Preview Card ---
        self.preview_card = tk.Frame(self.main_area, bg="white", highlightthickness=1, highlightbackground="#DCDDE1")
        self.preview_card.grid(row=0, column=1, padx=15, sticky="nsew")

        tk.Label(self.preview_card, text=" รายละเอียดประกอบ", font=(f_main, 18, "bold"), bg="white", fg=controller.midnight_blue).pack(anchor="w", padx=20, pady=15)
        
        # Image Preview Placeholder
        self.img_preview = tk.Label(self.preview_card, text="Select a record to view image", bg="#F8F9FA", fg="#BDC3C7", width=40, height=15)
        self.img_preview.pack(fill="x", padx=20, pady=10)

        # Info Labels
        self.lbl_plate = tk.Label(self.preview_card, text="ทะเบียน: -", font=(f_main, 20, "bold"), bg="white", fg=controller.kmitl_orange)
        self.lbl_plate.pack(pady=10)
        self.lbl_status = tk.Label(self.preview_card, text="สถานะ: -", font=(f_main, 16), bg="white")
        self.lbl_status.pack()

        # Footer
        tk.Button(self, text="ย้อนกลับ", command=lambda: controller.show_frame("StartPage"),
                  font=controller.font_btn, bg=controller.midnight_blue, fg="white", width=15, pady=10).pack(pady=20)

    def on_select(self, event):
        selected = self.tree.focus()
        if not selected: return
        data = self.tree.item(selected, 'values')
        
        # อัปเดตตัวหนังสือ
        self.lbl_plate.config(text=f"ทะเบียน: {data[0]}")
        status_color = "#27AE60" if data[3] == "เข้า" else "#C0392B"
        self.lbl_status.config(text=f"สถานะ: {data[3]}", fg=status_color)

        # จำลองการโหลดรูปภาพ (หากมี Path รูปใน CSV)
        # ในที่นี้จะลองหาบไฟล์จากโฟลเดอร์ images/ทะเบียน.jpg
        img_path = f"images/{data[0]}.jpg"
        if os.path.exists(img_path):
            img = Image.open(img_path).resize((350, 250), Image.Resampling.LANCZOS)
            render = ImageTk.PhotoImage(img)
            self.img_preview.config(image=render, text="")
            self.img_preview.image = render
        else:
            self.img_preview.config(image="", text="[ NO IMAGE FOUND ]")

    def refresh_logs(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        if os.path.exists(self.controller.log_file):
            with open(self.controller.log_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reversed(list(reader)):
                    # row คาดหวัง: [Plate, Date, Time, Status]
                    self.tree.insert("", "end", values=row)

if __name__ == "__main__":
    root = tk.Tk()
    app = ALPRSystem(root)
    root.mainloop()