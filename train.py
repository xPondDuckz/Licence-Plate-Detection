from ultralytics import YOLO

if __name__ == '__main__':
    # 1. โหลดโมเดล YOLO11 Nano
    print("--- Loading Model (GPU Mode) ---")
    model = YOLO('yolo11n.pt') 

    # 2. เริ่มเทรน
    print("--- Starting Training with NVIDIA GPU ---")
    results = model.train(
        data=r'C:\Users\Admin\Downloads\LPR plate.v1i.yolov8\data.yaml', 
        epochs=100, 
        imgsz=640,
        batch=16,       # ถ้า GPU แรง (VRAM > 6GB) ใช้ 16 หรือ 32 ได้เลยครับ เร็วปรู๊ด
        device='cpu',       # <--- เลข 0 คือรหัสลับสั่งให้ใช้ NVIDIA GPU ถ้าใส่'cpu' คือบังคับใช้ CPU (จะช้ามาก)
        name='yolo11_lpr_gpu',
        workers=4       # ใช้ CPU ช่วยโหลดภาพเข้า GPU 4 หัว
    )

    # 3. แปลงไฟล์เพื่อ Raspberry Pi (NCNN)
    print("--- Exporting for Raspberry Pi ---")
    model.export(format='ncnn')
    
    print("!!! Done !!!")