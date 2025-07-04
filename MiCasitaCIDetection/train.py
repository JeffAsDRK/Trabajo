import torch
from ultralytics import YOLO

# Cargar un modelo base (puedes cambiar a yolov8s.pt, yolov8m.pt, etc.)
model = YOLO(
    "/root/jastudillo/Trabajo/MiCasitaCIDetection/yolo11n-obb.pt"
)  # o "yolov8s.pt", o "yolov8n.yaml" para comenzar desde cero

# Entrenar el modelo
if __name__ == "__main__":
    print(torch.cuda.is_available())  # ¿True?
    print(torch.version.cuda)  # ¿"11.8", "12.1", etc?|
    print(torch.cuda.get_device_name(0))  # ¿Detecta tu GPU?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train(
        data="/root/jastudillo/Trabajo/Training_Cedulas/data.yaml",  # Ruta al archivo de configuración
        epochs=1000,  # Número de épocas
        imgsz=640,  # Tamaño de entrada de imagen
        batch=512,  # Tamaño del batch
        workers=32,  # Núm. de workers para carga de datos
        device=[-1,-1],  # Usa la GPU 0; usa "cpu" para CPU
        name="yolo11_obb_custom",  # Nombre de la carpeta del experimento
        verbose=True,
        patience=500,
        lr0=1e-4,
        lrf=0.001,
        degrees=45,
        optimizer="AdamW",
        weight_decay=1e-6,
        #hsv_v=0.2,
        hsv_s=0.5,
        translate=0.2,
        scale=0.3,
        #shear=2,
        flipud=0.5,
        #copy_paste_mode="mixup",
    )
