import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import transforms
import os
from PIL import Image
import random
from collections import defaultdict
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from Modules import *
from sklearn.model_selection import train_test_split

print(torch.cuda.is_available())  # ¿True?
print(torch.version.cuda)  # ¿"11.8", "12.1", etc?|
print(torch.cuda.get_device_name(0))  # ¿Detecta tu GPU?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_angle = pd.read_csv("/root/jastudillo/Trabajo/Cedulas/progreso_orientacion.csv")
ruta_base = "/root/jastudillo/Trabajo/Cedulas/imagenes_cedula_peq"
archivos_procesados = set(df_angle["File"].tolist())

# Asegura que la columna 'Matrix' exista
df_angle["Matrix"] = None

for carpeta_actual, subcarpetas, archivos in os.walk(ruta_base):
    for archivo in archivos:
        if archivo.lower().endswith((".jpg", ".png")):
            if archivo in archivos_procesados:
                ruta_img = os.path.join(carpeta_actual, archivo)
                try:
                    img = Image.open(ruta_img).convert("RGB")
                    # Buscar el índice correspondiente
                    idx = df_angle[df_angle["File"] == archivo].index
                    if not idx.empty:
                        df_angle.at[idx[0], "Matrix"] = img
                except Exception as e:
                    print(f"Error al procesar {archivo}: {e}")


rotation_counts = defaultdict(int)  # Guarda cuántas veces se ha rotado cada archivo


def generar_angulo_biased():
    # Opciones centradas en ±90 y ±180 con algo de ruido
    centros = [-180, -90, 90, 180]
    centro = random.choice(centros)
    ruido = np.random.normal(loc=0, scale=15)  # Ruido pequeño
    return centro + ruido


def aplicar_aumento(row):
    ran = generar_angulo_biased()
    row["Angle"] += ran
    row["Matrix"] = row["Matrix"].rotate(ran, expand=True)
    row["File"] = row["File"].replace(".jpg", f"_aumentado_{ran:.1f}.jpg")
    return row


def aumentar_df(df_base, frac, repeticiones, etiquetas):
    global df_angle

    for i in range(repeticiones):
        # Elegir un intervalo aleatoriamente
        df_aument = (
            df_base[df_base["Label"].isin(etiquetas)]
            .sample(frac=frac, random_state=i)
            .copy()
        )

        # Filtra imágenes que aún no han sido rotadas más de 2 veces
        df_aument = df_aument[df_aument["File"].apply(lambda f: rotation_counts[f] < 2)]

        if df_aument.empty:
            continue  # Saltar si no hay nada para aumentar

        # Rotación y actualización
        df_aument = df_aument.apply(aplicar_aumento, axis=1)

        # Actualizar contador
        for f in df_aument["File"]:
            rotation_counts[f] += 1

        # Concatenar
        df_angle = pd.concat([df_angle, df_aument], ignore_index=True)


print("Procesando Datos")
# Aplicar a diferentes clases
aumentar_df(df_angle, frac=0.5, repeticiones=5 * 16, etiquetas=["Cedula_Amarilla"])
aumentar_df(df_angle, frac=0.7, repeticiones=5 * 16, etiquetas=["Cedula_Celeste"])
aumentar_df(df_angle, frac=0.8, repeticiones=7 * 16, etiquetas=["Cedula_Guayaquil"])
aumentar_df(df_angle, frac=1.0, repeticiones=9 * 16, etiquetas=["Licencias"])
aumentar_df(df_angle, frac=1.0, repeticiones=8 * 16, etiquetas=["Papeleta"])
aumentar_df(df_angle, frac=1.0, repeticiones=11 * 16, etiquetas=["Otros"])


# Transformador para convertir imágenes PIL a tensores normalizados
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # convierte PIL o np.array a [C, H, W] y escala a [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Imágenes dummy y ángulos (en radianes)

angles = torch.tensor(df_angle["Angle"].values * np.pi / 180, dtype=torch.float32)

images = torch.stack(df_angle["Matrix"].apply(lambda img: transform(img)).tolist())


print("Cargando Datos")
# Dataset y dataloaders
batch = 128
dataset = TensorDataset(images, angles)
train, test_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
train_dataset, val_dataset = train_test_split(train, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=batch)
val_loader = DataLoader(val_dataset, batch_size=batch)
test_loader = DataLoader(test_dataset, batch_size=batch)


logger = CSVLogger("logs", name="mobilenet_angle")
# Entrenamiento

early_stop_callback = L.pytorch.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=20,
)
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    save_top_k=2,
    save_last=True,
    monitor="val_loss",
    mode="min",
)
print("Empezar entrenamiento")
model = MobileNetAngleRegression(lr=5e-5)
trainer = L.Trainer(
    max_epochs=500,
    callbacks=[early_stop_callback, checkpoint_callback],
    logger=logger,
    accelerator="gpu" if device.type == "cuda" else "cpu",
    devices=1,
)
trainer.fit(model, train_loader, val_loader)
result_val = trainer.validate(
    model,
    dataloaders=val_loader,
    ckpt_path="best",
)
result_val = trainer.test(
    model,
    dataloaders=val_loader,
    ckpt_path="best",
)

best_model_path = checkpoint_callback.best_model_path
print("✅ Mejor modelo guardado en:", best_model_path)

# Cargar el modelo Lightning desde checkpoint
best_model = MobileNetAngleRegression.load_from_checkpoint(best_model_path)
best_model.eval()  # Cambiar a modo evaluación


# Guardar todo el modelo completo (arquitectura + pesos)
torch.save(best_model, "mobilenet_angle_full_model.pt")
print("Modelo completo guardado en mobilenet_angle_full_model.pt")