import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import lightning as L
import math


# ------------------------------
# 📐 Función de pérdida angular
# ------------------------------
def angular_loss(pred, target):
    """
    Calcula la pérdida angular mínima entre dos ángulos en radianes
    Ambos deben estar en el rango [0, 2π)
    """
    diff = torch.remainder(pred - target + math.pi, 2 * math.pi) - math.pi
    return torch.mean(diff**2)


def cosine_angular_loss(pred, target):
    """
    Pérdida basada en el coseno de la diferencia angular
    """
    return 1 - torch.mean(torch.cos(pred - target))


def vector_angle_loss(pred, target):
    """
    MSE entre los vectores [sin(θ), cos(θ)] de predicho y real
    """
    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    pred_vec = torch.stack([torch.sin(pred), torch.cos(pred)], dim=1)
    target_vec = torch.stack([torch.sin(target), torch.cos(target)], dim=1)
    return F.mse_loss(pred_vec, target_vec)


# ------------------------------
# 🔧 Red con MobileNetV2
# ------------------------------
class MobileNetAngleRegression(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.lr = lr

        mobilenet = models.mobilenet_v2(pretrained=True)
        for param in mobilenet.features.parameters():
            param.requires_grad = False  # Congelamos extractor

        self.feature_extractor = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),  # salida final para el ángulo
        )
        self.loss = vector_angle_loss
        self.model = mobilenet  # ✅ clave: lo guardamos como atributo

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x  # convertimos a radianes

    def training_step(self, batch, batch_idx):
        images, target_angles = batch
        preds = self(images)
        loss = self.loss(preds, target_angles)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target_angles = batch
        preds = self(images)
        loss = self.loss(preds, target_angles)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, target_angles = batch
        preds = self(images)
        loss = self.loss(preds, target_angles)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-8
        )