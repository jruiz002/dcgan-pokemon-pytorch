import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from models import Generator, Discriminator, weights_init, Z_DIM
from utils import save_image_grid, plot_losses

# Hiperparámetros
BATCH_SIZE = 32
LR = 2e-4
BETA1 = 0.5
BETA2 = 0.999
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train():
    print(f"Usando dispositivo: {DEVICE}")
    
    # 1. Preparar DataLoader
    dataloader = get_dataloader(batch_size=BATCH_SIZE)
    
    # 2. Instanciar modelos y aplicar inicialización de pesos
    netG = Generator().to(DEVICE)
    netD = Discriminator().to(DEVICE)
    
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # 3. Función de pérdida y optimizadores
    criterion = nn.BCELoss()
    
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))
    
    # Ruido fijo para visualizar la evolución del generador (grilla de 16 = 4x4)
    fixed_noise = torch.randn(16, Z_DIM, 1, 1, device=DEVICE)
    
    # Listas para guardar las pérdidas
    G_losses = []
    D_losses = []
    
    print("Iniciando Entrenamiento...")
    for epoch in range(1, EPOCHS + 1):
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        
        for i, data in enumerate(dataloader):
            # Obtener batch de imágenes reales
            real_imgs = data.to(DEVICE)
            b_size = real_imgs.size(0)
            
            # Etiquetas reales (1) y falsas (0)
            label_real = torch.ones((b_size,), dtype=torch.float, device=DEVICE)
            label_fake = torch.zeros((b_size,), dtype=torch.float, device=DEVICE)
            
            # ==========================================
            # PASO DEL DISCRIMINADOR
            # ==========================================
            netD.zero_grad()
            
            # a) Pérdida sobre datos reales
            output_real = netD(real_imgs)
            errD_real = criterion(output_real, label_real)
            
            # b) Pérdida sobre datos falsos
            noise = torch.randn(b_size, Z_DIM, 1, 1, device=DEVICE)
            fake_imgs = netG(noise)
            # Usar .detach() para no propagar gradientes hacia el Generador
            output_fake = netD(fake_imgs.detach())
            errD_fake = criterion(output_fake, label_fake)
            
            # c) Sumar ambas pérdidas y actualizar D
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()
            
            # ==========================================
            # PASO DEL GENERADOR
            # ==========================================
            netG.zero_grad()
            
            # Generar un nuevo batch de imágenes falsas (según requerimiento Task 1.2)
            noise2 = torch.randn(b_size, Z_DIM, 1, 1, device=DEVICE)
            fake_imgs2 = netG(noise2)
            
            # Pasar fakes por el discriminador
            output_fake2 = netD(fake_imgs2)
            
            # Calcular pérdida usando etiquetas 1 (Truco de Goodfellow)
            errG = criterion(output_fake2, label_real)
            
            # Actualizar G
            errG.backward()
            optimizerG.step()
            
            # Acumular pérdidas
            epoch_loss_D += errD.item()
            epoch_loss_G += errG.item()
            
        # Promediar pérdidas de la época
        avg_loss_D = epoch_loss_D / len(dataloader)
        avg_loss_G = epoch_loss_G / len(dataloader)
        
        D_losses.append(avg_loss_D)
        G_losses.append(avg_loss_G)
        
        print(f"[{epoch}/{EPOCHS}] Loss_D: {avg_loss_D:.4f} Loss_G: {avg_loss_G:.4f}")
        
        # Guardar grilla de imágenes (Task 1.2)
        with torch.no_grad():
            fake_eval = netG(fixed_noise).detach().cpu()
        
        is_final = (epoch == EPOCHS)
        save_image_grid(fake_eval, epoch, is_final=is_final)
    
    # Producir visualización de curvas de pérdida (Task 1.3)
    plot_losses(G_losses, D_losses)
    print("Entrenamiento finalizado. Visualizaciones guardadas en 'outputs/'.")

if __name__ == '__main__':
    train()
