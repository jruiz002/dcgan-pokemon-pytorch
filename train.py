import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from models import Generator, Discriminator, weights_init, Z_DIM
from utils import save_image_grid, plot_losses, save_training_history

# Semilla fija para reproducibilidad — garantiza que los pesos iniciales
# y el ruido fijo sean los mismos en cada ejecución.
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Hiperparámetros — fijados por el Task 1.2
BATCH_SIZE = 32
LR = 2e-4
BETA1 = 0.5
BETA2 = 0.999

# Técnica de estabilización: Label Smoothing
# En lugar de etiquetas duras 0/1, usamos 0/0.9 para las reales.
# Esto evita que el discriminador se vuelva extremadamente seguro
# demasiado rápido, dejándole feedback útil al generador.
REAL_LABEL_SMOOTH = 0.9
FAKE_LABEL = 0.0
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
        
        # Ruido de instancia decreciente: añadimos ruido gaussiano pequeño
        # a las imágenes que entran al discriminador. Decae con las épocas
        # para ser fuerte al inicio (cuando D aprende demasiado rápido) y
        # desaparecer al final para no degradar la calidad de la salida.
        instance_noise_std = max(0.0, 0.1 * (1.0 - epoch / EPOCHS))
        
        for i, data in enumerate(dataloader):
            # Obtener batch de imágenes reales
            real_imgs = data.to(DEVICE)
            b_size = real_imgs.size(0)
            
            # Etiquetas con Label Smoothing: reales=0.9, falsas=0.0
            # Técnica de Salimans et al. para estabilizar el entrenamiento.
            label_real = torch.full((b_size,), REAL_LABEL_SMOOTH, dtype=torch.float, device=DEVICE)
            label_fake = torch.full((b_size,), FAKE_LABEL, dtype=torch.float, device=DEVICE)
            
            # ==========================================
            # PASO DEL DISCRIMINADOR
            # ==========================================
            netD.zero_grad()
            
            # Añadir ruido de instancia a imágenes reales para regularizar el discriminador
            noisy_real = real_imgs + instance_noise_std * torch.randn_like(real_imgs)
            
            # a) Pérdida sobre datos reales
            output_real = netD(noisy_real)
            errD_real = criterion(output_real, label_real)
            
            # b) Pérdida sobre datos falsos
            noise = torch.randn(b_size, Z_DIM, 1, 1, device=DEVICE)
            fake_imgs = netG(noise)
            
            # Añadir ruido de instancia a imágenes falsas también
            noisy_fake = fake_imgs.detach() + instance_noise_std * torch.randn_like(fake_imgs.detach())
            
            # Usar .detach() para no propagar gradientes hacia el Generador
            output_fake = netD(noisy_fake)
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

        # Persistir el historial en cada época evita perder las métricas si el
        # proceso se interrumpe y permite realizar el análisis de la Task 2.2.
        save_training_history(
            G_losses,
            D_losses,
            seed=SEED,
            epochs_completed=epoch,
            real_label=REAL_LABEL_SMOOTH,
            fake_label=FAKE_LABEL,
            discriminator_loss_definition=(
                "BCE_real + BCE_fake; equals -V(D,G) only with hard labels "
                "and without instance noise"
            ),
        )
        
        print(f"[{epoch:2d}/{EPOCHS}] Loss_D: {avg_loss_D:.4f}  Loss_G: {avg_loss_G:.4f}  "
              f"D(real): {output_real.mean().item():.3f}  D(fake): {output_fake.mean().item():.3f}  "
              f"Ruido inst.: {instance_noise_std:.4f}")
        
        # Guardar grilla de imágenes (Task 1.2)
        with torch.no_grad():
            fake_eval = netG(fixed_noise).detach().cpu()
        
        is_final = (epoch == EPOCHS)
        save_image_grid(fake_eval, epoch, is_final=is_final)
    
    # Producir visualización de curvas de pérdida (Task 1.3)
    plot_losses(G_losses, D_losses)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    torch.save(
        {
            "epoch": EPOCHS,
            "generator": netG.state_dict(),
            "discriminator": netD.state_dict(),
            "optimizer_G": optimizerG.state_dict(),
            "optimizer_D": optimizerD.state_dict(),
            "loss_G": G_losses,
            "loss_D": D_losses,
        },
        "outputs/checkpoints/task1_final.pt",
    )
    print("Entrenamiento finalizado. Visualizaciones guardadas en 'outputs/'.")

if __name__ == '__main__':
    train()
