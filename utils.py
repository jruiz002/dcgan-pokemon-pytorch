import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def denormalize(tensor):
    """
    Desnormaliza tensores del rango [-1, 1] al rango [0, 1] para visualización.
    """
    # DCGAN usa Tanh, por lo que las imágenes están en [-1, 1]
    # La transformación inversa es: (x + 1) / 2
    return (tensor + 1.0) / 2.0

def save_image_grid(tensor, epoch, output_dir="outputs/images", is_final=False):
    """
    Guarda una grilla de 4x4 imágenes.
    Si is_final es True, además devuelve la grilla desnormalizada en rango [0, 255] (Task 1.3).
    """
    os.makedirs(output_dir, exist_ok=True)
    # Desnormalizar a [0, 1]
    denorm_tensor = denormalize(tensor)
    
    # Crear grilla (4x4)
    grid = vutils.make_grid(denorm_tensor, nrow=4, padding=2, normalize=False)
    
    # Convertir a numpy para matplotlib o guardar directamente
    # vutils.save_image ya maneja tensores en [0, 1]
    if not is_final:
        vutils.save_image(denorm_tensor, os.path.join(output_dir, f"epoch_{epoch}.png"), nrow=4, padding=2)
    else:
        vutils.save_image(denorm_tensor, os.path.join(output_dir, "final_grid_4x4.png"), nrow=4, padding=2)
        
        # Para cumplir exactamente con "valores desnormalizados al rango [0, 255]" 
        # (Task 1.3 primera visualización)
        grid_255 = (grid.cpu().numpy() * 255).astype(np.uint8)
        grid_255 = np.transpose(grid_255, (1, 2, 0)) # CHW to HWC
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid_255)
        plt.axis("off")
        plt.title("Imágenes Finales (Rango [0, 255])")
        plt.savefig(os.path.join(output_dir, "final_grid_plot.png"))
        plt.close()

def plot_losses(losses_G, losses_D, output_dir="outputs/plots"):
    """
    Produce la segunda visualización (Task 1.3): curvas de loss_G y loss_D.
    Anota el punto donde las pérdidas son más cercanas.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = range(1, len(losses_G) + 1)
    
    # Encontrar el punto donde las pérdidas son más cercanas
    diffs = np.abs(np.array(losses_G) - np.array(losses_D))
    min_diff_idx = np.argmin(diffs)
    min_diff_epoch = epochs[min_diff_idx]
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses_G, label="Loss Generador (G)", color="blue")
    plt.plot(epochs, losses_D, label="Loss Discriminador (D)", color="red")
    
    # Anotar el punto
    plt.annotate(
        f'Menor dif. (Época {min_diff_epoch})', 
        xy=(min_diff_epoch, losses_G[min_diff_idx]), 
        xytext=(min_diff_epoch, losses_G[min_diff_idx] + 0.5),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
        horizontalalignment='center'
    )
    
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida (BCE Loss)")
    plt.title("Curvas de Entrenamiento de DCGAN")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, "loss_curves.png"))
    plt.close()
