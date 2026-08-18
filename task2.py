"""Experimentos y análisis requeridos por la Task 2.

Uso:
    python task2.py collapse
    python task2.py jsd
    python task2.py all
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloader
from models import Discriminator, Generator, Z_DIM, weights_init
from utils import denormalize, plot_jsd, save_training_history


SEED = 42
BATCH_SIZE = 32
LR = 2e-4
BETAS = (0.5, 0.999)
EPOCHS = 20
D_STEPS_PER_G_STEP = 5
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean_pairwise_l1(images):
    """Distancia L1 media entre pares; valores pequeños señalan poca diversidad."""
    flat = denormalize(images).flatten(start_dim=1)
    distances = torch.pdist(flat, p=1) / flat.size(1)
    return distances.mean().item() if distances.numel() else 0.0


def _save_collapse_grid(images, output_path, title):
    """Guarda una grilla 4x4 y anota una métrica simple de diversidad."""
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid = vutils.make_grid(denormalize(images).clamp(0, 1), nrow=4, padding=2)
    grid = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def train_mode_collapse(epochs=EPOCHS, max_batches=None, output_dir="outputs/task2"):
    """Entrena D cinco veces por cada actualización de G durante 20 épocas."""
    set_seed()
    print(f"Task 2.1 en {DEVICE}: {D_STEPS_PER_G_STEP} pasos de D por 1 de G")

    dataloader = get_dataloader(batch_size=BATCH_SIZE)
    netG = Generator().to(DEVICE)
    netD = Discriminator().to(DEVICE)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=BETAS)
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=BETAS)
    fixed_noise = torch.randn(16, Z_DIM, 1, 1, device=DEVICE)

    losses_G = []
    losses_D = []
    diversity = []
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        batches = 0

        for batch_idx, real_imgs in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            real_imgs = real_imgs.to(DEVICE)
            batch_size = real_imgs.size(0)
            real_targets = torch.ones(batch_size, device=DEVICE)
            fake_targets = torch.zeros(batch_size, device=DEVICE)
            discriminator_step_loss = 0.0

            # Cinco actualizaciones completas de D usando nuevos fakes. Repetir
            # el batch real deliberadamente favorece que D domine el juego.
            for _ in range(D_STEPS_PER_G_STEP):
                optimizerD.zero_grad(set_to_none=True)
                output_real = netD(real_imgs)
                loss_real = criterion(output_real, real_targets)

                noise = torch.randn(batch_size, Z_DIM, 1, 1, device=DEVICE)
                with torch.no_grad():
                    fake_imgs = netG(noise)
                output_fake = netD(fake_imgs)
                loss_fake = criterion(output_fake, fake_targets)

                loss_D = loss_real + loss_fake
                loss_D.backward()
                optimizerD.step()
                discriminator_step_loss += loss_D.item()

            # Una actualización de G con la pérdida no saturante de Task 1.2.
            # Se congelan los parámetros de D para no acumular gradientes inútiles.
            optimizerG.zero_grad(set_to_none=True)
            netD.requires_grad_(False)
            noise = torch.randn(batch_size, Z_DIM, 1, 1, device=DEVICE)
            fake_imgs = netG(noise)
            output_fake_for_G = netD(fake_imgs)
            loss_G = criterion(output_fake_for_G, real_targets)
            loss_G.backward()
            optimizerG.step()
            netD.requires_grad_(True)

            epoch_loss_D += discriminator_step_loss / D_STEPS_PER_G_STEP
            epoch_loss_G += loss_G.item()
            batches += 1

        if batches == 0:
            raise RuntimeError("No se procesó ningún batch; revise el dataset o --max-batches.")

        avg_loss_D = epoch_loss_D / batches
        avg_loss_G = epoch_loss_G / batches
        losses_D.append(avg_loss_D)
        losses_G.append(avg_loss_G)

        netG.eval()
        with torch.no_grad():
            samples = netG(fixed_noise).cpu()
        netG.train()
        epoch_diversity = _mean_pairwise_l1(samples)
        diversity.append(epoch_diversity)

        _save_collapse_grid(
            samples,
            os.path.join(output_dir, "images", f"epoch_{epoch:02d}.png"),
            f"Task 2.1 — época {epoch} | diversidad L1={epoch_diversity:.4f}",
        )
        save_training_history(
            losses_G,
            losses_D,
            os.path.join(output_dir, "collapse_history.json"),
            diversity_pairwise_l1=diversity,
            d_steps_per_g_step=D_STEPS_PER_G_STEP,
            epochs_completed=epoch,
            seed=SEED,
        )
        print(
            f"[{epoch:02d}/{epochs}] Loss_D={avg_loss_D:.4f} "
            f"Loss_G={avg_loss_G:.4f} diversidad_L1={epoch_diversity:.4f}"
        )

    final_grid = os.path.join(output_dir, "images", "mode_collapse_grid_4x4.png")
    _save_collapse_grid(
        samples,
        final_grid,
        f"Modo colapso 5D:1G — época {epochs} | diversidad L1={diversity[-1]:.4f}",
    )
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    torch.save(
        {
            "epoch": epochs,
            "generator": netG.state_dict(),
            "discriminator": netD.state_dict(),
            "optimizer_G": optimizerG.state_dict(),
            "optimizer_D": optimizerD.state_dict(),
            "loss_G": losses_G,
            "loss_D": losses_D,
            "diversity_pairwise_l1": diversity,
        },
        os.path.join(output_dir, "checkpoints", "mode_collapse_final.pt"),
    )
    print(f"Grilla final guardada en {final_grid}")
    return losses_G, losses_D, diversity


def analyze_jsd(history_path="outputs/training_history.json", output_dir="outputs/task2"):
    """Calcula y grafica JSD_hat usando el historial de la Task 1.2."""
    if not os.path.exists(history_path):
        raise FileNotFoundError(
            f"No existe {history_path}. Ejecute primero 'python train.py' para "
            "registrar las pérdidas de las 50 épocas de Task 1.2."
        )

    with open(history_path, "r", encoding="utf-8") as file:
        history = json.load(file)

    losses_G = history.get("loss_G", [])
    losses_D = history.get("loss_D", [])
    if not losses_G or not losses_D or len(losses_G) != len(losses_D):
        raise ValueError("El historial debe contener loss_G y loss_D con igual longitud.")

    jsd_values = plot_jsd(losses_D, os.path.join(output_dir, "plots"))
    result = {
        "formula": "JSD_hat = (log(4) - loss_D) / 2",
        "epochs": list(range(1, len(jsd_values) + 1)),
        "loss_G": [float(value) for value in losses_G],
        "loss_D": [float(value) for value in losses_D],
        "jsd_hat": [float(value) for value in jsd_values],
        "theoretical_limit_if_converged": 0.0,
        "warning": (
            "La identidad es exacta solo para D optimo y la BCE minimax sin "
            "label smoothing ni instance noise. No se recortan valores negativos."
        ),
    }
    analysis_path = os.path.join(output_dir, "jsd_estimates.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(analysis_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"JSD_hat inicial={jsd_values[0]:.4f}, final={jsd_values[-1]:.4f}")
    print(f"Gráfica guardada en {os.path.join(output_dir, 'plots', 'jsd_evolution.png')}")
    return jsd_values


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 de la DCGAN de Pokémon")
    parser.add_argument("action", choices=("collapse", "jsd", "all"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Solo para smoke tests; omitir en el experimento entregable.",
    )
    parser.add_argument("--history", default="outputs/training_history.json")
    parser.add_argument("--output-dir", default="outputs/task2")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.action in ("collapse", "all"):
        train_mode_collapse(args.epochs, args.max_batches, args.output_dir)
    if args.action in ("jsd", "all"):
        analyze_jsd(args.history, args.output_dir)


if __name__ == "__main__":
    main()
