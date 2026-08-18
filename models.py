import torch
import torch.nn as nn

# Hiperparámetros fijos según Radford et al. (2015)
Z_DIM = 100
IMG_SIZE = 64
IMG_CHANNELS = 3
FEATURES_G = 64
FEATURES_D = 64

class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, features_g=FEATURES_G, img_channels=IMG_CHANNELS):
        super(Generator, self).__init__()
        # Entrada: (batch, Z_DIM, 1, 1)
        self.gen = nn.Sequential(
            # Capa 1: Entrada z_dim -> features_g * 8
            nn.ConvTranspose2d(z_dim, features_g * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            # Salida: (features_g * 8) x 4 x 4
            
            # Capa 2: (features_g * 8) -> features_g * 4
            nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            # Salida: (features_g * 4) x 8 x 8
            
            # Capa 3: (features_g * 4) -> features_g * 2
            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            # Salida: (features_g * 2) x 16 x 16
            
            # Capa 4: (features_g * 2) -> features_g
            nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            # Salida: (features_g) x 32 x 32
            
            # Capa 5: features_g -> img_channels
            nn.ConvTranspose2d(features_g, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Salida: img_channels x 64 x 64
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS, features_d=FEATURES_D):
        super(Discriminator, self).__init__()
        # Entrada: (batch, img_channels, 64, 64)
        self.disc = nn.Sequential(
            # Capa 1: img_channels -> features_d
            # Según las instrucciones: BatchNorm después de cada capa EXCEPTO la primera y la última.
            nn.Conv2d(img_channels, features_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Salida: features_d x 32 x 32
            
            # Capa 2: features_d -> features_d * 2
            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Salida: (features_d * 2) x 16 x 16
            
            # Capa 3: features_d * 2 -> features_d * 4
            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Salida: (features_d * 4) x 8 x 8
            
            # Capa 4: features_d * 4 -> features_d * 8
            nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Salida: (features_d * 8) x 4 x 4
            
            # Capa 5: features_d * 8 -> 1
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0, bias=False),
            nn.Sigmoid()
            # Salida: 1 x 1 x 1
        )

    def forward(self, x):
        # Aplanar la salida (batch, 1, 1, 1) a (batch) para coincidir con shape = (4,) en el assert
        return self.disc(x).view(-1)

# Inicialización de pesos recomendada por DCGAN
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    # Test requerido en las instrucciones:
    z = torch.randn(4, Z_DIM, 1, 1)
    G = Generator()
    
    assert G(z).shape == (4, 3, 64, 64), "Forma del generador incorrecta"
    print("Forma del generador correcta:", G(z).shape)
    
    D = Discriminator()
    assert D(torch.randn(4, 3, 64, 64)).shape == (4,), "Forma del discriminador incorrecta"
    print("Forma del discriminador correcta:", D(torch.randn(4, 3, 64, 64)).shape)
    print("Todos los tests de formas pasaron con éxito.")
