import os
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class PokemonDataset(Dataset):
    def __init__(self, data_dir='data/pokemon', img_size=64, download=True):
        self.data_dir = data_dir
        self.img_size = img_size
        
        if download and not os.path.exists(self.data_dir):
            self.download_dataset()
            
        # Lista de archivos en el directorio
        self.image_files = [f for f in os.listdir(self.data_dir) if f.endswith('.png')]
        
        # Transformaciones: Redimensionar, Convertir a Tensor y Normalizar a [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def download_dataset(self):
        print(f"Descargando dataset en {self.data_dir}...")
        os.makedirs(self.data_dir, exist_ok=True)
        # El enunciado menciona 898 imágenes de PokeAPI
        for i in range(1, 899):
            url = f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{i}.png"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    # Guardamos el PNG original con su canal alfa intacto
                    img_raw = Image.open(BytesIO(response.content))
                    img_raw.save(os.path.join(self.data_dir, f"{i}.png"))
                if i % 100 == 0:
                    print(f"Descargadas {i}/898 imágenes...")
            except Exception as e:
                print(f"Error descargando la imagen {i}: {e}")
        print("Descarga finalizada.")

    def _composite_on_white(self, img: Image.Image) -> Image.Image:
        """
        Pega el sprite sobre un fondo blanco para manejar la transparencia.
        Los sprites de Pokémon tienen fondo transparente (RGBA). Si hacemos
        .convert('RGB') directamente, el fondo transparente se convierte en
        negro puro, lo que crea un patrón artificial que el discriminador
        aprende trivialmente. Al usar fondo blanco obtenemos imágenes más
        ricas y diversas para el entrenamiento.
        """
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Canal alfa como máscara
            return background
        return img.convert('RGB')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        img_raw = Image.open(img_name)
        # Fix clave: compositar sobre fondo blanco en lugar de convertir directamente
        image = self._composite_on_white(img_raw)
        
        if self.transform:
            image = self.transform(image)
            
        return image

def get_dataloader(batch_size=32, data_dir='data/pokemon', img_size=64):
    dataset = PokemonDataset(data_dir=data_dir, img_size=img_size, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return dataloader

if __name__ == "__main__":
    dataloader = get_dataloader(batch_size=32)
    for imgs in dataloader:
        print("Forma del batch:", imgs.shape)
        print("Min/Max del tensor:", imgs.min().item(), imgs.max().item())
        break
