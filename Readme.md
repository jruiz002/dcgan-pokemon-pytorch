# DCGAN para Sprites de Pokémon 🎮

Proyecto para el curso de Deep Learning en UVG — Task 1: implementación de una DCGAN desde cero en PyTorch para generar sprites de Pokémon a partir de ruido gaussiano.

## ¿Qué hace esto?

Se entrena un generador adversarial que aprende a convertir vectores de ruido aleatorio en imágenes de 64×64 píxeles que se parecen a los sprites de las primeras generaciones de Pokémon. El dataset viene de los sprites públicos de PokeAPI (898 imágenes en total).

---

## Estructura del Proyecto

```
dcgan-pokemon-pytorch/
│
├── models.py      # Generador y Discriminador (arquitectura DCGAN)
├── dataset.py     # Carga y descarga automática del dataset
├── train.py       # Ciclo de entrenamiento (Task 1.2)
├── utils.py       # Visualizaciones y desnormalización (Task 1.3)
├── docs.md        # Documentación técnica detallada
├── requirements.txt
└── outputs/
    ├── images/    # Grillas por época + grilla final
    └── plots/     # Curva de pérdidas
```

---

## Cómo correrlo

### 1. Crear entorno virtual e instalar dependencias
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Validar la arquitectura (Task 1.1)
```bash
python models.py
```
Deberías ver esto en la consola:
```
Forma del generador correcta: torch.Size([4, 3, 64, 64])
Forma del discriminador correcta: torch.Size([4])
Todos los tests de formas pasaron con éxito.
```

### 3. Entrenar (Task 1.2 y 1.3)
```bash
python train.py
```
El script descarga el dataset automáticamente si no existe, entrena por 50 épocas y guarda las visualizaciones en `outputs/`.

---

## Arquitectura (Task 1.1)

La red sigue las reglas de **Radford et al. (2015)** con los siguientes parámetros fijos:

| Parámetro    | Valor |
|--------------|-------|
| Z_DIM        | 100   |
| IMG_SIZE     | 64    |
| IMG_CHANNELS | 3     |
| FEATURES_G   | 64    |
| FEATURES_D   | 64    |

### Generador
Cinco capas `ConvTranspose2d` que hacen upsampling progresivo de `(100,1,1)` hasta `(3,64,64)`:
```
z (100,1,1) → 512×4×4 → 256×8×8 → 128×16×16 → 64×32×32 → 3×64×64
```
Cada capa intermedia usa `BatchNorm2d` + `ReLU`. La última usa `Tanh` para mapear la salida a `[-1, 1]`.

### Discriminador
Cinco capas `Conv2d` con `stride=2` haciendo downsampling de `(3,64,64)` hasta un escalar:
```
(3,64,64) → 64×32×32 → 128×16×16 → 256×8×8 → 512×4×4 → escalar
```
Cada capa (excepto la primera y última) usa `BatchNorm2d` + `LeakyReLU(0.2)`. La última usa `Sigmoid`.

---

## Entrenamiento (Task 1.2)

El ciclo alternado sigue el **Truco de Goodfellow**:

**Paso del Discriminador:**
1. Calcular pérdida en imágenes reales con etiqueta `0.9` (label smoothing)
2. Generar imágenes falsas y calcular pérdida con etiqueta `0.0`
3. Usar `.detach()` en las fakes para cortar el grafo de gradientes del generador
4. Sumar ambas pérdidas y actualizar θ_D

**Paso del Generador:**
1. Generar un nuevo batch de imágenes falsas
2. Pasarlas por el discriminador y calcular pérdida con etiqueta `1` (el generador quiere engañar al D)
3. Actualizar θ_G

**Hiperparámetros (fijados por el Task):**
- Loss: `BCELoss`
- Optimizer: `Adam(lr=2e-4, betas=(0.5, 0.999))` para ambas redes
- Batch size: `32`
- Épocas: `50`

**Técnicas de estabilización aplicadas:**
- **Label Smoothing** (Salimans et al.): etiquetas reales = 0.9 en lugar de 1.0, evita que D se vuelva muy seguro muy rápido
- **Instance Noise decreciente**: ruido gaussiano pequeño en las entradas de D que decae a 0 conforme avanza el entrenamiento
- **Fondo blanco en sprites**: los sprites de Pokémon tienen fondo transparente (RGBA). Se compositaron sobre fondo blanco para evitar el patrón artificial de fondo negro que D aprendería trivialmente

---

## Resultados (Task 1.3)

Al terminar el entrenamiento se generan automáticamente:

1. **`outputs/images/final_grid_plot.png`**: Grilla 4×4 con las imágenes finales desnormalizadas al rango `[0, 255]`
2. **`outputs/plots/loss_curves.png`**: Curvas de `Loss_G` y `Loss_D` a lo largo de las 50 épocas, con una anotación en el punto donde ambas pérdidas fueron más cercanas

---

## Dependencias

```
torch >= 2.0.0
torchvision >= 0.15.0
matplotlib >= 3.7.0
Pillow >= 9.5.0
requests >= 2.28.0
numpy >= 1.24.0
```
