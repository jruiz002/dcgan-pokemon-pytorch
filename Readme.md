# DCGAN para Sprites de Pokémon 🎮

Implementación de una DCGAN desde cero en PyTorch para generar sprites de Pokémon. Proyecto del curso de Deep Learning — UVG.

---

## Estructura del Proyecto

```
dcgan-pokemon-pytorch/
│
├── models.py        # Generador y Discriminador (Task 1.1)
├── dataset.py       # Carga y descarga del dataset desde PokeAPI
├── train.py         # Ciclo de entrenamiento alternado (Task 1.2)
├── utils.py         # Visualizaciones y desnormalización (Task 1.3)
├── Readme.md
├── requirements.txt
└── outputs/
    ├── images/      # Grilla por época + grilla final
    └── plots/       # Curva de pérdidas
```

---

## Cómo ejecutarlo

### 1. Instalar dependencias
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Validar arquitectura (Task 1.1)
```bash
python models.py
```
Resultado esperado:
```
Forma del generador correcta: torch.Size([4, 3, 64, 64])
Forma del discriminador correcta: torch.Size([4])
Todos los tests de formas pasaron con éxito.
```

### 3. Entrenar (Task 1.2 + 1.3)
```bash
python train.py
```
Descarga el dataset automáticamente si no existe, entrena 50 épocas y guarda las visualizaciones en `outputs/`.

---

## Task 1.1 — Arquitectura

Parámetros fijos del enunciado:

| Parámetro    | Valor |
|--------------|-------|
| Z_DIM        | 100   |
| IMG_SIZE     | 64    |
| IMG_CHANNELS | 3     |
| FEATURES_G   | 64    |
| FEATURES_D   | 64    |

### Generador
5 capas `ConvTranspose2d` que hacen upsampling progresivo:
```
z (100×1×1) → 512×4×4 → 256×8×8 → 128×16×16 → 64×32×32 → 3×64×64
```
Cada capa intermedia: `BatchNorm2d` + `ReLU`. Última capa: `Tanh` → salida en `[-1, 1]`.

### Discriminador
5 capas `Conv2d` con `stride=2` haciendo downsampling:
```
(3×64×64) → 64×32×32 → 128×16×16 → 256×8×8 → 512×4×4 → escalar
```
Capas intermedias (menos la primera): `BatchNorm2d` + `LeakyReLU(0.2)`. Última: `Sigmoid`.

---

## Task 1.2 — Entrenamiento Alternado

**Hiperparámetros (fijados por el task):**
- Loss: `BCELoss`
- Optimizer: `Adam(lr=2e-4, betas=(0.5, 0.999))` para ambas redes
- Batch size: `32`
- Épocas: `50`

**Ciclo por batch:**

1. **Paso del Discriminador:** pérdida en reales (etiqueta `0.9`) + pérdida en falsas (etiqueta `0`) usando `.detach()` para no propagar gradientes al generador → actualizar θ_D.
2. **Paso del Generador:** generar nuevo batch → pasar por D → calcular pérdida con etiqueta `1` (Truco de Goodfellow) → actualizar θ_G.

**Técnicas de estabilización aplicadas:**
- **Label Smoothing:** etiqueta real = `0.9` (Salimans et al.) para evitar que D se vuelva demasiado confiado
- **Instance Noise decreciente:** ruido gaussiano en las entradas de D, empieza en `0.098` y llega a `0` en la época 50
- **Compositing sobre fondo blanco:** los sprites de Pokémon son RGBA; se compuestaron sobre fondo blanco para evitar que D aprenda el patrón artificial del fondo negro

**Resultados reales del entrenamiento (con `seed=42`, dispositivo: MPS):**

| Época | Loss_D | Loss_G |
|-------|--------|--------|
| 1     | 0.6965 | 6.0341 |
| 10    | 0.3355 | 0.3405 |
| 24    | 0.9224 | 22.293 ← spike |
| 27    | 0.6711 | 3.0856 |
| 50    | 0.8757 | 2.0562 |

---

## Task 1.3 — Visualizaciones

### Grilla de imágenes generadas

El mismo vector de ruido fijo (`fixed_noise` de 16 vectores) se evalúa al final de cada época para ver la evolución del generador:

- **Época 1:** Ruido puro — el modelo no aprendió nada aún
- **Época ~15:** Blobs blancos centrales — el generador aprendió la estructura espacial básica
- **Época ~40:** Siluetas con colores cálidos y fríos que imitan la distribución de sprites
- **Época 50:** Formas naranjas y azules estructuradas con diversidad entre las 16 imágenes

La grilla final se guarda desnormalizada al rango `[0, 255]` aplicando la transformación inversa de `Tanh`: `pixel = (tensor + 1) / 2 * 255`.

### Curva de pérdidas

La gráfica muestra **tres fases claras** del entrenamiento:

**Fase 1 — Épocas 5 a 22** (`Loss_D ≈ Loss_G ≈ 0.33`):  
Ambas redes alcanzaron el piso matemático del label smoothing. Con etiqueta real = 0.9, el BCE mínimo alcanzable es ≈ 0.33. La anotación marca la **Época 15** como el punto de menor diferencia absoluta entre ambas curvas, cumpliendo el requerimiento del task.

**Fase 2 — Época 24** (`Loss_G` sube a **22**):  
El discriminador rompió el equilibrio anterior. Al recibir señales más limpias (el ruido de instancia se fue reduciendo), D aprendió a detectar los fakes con alta confianza, dejando al generador sin señal útil. Este es el juego adversarial funcionando exactamente como predice la teoría.

**Fase 3 — Épocas 27 a 50** (equilibrio dinámico):  
`Loss_D` oscila entre `0.6 – 0.9` y `Loss_G` entre `2.0 – 3.2`. Ninguna red domina a la otra de forma absoluta. Esta tensión sostenida es la que produce las imágenes más estructuradas y diversas del entrenamiento.

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
