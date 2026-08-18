# DCGAN para Sprites de Pokémon

Implementación de una DCGAN desde cero en PyTorch para generar sprites de Pokémon. Proyecto del curso de Deep Learning — UVG.

---

## Estructura del Proyecto

```
dcgan-pokemon-pytorch/
│
├── models.py        # Generador y Discriminador (Task 1.1)
├── dataset.py       # Carga y descarga del dataset desde PokeAPI
├── train.py         # Ciclo de entrenamiento alternado (Task 1.2)
├── task2.py         # Modo colapso 5D:1G y estimación de JSD
├── utils.py         # Visualizaciones y desnormalización (Task 1.3)
├── Readme.md
├── requirements.txt
└── outputs/
    ├── images/      # Grilla por época + grilla final
    ├── plots/       # Curva de pérdidas
    ├── training_history.json
    ├── checkpoints/
    └── task2/       # Grillas, métricas, gráfica JSD y checkpoint de Task 2
```

---

## Cómo ejecutarlo

### 1. Instalar dependencias
MacOS
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows
```Powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
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

El entrenamiento también guarda `outputs/training_history.json` y
`outputs/checkpoints/task1_final.pt`; el historial numérico es la entrada del
análisis de la Task 2.2.

### 4. Ejecutar Task 2

Después de completar las 50 épocas de la Task 1:

```bash
# Task 2.1: 20 épocas, 5 actualizaciones de D por cada actualización de G
python task2.py collapse

# Task 2.2: calcular y graficar JSD desde el historial de Task 1
python task2.py jsd

# Ejecutar ambas partes en secuencia
python task2.py all
```
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

## Task 2.1 — Inducción deliberada de modo colapso

El comando `python task2.py collapse` parte de pesos nuevos y entrena durante
20 épocas. Por cada batch repite cinco actualizaciones completas de `D`, con
nuevas muestras falsas, y después hace una actualización de `G`. Para no
amortiguar el desequilibrio deliberado usa etiquetas duras (`1` para reales y
`0` para falsas), sin label smoothing ni ruido de instancia.

La evidencia visual se guarda en
`outputs/task2/images/mode_collapse_grid_4x4.png`. Además, cada grilla registra
la distancia L1 media entre los 120 pares de las 16 imágenes. Una distancia
pequeña o decreciente, junto con formas y colores repetidos en la grilla, es
evidencia cuantitativa complementaria de baja diversidad; no reemplaza la
inspección visual.

### a) ¿Por qué el desequilibrio puede inducir modo colapso?

Sea `a(x)` el logit de `D`, de modo que `D(x) = σ(a(x))`. Para el objetivo
minimax saturante original,

```text
L_G^MM = E_z[log(1 - D(G(z)))]
```

su gradiente es

```text
∇θG L_G^MM = E_z[-D(G(z)) · ∇x a(x)|x=G(z) · JθG G(z)].
```

Si las cinco actualizaciones permiten que `D(G(z)) ≈ 0`, el factor
`D(G(z))` lleva el gradiente hacia cero. `G` deja de recibir una dirección útil
para mover masa hacia `p_data`; si por azar una salida logra engañar un poco a
`D`, muchos valores de `z` son empujados hacia esa misma salida. El modo se
refuerza y, con gradientes casi nulos para las demás salidas, el generador no
puede recuperar diversidad.

<!-- madagascar -->

Hay una precisión importante para esta implementación: la Task 1.2 exige el
truco no saturante,

```text
L_G^NS = -E_z[log D(G(z))],
∂L_G^NS/∂a = -(1 - D(G(z))).
```

Por tanto, cuando `D(G(z)) → 0`, este último factor tiende a `-1`, no a `0`.
El colapso 5D:1G no está matemáticamente garantizado solo por la saturación del
sigmoide. Aun así, un `D` sobreentrenado puede crear alrededor de las muestras
falsas un campo espacial con `∇x a(x)` muy pequeño o que apunta a unos pocos
modos. En ese caso, el producto completo sí es pobre y `G` recibe señales muy
similares para distintos `z`, lo que favorece la pérdida de diversidad. Esta
distinción evita atribuir a la pérdida no saturante el fallo exacto que esta fue
diseñada para corregir.

### b) Valor del discriminador óptimo en el modo repetido

El discriminador óptimo para un generador fijo es

```text
D*(x) = p_data(x) / (p_data(x) + p_G(x)).
```

En una imagen que el generador repite, `p_G(x)` concentra mucha más masa local
que `p_data(x)`. En consecuencia, `D*(x) ≈ 0`; si esa imagen está fuera del
soporte de los datos (`p_data(x) = 0`), el valor es exactamente `0`. No sería
`1/2`: ese valor solo aparece donde ambas densidades coinciden.

### c) Modificación concreta para prevenirlo

Se puede añadir ruido de instancia gaussiano a las entradas reales y falsas de
`D`, con una desviación que decaiga gradualmente hasta cero. Esto reemplaza
temporalmente las distribuciones por sus convoluciones con una gaussiana, cuyos
soportes se solapan. Así `D` no puede separar perfectamente real y falso al
inicio, se evita que `D(G(z))` se fije inmediatamente en cero y se mantiene un
campo `∇x a(x)` informativo para que distintas muestras de `z` puedan desplazarse
hacia diferentes regiones de los datos. Es una modificación del objetivo que
ve el discriminador, no un cambio de la proporción de pasos.

---

## Task 2.2 — Estimación empírica de Jensen-Shannon

En el objetivo minimax,

```text
V(D,G) = E_data[log D(x)] + E_z[log(1-D(G(z)))].
```

La pérdida registrada del discriminador es la suma de las dos BCE, por lo que
`loss_D = -V(D,G)`. La estimación implementada en `utils.estimate_jsd` es

```text
JSD_hat = (V(D,G) + log(4)) / 2
        = (log(4) - loss_D) / 2.
```

`loss_G` se conserva junto con `loss_D` en el JSON para comparar la dinámica de
ambas redes, pero no se sustituye en esta ecuación: la pérdida no saturante de
`G` no es uno de los dos términos que definen `V(D,G)`. La curva completa queda
en `outputs/task2/plots/jsd_evolution.png` y los valores por época en
`outputs/task2/jsd_estimates.json`.

### a) Dirección del sesgo

Para un `G` fijo, `D*` maximiza `V`; por definición,

```text
V(D,G) ≤ V(D*,G).
```

Como la transformación `(V + log(4))/2` es creciente, usar un discriminador no
óptimo produce, en las condiciones de la derivación, una **subestimación** de
la JSD verdadera. Incluso puede dar un valor negativo, aunque la JSD real no lo
sea. Por eso el código no recorta la estimación a cero: un valor negativo hace
visible que el supuesto `D = D*` falló.

Existe otra salvedad en esta corrida de Task 1: se usaron label smoothing y
ruido de instancia durante el entrenamiento. En esas épocas `loss_D` no es
exactamente `-V` para las distribuciones originales, así que, además del sesgo
por `D` subóptimo, hay un error de modelo cuya dirección no está garantizada.

### b) Límite teórico y verificación

Si la GAN converge bien, `p_G = p_data`, `D*(x) = 1/2` y
`V(D*,G) = -log(4)`. Por tanto, la curva debería tender a
`JSD(p_data || p_G) = 0`.

Con el valor final documentado de la corrida (`loss_D = 0.8757`), el cálculo
directo da

```text
JSD_hat_50 = (log(4) - 0.8757) / 2 ≈ 0.2553 nats.
```

Ese resultado no es cercano a cero, de modo que la corrida empírica no es
consistente con convergencia completa en la época 50. Debe interpretarse junto
con las salvedades anteriores: indica que el juego seguía en desequilibrio, no
una medición exacta de la JSD verdadera.

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
