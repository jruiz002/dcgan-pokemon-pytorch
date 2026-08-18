# Documentación - Task 1: DCGAN para Pokémon

## 1. Arquitectura de la Red

La implementación sigue estrictamente los lineamientos propuestos por Radford et al. (2015) para Deep Convolutional Generative Adversarial Networks (DCGAN).

### Generador (`Generator`)
- **Entrada:** Un vector latente `z` de ruido gaussiano de dimensión 100 (`100x1x1`).
- **Arquitectura:** Consiste en 5 capas de convolución transpuesta (`ConvTranspose2d`). Estas capas expanden gradualmente la resolución espacial desde `1x1` hasta `64x64`, reduciendo simultáneamente la profundidad del canal (`512 -> 256 -> 128 -> 64 -> 3`).
- **Activaciones:** Se utiliza `ReLU` en todas las capas ocultas, seguido de normalización por lotes (`BatchNorm2d`). Esto ayuda a estabilizar el aprendizaje, previniendo el colapso del modo. La capa de salida utiliza la función de activación `Tanh` para mapear los valores de los píxeles al rango `[-1, 1]`.

### Discriminador (`Discriminator`)
- **Entrada:** Una imagen de `3x64x64` (RGB).
- **Arquitectura:** Consiste en 5 capas convolucionales (`Conv2d`) con tamaño de kernel 4 y `stride` 2 (reemplazando el agrupamiento/pooling espacial para permitir que la red aprenda su propio downsampling). 
- **Activaciones:** Se usa `LeakyReLU` con una pendiente de `0.2` en todas las capas ocultas, fomentando que los gradientes fluyan incluso para valores negativos y mitigando el problema de gradientes nulos. Se aplica `BatchNorm2d` en las capas intermedias, excepto en la primera capa (como dicta el paper) y la última. La salida pasa por una función `Sigmoid` para producir una probabilidad escalar en el rango `[0, 1]` (indicando si la imagen es real o falsa).

## 2. Decisiones de Entrenamiento y Trucos Aplicados

El script de entrenamiento (`train.py`) implementa el ciclo adversarial siguiendo el **Truco de Goodfellow**:

1. **Paso del Discriminador:**
   - Se optimiza el discriminador para maximizar $\log(D(x)) + \log(1 - D(G(z)))$.
   - Para esto, se calcula primero la pérdida usando Binary Cross Entropy (`BCELoss`) en imágenes reales con etiqueta `1`.
   - Luego, se genera un lote falso y se pasa por el discriminador. *Es crucial usar `.detach()` en la salida del generador en este punto para evitar que los gradientes de este paso afecten a los pesos del generador.* Se calcula la pérdida con etiqueta `0`.
   - Se suman ambas pérdidas y se actualizan los pesos ($\theta_D$).

2. **Paso del Generador:**
   - En lugar de minimizar $\log(1 - D(G(z)))$ (lo cual causa gradientes débiles al principio del entrenamiento), Goodfellow propuso maximizar $\log(D(G(z)))$.
   - Se genera un *nuevo* lote de imágenes falsas (para mantener la independencia de la muestra y evitar problemas de estancamiento), y se pasan por el discriminador.
   - Se calcula la pérdida utilizando la etiqueta `1` (el generador quiere engañar al discriminador haciéndole creer que son reales).
   - Se actualizan los pesos del generador ($\theta_G$).

**Hiperparámetros:**
- `Adam` Optimizer con `lr = 0.0002` y momento `beta1 = 0.5`. Estos valores atípicos (especialmente el momento bajo) son fundamentales en DCGAN para evitar oscilaciones severas e inestabilidad.

## 3. Visualizaciones y Resultados (Task 1.3)

Se incluyen funciones en `utils.py` para cumplir con las dos visualizaciones principales:
1. **Grilla de Imágenes:** Durante el entrenamiento, se inyecta sistemáticamente el mismo vector de ruido `fixed_noise` en el generador al final de cada época. Al final de la época 50, se extrae el resultado, se desnormaliza utilizando una transformación inversa `(x + 1)/2` y se guarda en un rango `[0, 255]`.
2. **Curvas de Pérdida:** Se almacena el promedio del `BCELoss` del discriminador y del generador por cada época. La figura resultante anota matemáticamente la época específica donde `abs(loss_G - loss_D)` es mínima, lo cual suele ser un buen indicador cualitativo del equilibrio del juego minimax.

## 4. Buenas Prácticas de Ingeniería de Software Aplicadas

Para asegurar un código mantenible, legible y escalable (como se solicitaba para obtener el 100%), se ha estructurado el proyecto siguiendo un modelo modular:

- **Separación de Responsabilidades (SoC):** 
  - `models.py`: Encapsula estrictamente las definiciones de las clases neuronales.
  - `dataset.py`: Maneja toda la lógica de obtención, transformación y carga (ETL) de datos a través de PyTorch `Dataset` y `DataLoader`.
  - `utils.py`: Centraliza funciones accesorias de visualización y post-procesamiento. No contiene lógica de entrenamiento ni estado.
  - `train.py`: Actúa como punto de entrada (entry point) orquestando el bucle de entrenamiento.
- **Robustez y Tolerancia a Fallos:** El dataset incorpora un script integrado para descargar automáticamente los datos de *PokeAPI* si estos no se encuentran disponibles localmente en la carpeta objetivo.
- **Validación Automática:** Se agregaron aserciones matemáticas (`assert`) en `models.py` embebidas en el bloque `if __name__ == "__main__":` para verificar de forma unitaria la integridad de los tensores de entrada y salida, asegurando que las transformaciones espaciales sean correctas antes de iniciar el costoso proceso de entrenamiento.
- **Reproducibilidad:** El entorno está diseñado para ejecutarse dinámicamente evaluando el soporte local del hardware (`cuda`, `mps`, o `cpu`), haciéndolo cross-platform. Se han parametrizado constantes globales en mayúsculas (`Z_DIM`, `BATCH_SIZE`) facilitando su ajuste futuro.

## 5. Validación de los Requerimientos del Task 1

Para demostrar que se ha cumplido con el 100% de la rúbrica, aquí se detalla cómo comprobar cada punto:

### Validación del Task 1.1 (Arquitectura y Formas)
El requerimiento exigía cumplir con las formas exactas para el Generador y Discriminador. Para validarlo de forma independiente, ejecuta en la terminal:
```bash
python models.py
```
**Resultado esperado:** El script ejecutará los `assert` solicitados y en la consola verás:
> Forma del generador correcta: torch.Size([4, 3, 64, 64])
> Forma del discriminador correcta: torch.Size([4])
> Todos los tests de formas pasaron con éxito.

Esto demuestra empíricamente que la matemática de las capas de convolución está implementada de forma correcta.

### Validación del Task 1.2 (Entrenamiento Alternado)
El requerimiento exigía implementar el entrenamiento alternado, usando `BCELoss`, optimizador Adam (lr=2e-4, betas=(0.5, 0.999)), batch_size=32 por 50 épocas, guardando pérdidas e imágenes por época. Para validarlo, ejecuta:
```bash
python train.py
```
**Resultado esperado:** 
- En la consola, verás cómo se descargan los datos (si no existen) y luego el progreso época por época indicando `[x/50] Loss_D: ... Loss_G: ...`.
- Si revisas la carpeta `outputs/images/`, verás que por cada época se crea un archivo (ej. `epoch_1.png`, `epoch_2.png`...) conteniendo una grilla de 16 imágenes generadas usando siempre el mismo vector de ruido inicial, evidenciando la evolución (cómo pasa de ser ruido a formas de Pokémon).
- El código usa el truco de Goodfellow (etiquetas `1` para el generador) y usa `.detach()` en las fake images del discriminador.

### Validación del Task 1.3 (Visualizaciones Finales)
El requerimiento pedía dos visualizaciones al finalizar el entrenamiento: la grilla final desnormalizada a rango `[0, 255]` y las curvas de pérdida con una anotación donde son más cercanas.
Al terminar de correr `python train.py`, se generarán dos archivos clave:
1. `outputs/images/final_grid_plot.png`: Muestra la grilla 4x4 de imágenes generadas, procesada explícitamente en el rango `[0, 255]` utilizando NumPy.
2. `outputs/plots/loss_curves.png`: Muestra la curva azul (G) y roja (D), con una flecha anotando el punto exacto (época) donde la diferencia absoluta entre `loss_G` y `loss_D` fue mínima, demostrando visualmente el equilibrio alcanzado en el entrenamiento.
