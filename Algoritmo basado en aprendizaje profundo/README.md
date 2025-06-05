# Reconstrucción de Imágenes con Denoising Autoencoder

## Descripción General

Este proyecto implementa un **Denosing Autoencoder** diseñado para la **restauración de imágenes**. El modelo aprende a reconstruir imágenes limpias y de alta calidad a partir de versiones degradadas afectadas por diferentes tipos de distorsiones o artefactos. La metodología combina aprendizaje profundo con aumentos de datos dinámicos y un entrenamiento ponderado para mejorar la robustez frente a múltiples degradaciones.

---

## Qué Hace el Modelo

- **Entrada:** Imágenes degradadas generadas aplicando diferentes transformaciones sintéticas a imágenes originales limpias.
- **Salida:** Imágenes restauradas que aproximan las imágenes originales sin degradación.
- **Objetivo:** Minimizar la pérdida de reconstrucción aprendiendo a invertir las degradaciones, mejorando así la calidad de imagen en entornos donde las capturas originales están ruidosas o corruptas.

---

## Componentes Principales

### 1. Dataset y Transformaciones

- Se cargan imágenes originales desde un directorio especificado.
- Cada muestra de entrenamiento se aumenta aplicando una **transformación aleatoria** seleccionada según pesos dinámicos.
- Estas transformaciones simulan artefactos reales, como desenfoque, ruido, variaciones de iluminación, entre otros.
- Los pesos de estas transformaciones se actualizan adaptativamente durante el entrenamiento, enfocándose en aquellas degradaciones que resultan más difíciles para el modelo.

### 2. Arquitectura del Modelo

- El modelo base es un **Autoencoder Convolucional** implementado en PyTorch.
- Usa capas codificadoras y decodificadoras con **skip connections** (similar a arquitectura U-Net) para preservar detalles espaciales.
- El codificador comprime la imagen de entrada a un espacio latente de menor dimensión.
- El decodificador reconstruye la imagen desde esta representación comprimida.

### 3. Función de Pérdida

- Se emplea una función combinada de **MSE + SSIM** para optimizar tanto la precisión por píxel como la similitud perceptual.
- La inclusión de SSIM ayuda a preservar la estructura, texturas y bordes en la imagen restaurada.

### 4. Proceso de Entrenamiento

- El conjunto de datos se divide en subconjuntos de entrenamiento y validación.
- El entrenamiento se ejecuta por un número fijo de épocas, actualizando dinámicamente los pesos de las transformaciones según la pérdida de entrenamiento.
- Se guarda el modelo que obtiene mejor rendimiento en validación.
- Se generan checkpoints para permitir continuar el entrenamiento desde el último punto guardado.
- Se registra el historial de pérdidas y la evolución de los pesos de las transformaciones.

---

## Cómo Usar el Código

### Requisitos Previos

- Python 3.8 o superior
- PyTorch (probado con versiones 1.10+)
- Librerías adicionales: `numpy`, `pandas`, `openpyxl` (para guardar archivos Excel), y otras usadas en los módulos de dataset y transformaciones.

### Ejecutar el Entrenamiento

Antes de ejecutar, actualiza las rutas y parámetros en el script:

```python
train(
    image_dir="RUTA/A/TUS/IMAGENES",  # Carpeta con imágenes limpias originales
    epochs=30,
    batch_size=16,
    learning_rate=1e-4,
    val_ratio=0.15,
    save_path="weights"  # Carpeta donde se guardarán los modelos y registros
)
