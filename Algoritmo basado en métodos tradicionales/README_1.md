# Algoritmo basado en métodos tradicionales

Este repositorio contiene un pipeline de mejora de imágenes que aplica varios filtros y ajustes para mejorar la calidad visual de las imágenes. Además, se incluyen métricas de evaluación de calidad para medir la efectividad de los procesos aplicados.

## Desarrollo del algoritmo

El script realiza las siguientes etapas:

1. **Carga y Redimensionado:**

    - Se carga la imagen desde el disco y se ajusta su tamaño a 1024x1365 píxeles.

2. **Preprocesamiento:**

    - Reducción de Ruido: Se aplica un filtro de mediana para suavizar la imagen.
    - Corrección de Contraste: Se usa CLAHE en el espacio de color LAB para mejorar el contraste de la imagen.
    - Reducción de Saturación: Se disminuye la saturación en un 10% en el espacio de color HSV.
    - Agudización: Se aplica un filtro de convolución para resaltar bordes y detalles.
    - Normalización: Se reescala la imagen a un rango de 0 a 255.

3. **Evaluación de la Imagen:**

    - Se calculan diversas métricas de calidad.

4. **Guardado de Resultados:**

    - Se almacena la imagen procesada en un directorio de salida.
    - Se guardan las métricas de evaluación en un archivo CSV.

## Resultados

# Evaluación del Pipeline de Mejora de Imágenes

El pipeline desarrollado ha sido probado con una base de datos pública. Se han comparado las métricas de calidad para determinar si las mejoras obtenidas son óptimas desde un punto de vista cuantitativo.

## Métricas de Calidad

Las métricas sin referencia analizan únicamente la imagen degradada o restaurada sin comparar con la original. Las métricas con referencia calculan la similitud entre la imagen original-degradada y la imagen original-restaurada.

### Tabla 1: Resultados de métricas sin referencia en métodos tradicionales

| Transformación           | Comparación | Entropía (>7) | Contraste (%) | Nitidez (%) | Colorido (%) | Brisque (<0.2) | Niqe (>0.2) | Nima (>8) |
|-------------------------|-------------|---------------|---------------|-------------|--------------|----------------|-------------|-----------|
| Desenfoque Gaussiano    | Degradada   | 7.21          | 20.98         | 2.91        | 44.35        | 0.72           | 0.64        | 2.33      |
|                         | Restaurada  | 7.45          | 23.76         | 0.76        | 45.76        | 0.31           | 0.58        | 2.85      |
| Ruido Gaussiano         | Degradada   | 7.64          | 20.67         | 100         | 54.32        | 0.72           | 0.40        | 6.34      |
|                         | Restaurada  | 7.74          | 22.61         | 100         | 56.47        | 0.30           | 0.42        | 6.46      |
| Movimiento Involuntario | Degradada   | 7.10          | 20.51         | 13.85       | 45.82        | 0.51           | 0.61        | 2.41      |
|                         | Restaurada  | 7.37          | 22.66         | 92.07       | 46.10        | 0.24           | 0.58        | 3.05      |
| Iluminación y contraste | Degradada   | 7.01          | 21.05         | 67.6        | 47.19        | 0.20           | 0.63        | 2.87      |
|                         | Restaurada  | 7.33          | 23.71         | 99.29       | 48.94        | 0.12           | 0.59        | 3.64      |
| Iluminación desigual    | Degradada   | 7.11          | 26.72         | 88.38       | 42.89        | 0.17           | 0.67        | 3.69      |
|                         | Restaurada  | 7.30          | 27.90         | 99.81       | 44.63        | 0.12           | 0.64        | 4.34      |
| Sombras aleatorias      | Degradada   | 6.50          | 14.78         | 60.15       | 27.9         | 0.26           | 0.78        | 2.15      |
|                         | Restaurada  | 7.00          | 19.32         | 100         | 35.89        | 0.11           | 0.65        | 3.02      |
| Reflejos especulares    | Degradada   | 7.18          | 21.46         | 81.92       | 48.82        | 0.15           | 0.62        | 2.86      |
|                         | Restaurada  | 7.42          | 23.27         | 100         | 48.99        | 0.11           | 0.59        | 3.60      |
| Defectos del preservativo | Degradada | 7.14          | 21.53         | 91.09       | 47.56        | 0.10           | 0.60        | 3.25      |
|                         | Restaurada  | 7.40          | 23.72         | 100         | 48.58        | 0.07           | 0.56        | 4.40      |

---

### Tabla 2: Resultados de las métricas con referencia en métodos tradicionales

| Transformación           | Comparación           | PSNR (>30dB) | SSIM (>0.9) | MSE (~0) | LPIPS (<0.1) |
|-------------------------|----------------------|--------------|-------------|----------|--------------|
| Desenfoque Gaussiano    | Original - Degradada  | 35.81        | 0.88        | 18.91    | 0.16         |
|                         | Original - Restaurada | 29.27        | 0.82        | 77.72    | 0.13         |
| Ruido Gaussiano         | Original - Degradada  | 28.12        | 0.14        | 100.34   | 0.75         |
|                         | Original - Restaurada | 28.10        | 0.20        | 100.64   | 0.72         |
| Movimiento Involuntario | Original - Degradada  | 36.82        | 0.89        | 16.05    | 0.07         |
|                         | Original - Restaurada | 29.58        | 0.81        | 72.50    | 0.09         |
| Iluminación y contraste | Original - Degradada  | 29.61        | 0.93        | 84.23    | 0.01         |
|                         | Original - Restaurada | 29.98        | 0.81        | 83.93    | 0.07         |
| Iluminación desigual    | Original - Degradada  | 31.98        | 0.96        | 45.13    | 0.02         |
|                         | Original - Restaurada | 29.47        | 0.81        | 74.94    | 0.08         |
| Sombras aleatorias      | Original - Degradada  | 28.45        | 0.82        | 95.35    | 0.09         |
|                         | Original - Restaurada | 29.16        | 0.82        | 82.06    | 0.11         |
| Reflejos especulares    | Original - Degradada  | 39.23        | 0.95        | 8.25     | 0.07         |
|                         | Original - Restaurada | 29.48        | 0.81        | 73.91    | 0.14         |
| Defectos del preservativo | Original - Degradada | 41.08       | 0.94        | 5.52     | 0.10         |
|                         | Original - Restaurada | 29.58        | 0.80        | 72.95    | 0.18         |

---

### Análisis de Resultados

Las métricas sin referencia indican que la imagen procesada mejora características visuales como detalle, nitidez y colorido, reflejando una mejor calidad perceptual. Sin embargo, las métricas con referencia muestran valores inferiores en la comparación original-restaurada, lo que evidencia que el algoritmo no logra revertir completamente las degradaciones ni recuperar fielmente la imagen original. Por tanto, aunque el pipeline mejora la percepción visual, no consigue una restauración fiel a la imagen original, limitando su eficacia en aplicaciones donde la precisión es crítica.