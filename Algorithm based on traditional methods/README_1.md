# Descripción del código de mejora de imagen sin IA

Este script realiza un proceso completo de mejora de imágenes basado en técnicas clásicas de procesamiento de imagen, sin usar inteligencia artificial. A continuación se explica su funcionamiento paso a paso:

---

## Funcionalidad principal

1. **Carga y preprocesamiento de imágenes**

   - Lee imágenes desde una carpeta especificada.
   - Redimensiona cada imagen a un tamaño fijo de 320x320 píxeles para uniformidad.

2. **Procesos de mejora de imagen**

   Se aplican varias técnicas clásicas para mejorar la calidad visual:

   - **Filtro de mediana** para reducir ruido impulsivo.
   - **Ecualización de histograma adaptativa CLAHE** en el espacio de color LAB para mejorar el contraste local sin saturar colores.
   - **Reducción de saturación** para suavizar colores intensos, aumentando naturalidad.
   - **Filtro de nitidez (sharpening)** mediante un kernel de convolución para realzar bordes y detalles.
   - **Normalización** de valores de píxeles para asegurar rango dinámico óptimo.

3. **Guardado y cálculo de métricas**

   - Guarda las imágenes mejoradas en una carpeta destino.
   - Calcula métricas clásicas de calidad de imagen comparando la imagen original con la mejorada:
     - PSNR (Peak Signal-to-Noise Ratio)
     - SSIM (Structural Similarity Index)
     - MSE (Mean Squared Error)
   - Calcula métricas sin referencia sobre la imagen mejorada:
     - Entropía (medida de información/ruido)
     - Nitidez
     - Contraste
     - Colorfulness (vibrancia de colores)

4. **Ejecución paralela**

   - Procesa múltiples imágenes simultáneamente usando `ThreadPoolExecutor` para acelerar el procesamiento.

5. **Resumen estadístico**

   - Al final, calcula la media y desviación estándar de todas las métricas obtenidas para evaluar globalmente el desempeño del proceso.
   - Guarda los resultados completos en un fichero CSV para análisis posterior.

6. **Tiempos de ejecución**

   - Muestra el tiempo total de procesamiento y el tiempo promedio por imagen.

---

## Resumen de funciones clave

- `resize_image`: Cambia el tamaño de la imagen a 320x320.
- `apply_median_filter`: Aplica filtro de mediana para eliminar ruido.
- `clahe_lab`: Mejora el contraste local en espacio LAB con CLAHE.
- `reduce_saturation`: Disminuye la saturación para suavizar colores.
- `apply_sharpening`: Realza detalles con filtro de nitidez.
- `image_enhancement_pipeline`: Aplica toda la cadena de mejora y calcula métricas.
- `process_images_in_folder`: Obtiene rutas de imágenes válidas de una carpeta.
- `process_images_in_parallel`: Ejecuta la mejora en paralelo.
- `calculate_stats`: Calcula medias y desviaciones estándar de métricas.

---

## Uso

Configurar las rutas de entrada y salida en:

```python
folder_path = "carpeta_de_imagenes_entrada/"
save_folder = "carpeta_de_imagenes_mejoradas/"
output_csv = "ruta_resultados/resultados.csv"
