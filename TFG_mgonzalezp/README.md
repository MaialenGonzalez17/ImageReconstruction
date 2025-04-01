# Algoritmo basado en métodos tradicionales
Este repositorio contiene un pipeline de mejora de imágenes que aplica varios filtros y ajustes para mejorar la calidad visual de las imágenes. Además, se incluyen métricas de evaluación de calidad para medir la efectividad de los procesos aplicados.

## Desarollo del algoritmo
El script realiza las siguientes etapas:

1. Carga y Redimensionado:

    - Se carga la imagen desde el disco y se ajusta su tamaño a 1024x1365 píxeles.

2. Preprocesamiento:

    - Reducción de Ruido: Se aplica un filtro de mediana para suavizar la imagen.
    - Corrección de Contraste: Se usa CLAHE en el espacio de color LAB para mejorar el contraste de la imagen.
    - Reducción de Saturación: Se disminuye la saturación en un 10% en el espacio de color HSV.
    - Agudización: Se aplica un filtro de convolución para resaltar bordes y detalles.
    - Normalización: Se reescala la imagen a un rango de 0 a 255.

3. Evaluación de la Imagen:

    - Se calculan diversas las métricas de calidad

4. Guardado de Resultados:

    - Se almacena la imagen procesada en un directorio de salida.
    - Se guardan las métricas de evaluación en un archivo CSV.

## Resultados

# Evaluación del Pipeline de Mejora de Imágenes

El pipeline desarrollado ha sido probado con una base de datos pública. Se han comparado las métricas de calidad para determinar si las mejoras obtenidas son óptimas desde un punto de vista cuantitativo.

## Métricas de Calidad
<div align="center">

| Métrica              | Imagen Original | Imagen Procesada |
|----------------------|----------------|----------------|
| **Entropía**        | 7.12           | 7.29           |
| **Contraste**       | 54.56          | 70.22          |
| **Nitidez**         | 0.52           | 4.97           |
| **Colorido**        | 119.72         | 137.22         |
| **BRISQUE**         | 16.116         | 12.77          |
| **NIQE**           | 7.01           | 34.63          |
</div>

### Análisis de Resultados

- La imagen procesada presenta una mayor **entropía**, lo que refleja un incremento en la cantidad de detalles visuales.
- El **contraste** ha mejorado significativamente, facilitando una mejor diferenciación de las estructuras.
- La **nitidez** ha aumentado notablemente, ofreciendo una imagen más clara y definida.
- Los **colores** se perciben más intensos debido a una mayor saturación.
- Aunque el valor de **NIQE** sugiere una menor calidad, esto se debe a que compara la imagen con referencias "naturales" y puede penalizar un realce agresivo de detalles o contrastes.
- En contraste, **BRISQUE** indica una mejora en la calidad basada en la percepción humana.

Como se ha mencionado anteriormente, es fundamental una evaluación cualitativa para determinar si estas mejoras son realmente significativas en un entorno clínico. Se muestran a continuación las diferencias de calidad visuales.
<div align="center">
![Comparación de imágenes](images/images.png)
</div>
