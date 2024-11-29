# practicas_mgonzalezp

Este repositorio recoge las prácticas de Maialen Gonzalez Pancorbo de MU desde Noviembre 2024 - Julio 2025.

FIXME: mgonzalezp revisar los bloques y editar
Bloques de trabajo principales:
- Regulatoria para recopilar propósito del software y requisitos
- Formación técnica (programación Python, computer vision, deep learning)
- Anotación de imágenes
- Algoritmia de visión artificial para mejora de calidad de imagen
- (Opcional) Algoritmia de visión artificial para mejora de calidad de imagen

## Experimentación de mejora de calidad de imagen

### Contexto
Las dimensiones de calidad de imagen a mejorar son:
- Eliminación de ruido
- Corrección de contraste
- Corrección de iluminación
- Normalización
- Corrección de color

### Métricas para evaluar la mejora
Las métricas de valoración cuantitativa de calidad de imagen a optimizar son:

- Valores de cada imagen
    - Entropía - Mide la información contenida en la imagen, con valores más altos indicando más detalle.
    - Contraste – Refleja la variación en el brillo, valores más altos son preferibles.
    - Nitidez – Indica la claridad de los bordes y los detalles, valores más altos son preferibles.

- Comparación entre imágenes
    - PSNR  - valor más alto, mejor fidelidad a la imagen original. Valores entre 30-40: Buena calidad, lo deseado para la mayoría de las aplicaciones.
    - SSIM - 0.9 ≤ SSIM < 1: Buena Calidad, diferencias poco perceptibles.
    - MSE – Más alto el valor, más diferencias.

### Experimentación de algoritmos

#### Pipeline clásico 

TODO: Completar con algoritmo de pnmartinez

#### Mejora de iluminación con Deep Learning – Low light enhancement with T-Diffusion (MICCAI 2024)

No parece funcionar. pnmartinez está tratando de revisar.



