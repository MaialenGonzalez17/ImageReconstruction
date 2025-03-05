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

## Contenido

El estudio se centra en dos algoritmos: 

- Algoritmo de métodos tradiconales: actúan eliminando el ruido de una imagen y aplicándoles un filtro que conserve los bordes mejorando así la relación señal-ruido máxima (PSNR).
- Algoritmo de IA: Utilizan herramientas como redes neuronales convolucionales, autocodificadores y redes generativas adversarias para aprender automáticamente a mejorar la calidad de las imágenes, destacándose por preservar detalles y colores con gran precisión.

Para desarrollar ambos algoritmos, el primer paso ha sido analizar y recopilar información de estudios previos disponibles en el mercado. Una vez identificados los estudios que mejor resultados obtuviesen en la corrección de calidad, se ha realizado una comparación detallada de las métricas y resultados que estos obtenian. Esto ha permitido determinar que procesos utilizados son los más adecuados o los que mejor resultados pueden obtener en nuestra base de datos.

Finalmente, la carpeta **[TFG_mgonzalezp](https://gitlab.com/vicomtech/v6/projects/VISUALIZE_INNITIUS/practicas_mgonzalezp/-/tree/features/TFG_mgonzalezp)** contiene los pipelines finales, que integran las funciones con las mejores métricas aplicadas a la base de datos.

**Para más información, ver dentro de la carpeta.**
