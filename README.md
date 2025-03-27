# practicas_mgonzalezp

Este repositorio recoge las prácticas de Maialen Gonzalez Pancorbo de MU desde Noviembre 2024 - Julio 2025.

El objetivo principal de este proyecto es desarrollar el algoritmo que mejore la calidad de las imágenes de una sonda transvaginal, para luego combinarlo con un modelo de detección que permita visualizar el cérvix en tiempo real. 

## Experimentación de mejora de calidad de imagen

### Contexto
El dispositivo incorpora una cámara en la punta de la sonda y una pantalla para mostrar imágenes en tiempo real. Sin embargo, diversos factores afectan la calidad de estas imágenes, los cuales pueden clasificarse en tres categorías:

1. Factores asociados al hardware: Incluyen limitaciones en las dimensiones físicas del sensor, que reducen la captación de luz y afectan la relación señal-ruido (SNR); distorsión geométrica debido a la lente gran angular, que altera la forma de las estructuras capturadas; y retraso en la transmisión, lo que puede dificultar procedimientos médicos en tiempo real.

2. Factores del entorno clínico: La presencia de un profiláctico en el dispositivo puede generar reflejos y distorsiones ópticas, mientras que la humedad y los fluidos biológicos pueden ensuciar la lente, afectando la nitidez de la imagen.

3. Factores de captura de imagen: Incluyen ruido generado por el sistema de captura (artefactos de movimiento y compresión), iluminación inadecuada (excesiva, insuficiente o no uniforme), bajo contraste que dificulta la diferenciación de tejidos, y problemas de color debido a las limitaciones de los sensores digitales, lo que puede generar tonos alterados y sombras no deseadas.

### Métricas para evaluar la mejora
Para determinar el nivel adecuado de mejora en la calidad de las imágenes, es fundamental definir qué se entiende por buena calidad en una imagen endoscópica. Dado que el objetivo es facilitar el trabajo de guiado de la sonda a un profesional clínico, su criterio de calidad de imagen es clave para lograr la mejora. Como este criterio es difícil de describir con precisión por su componente subjetivo, se propone evaluar las imágenes desde otro enfoque. Por ese motivo, además de la opinión de los profesionales, también se han calculado métricas cuantitativas: 

- **Métricas sin referencia**: Estas métricas analizan la imagen basándose en modelos de percepción visual para estimar su calidad.  Se utilizan a lo largo del proyecto para calcular la mejora de una imagen tras ser procesada. Algunos ejemplos incluyen:
    - Entropía - Es una medida de la cantidad de información o detalle que contiene una imagen (en un margen de 0 a 8). Una imagen con alta entropía tiene más variación en sus píxeles y detalles, lo que significa que es más diversa.
    - Contraste – Refleja la variación en el brillo, valores más altos son preferibles.
    - Nitidez – Indica la claridad de los bordes y los detalles, valores más altos son preferibles. Cuantos más detalles y más contraste tengan estos, mayor será la calidad de imagen.
    - Colorido - Indica la saturación y la gama de colores de la imagen. Se evalúa para determinar si los colores son naturales o si la imagen está sobresaturada, lo que puede hacer que se vea artificial o poco realista.
    - Calidad general - Mide la calidad general de una imagen de manera perceptual, es decir, cómo se vería desde la perspectiva de un observador humano. La calidad general se evalúa mediante las métricas finales de calidad, complementadas con la valoración cualitativa de los expertos.
    - Diversidad: Indica la cantidad de variabilidad en los colores, texturas y detalles presentes en la imagen. La diversidad de una imagen se mide a través de la entropía.


- **Métricas con referencia**: Sirven para evaluar la similitud entre una imagen original y su versión reconstruida o la degradada. Se utilizan a lo largo del proyecto para evaluar qué técnicas logran mejorar de manera más efectiva las imágenes degradadas. Estas métricas permiten calcular el nivel de ruido y la similitud entre la imagen degradada y la original, así como entre la imagen procesada y la original, con el fin de evaluar si la técnica aplicada ha logrado reducir el ruido y mejorar la calidad de la imagen, acercándola más a la original.
    - PSNR - (Peak Signal-to-Noise Ratio): Medida que compara el nivel máximo de una señal con el nivel de ruido de fondo. Un PSNR más alto indica una calidad mejor y más cercana a la imagen original (30 dB a 40 dB). 
    - SSIM - (Structural Similarity Index Measure): Medida de cuán similar es una imagen a una de referencia, en términos de estructura, percepción y luminosidad. Un valor de SSIM más cercano a 1 indica una mayor similitud con la imagen original. 
    - MSE – (Mean square error): El MSE calcula la media del error cuadrático entre los píxeles correspondientes de dos imágenes. Este valor siempre es positivo y un MSE de cero indica la ausencia total de error. Cuanto más alto, más diferencias.
    - Fidelidad: Indica cuanto se conservan las características originales de la imagen después de un proceso de mejora o transformación. Dado que ya se están utilizando PSNR y SSIM para medir la similitud estructural de la imagen, no es necesario calcularla.

### Experimentación de algoritmos

#### Pipeline clásico 

TODO: Completar con algoritmo de pnmartinez

#### Mejora de iluminación con Deep Learning – Low light enhancement with T-Diffusion (MICCAI 2024)

No parece funcionar. pnmartinez está tratando de revisar.

## Contenido

El estudio se centra en dos algoritmos: 

- Algoritmo de métodos tradiconales:  Para la reducción de ruido, se ha seleccionado el Filtro de Mediana por su equilibrio entre calidad y velocidad. En la mejora del contraste, CLAHE ha resultado ser la opción más efectiva al resaltar detalles sin aumentar el ruido. Para el realce de bordes, se ha descartado el filtro Canny en favor de la función de agudización, que ha mejorado la nitidez. Finalmente, se ha aplicado normalización y ajuste de saturación para optimizar la representación visual de las imágenes.

- Algoritmo de IA: Utilizan herramientas como redes neuronales convolucionales, autocodificadores y redes generativas adversarias para aprender automáticamente a mejorar la calidad de las imágenes, destacándose por preservar detalles y colores con gran precisión.

Para desarrollar ambos algoritmos, el primer paso ha sido analizar y recopilar información de estudios previos disponibles en el mercado. Una vez identificados los estudios que mejor resultados obtuviesen en la corrección de calidad, se ha realizado una comparación detallada de las métricas y resultados que estos obtenian. Esto ha permitido determinar que procesos utilizados son los más adecuados o los que mejor resultados pueden obtener en nuestra base de datos.

La carpeta **[TFG_mgonzalezp](https://gitlab.com/vicomtech/v6/projects/VISUALIZE_INNITIUS/practicas_mgonzalezp/-/tree/features/TFG_mgonzalezp)** contiene los algoritmos finales.


