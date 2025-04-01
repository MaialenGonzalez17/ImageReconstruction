# practicas_mgonzalezp

Este repositorio recoge las prácticas de Maialen Gonzalez Pancorbo de MU desde Noviembre 2024 - Julio 2025.

El objetivo principal de este proyecto es desarrollar un algoritmo que mejore la calidad de las imágenes de una sonda transvaginal, para luego combinarlo con un modelo de detección que permita visualizar el cérvix en tiempo real. 

## Experimentación de mejora de calidad de imagen

### Contexto
El dispositivo incorpora una cámara en la punta de la sonda y una pantalla para mostrar imágenes en tiempo real. Sin embargo, diversos factores afectan la calidad de estas imágenes, los cuales pueden clasificarse en tres categorías:

1. Factores asociados al hardware: Incluyen limitaciones en las dimensiones físicas del sensor, que reducen la captación de luz y afectan la relación señal-ruido (SNR); distorsión geométrica debido a la lente gran angular, que altera la forma de las estructuras capturadas; y retraso en la transmisión, lo que puede dificultar procedimientos médicos en tiempo real.

2. Factores del entorno clínico: La presencia de un profiláctico en el dispositivo puede generar reflejos y distorsiones ópticas, mientras que la humedad y los fluidos biológicos pueden ensuciar la lente, afectando la nitidez de la imagen.

3. Factores de captura de imagen: Incluyen ruido generado por el sistema de captura (artefactos de movimiento y compresión), iluminación inadecuada (excesiva, insuficiente o no uniforme), bajo contraste que dificulta la diferenciación de tejidos, y problemas de color debido a las limitaciones de los sensores digitales, lo que puede generar tonos alterados y sombras no deseadas.

### Métricas para evaluar la mejora
Para determinar el nivel adecuado de mejora en la calidad de las imágenes, es fundamental definir qué se entiende por buena calidad en una imagen endoscópica. Dado que el objetivo es facilitar el trabajo de guiado de la sonda a un profesional clínico, su criterio de calidad de imagen es clave para lograr la mejora. Como este criterio es difícil de describir con precisión por su componente subjetivo, se propone evaluar las imágenes desde otro enfoque. Por ese motivo, además de la opinión de los profesionales, también se han calculado métricas cuantitativas: 

- **Métricas sin referencia**: Sirven para evaluar la similitud entre una imagen original y su versión reconstruida o la degradada. Permiten calcular el nivel de ruido y la similitud entre la imagen degradada y la original, así como entre la imagen procesada y la original, con el fin de evaluar si la técnica aplicada ha logrado reducir el ruido y mejorar la calidad de la imagen, acercándola más a la original. 
Se utilizan a lo largo del proyecto para evaluar qué técnicas logran mejorar de manera más efectiva las imágenes degradadas.

# Métricas de Evaluación

## Métricas de Fidelidad
Estiman la similitud a nivel de píxel entre la imagen referencial y la imagen derivada.

| Métrica | Función | Valor óptimo |
|---------|---------|--------------|
| MSE     | Mide la diferencia promedio entre los píxeles de la imagen original y la procesada | Cuanto más alto, más diferencias |
| PSNR    | Compara el nivel máximo de la señal con el ruido de fondo. Un valor más alto indica mejor calidad | PSNR más alto indica una calidad mejor y más cercana a la imagen original (30 dB a 40 dB) |

## Métricas de Calidad
A diferencia de las métricas de fidelidad, estas consideran la percepción humana, evaluando mejor cómo las diferencias entre imágenes afectan la calidad percibida.

| Métrica | Función | Valor óptimo |
|---------|---------|--------------|
| SSIM    | Evalúa la similitud estructural entre dos imágenes, teniendo en cuenta luminancia, contraste y estructura | Un valor de SSIM más cercano a 1 indica una mayor similitud con la imagen original |
| LPIPS   | Evalúa la similitud perceptual entre imágenes, alineándose mejor con la percepción humana que métricas tradicionales | Más bajo significa más parecido |
| VDP     | Evalúa las diferencias físicas visibles entre pares de imágenes. Toma en cuenta cómo un observador humano vería las diferencias en la imagen | Un valor cercano a 0 indica que no hay diferencias visibles entre las imágenes. |



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


La carpeta **[TFG_mgonzalezp](https://gitlab.com/vicomtech/v6/projects/VISUALIZE_INNITIUS/practicas_mgonzalezp/-/tree/features/TFG_mgonzalezp)** contiene los algoritmos finales.


