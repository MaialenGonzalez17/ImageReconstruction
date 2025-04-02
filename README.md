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

- **Métricas con referencia**: Sirven para evaluar la similitud entre una imagen original y su versión reconstruida o la degradada. Permiten calcular el nivel de ruido y la similitud entre la imagen degradada y la original, así como entre la imagen procesada y la original, con el fin de evaluar si la técnica aplicada ha logrado reducir el ruido y mejorar la calidad de la imagen, acercándola más a la original. 
Se utilizan a lo largo del proyecto para evaluar qué técnicas logran mejorar de manera más efectiva las imágenes degradadas.


##### Métricas de Fidelidad
Estiman la similitud a nivel de píxel entre la imagen referencial y la imagen derivada.

| Métrica | Función | Valor óptimo |
| --- | --- | --- |
| **MSE** | Mide la diferencia promedio entre los píxeles de la imagen original y la procesada | Valores cercanos a cero indican alta similitud |
| **PSNR** | Compara el nivel máximo de la señal con el ruido de fondo. Un valor más alto indica mejor calidad | Valores superiores a 30 dB suelen indicar una buena calidad de imagen |

##### Métricas de Calidad
A diferencia de las métricas de fidelidad, estas consideran la percepción humana, evaluando mejor cómo las diferencias entre imágenes afectan la calidad percibida.

| Métrica | Función | Valor óptimo/mínimo |
| --- | --- | --- |
| **SSIM** | Evalúa la similitud estructural entre dos imágenes, teniendo en cuenta luminancia, contraste y estructura | Un valor de SSIM más cercano a 1 indica una mayor similitud con la imagen original (>0.85) |
| **LPIPS** | Evalúa la similitud perceptual entre imágenes, alineándose mejor con la percepción humana que métricas tradicionales | Más bajo significa más parecido (<0.1) |
| **VDP - Visible Differences Predictor** | Evalúa las diferencias físicas visibles entre pares de imágenes. Toma en cuenta cómo un observador humano vería las diferencias en la imagen | LPIPS es razonablemente cercano en términos de comparaciones de calidad visual, por lo que no es necesario calcular esta métrica. |



- **Métricas sin referencia**: Estas métricas solo utilizan un dato de entrada: la imagen analizada. Por lo tanto, la evaluación de la calidad de la imagen se realiza sin la ventaja de datos comparativos o de referencia, lo que significa que se basa únicamente en los atributos inherentes de la propia imagen.  
Se utilizan a lo largo del proyecto para calcular la mejora de una imagen tras ser procesada. 
Se utilizan a lo largo del proyecto para calcular la mejora de una imagen tras ser procesada. 



##### Métricas de Calidad

| Métrica | Función | Valor óptimo/mínimo |
| --- | --- | --- |
| **Entropía** | Mide la cantidad de información o detalle presente en una imagen. | Valores más altos son preferibles, en un rango de 0 a 8. |
| **Contraste** | Evalúa la variación en el brillo entre diferentes áreas de la imagen. | El contraste se ajusta en un rango de 0 a 100, donde 0 es el mínimo y 100 es el máximo (diferencias marcadas entre claras y oscuras). Valores óptimos entre 40 y 70%. |
| **Nitidez** | Indica la claridad de los bordes y detalles en una imagen. | La nitidez varía entre 0 (mínima nitidez, imagen borrosa) y 100 (máxima nitidez, resaltando en exceso los detalles). Para una calidad óptima de imagen, se recomienda un valor entre 40 y 50. |
| **Colorido** | Mide la saturación y gama de colores en la imagen. | 0%: Representa una imagen en escala de grises <br> 100%: Indica la máxima saturación <br> El nivel de color adecuado es alrededor del 50%. |
| **Calidad General** | Evalúa la calidad perceptual de la imagen desde la perspectiva de un observador humano. | La calidad general se evalúa mediante las métricas finales de calidad, complementadas con la valoración cualitativa de los expertos. |
| **Diversidad** | Representa la variabilidad en colores, texturas y detalles presentes en la imagen. | La diversidad de una imagen se mide a través de la entropía por lo que no es necesario calcularla. |

##### Métricas Basadas en Redes Neuronales

Existen otras métricas sin referencia que utilizan un modelo entrenado para calcular una puntuación de calidad. Estos modelos buscan detectar distorsiones de manera similar a cómo lo haría un ojo humano:

| Métrica | Función | Valor óptimo/mínimo |
| --- | --- | --- |
| **BRISQUE - Blind/Referenceless Image Spatial Quality Evaluator** | Evalúa la calidad perceptual de una imagen a través de características estadísticas. | Un valor más bajo indica una mejor calidad de imagen. (óptimo → 0) |
| **NIQE - Natural Image Quality Evaluator** | Utiliza estadísticas de la imagen para predecir la calidad percibida por un observador humano. | Valores más bajos corresponden a imágenes de mayor calidad. (óptimo → 0) |
| **NIMA - Neural Image Assessment** | Se basa en una red neuronal convolucional que funciona puntuando imágenes de forma fiable y con alta correlación con la percepción humana. | 1 - 4 → Baja calidad <br> 4 - 7 → Calidad aceptable <br> 7 - 10 → Alta calidad. |


## Contenido

Realizar una mejora de la calidad de imagen es fundamental para lograr los mejores resultados visuales y cuantitativos posibles. Se han propuesto dos tipos de métodos de procesamiento para mejorar las imágenes médicas:   

- Algoritmo de basado en métodos tradiconales
- Algoritmo basado en aprendizaje profundo

Para ver información más detallada de cada uno ver **[TFG_mgonzalezp](https://gitlab.com/vicomtech/v6/projects/VISUALIZE_INNITIUS/practicas_mgonzalezp/-/tree/features/TFG_mgonzalezp)**.


