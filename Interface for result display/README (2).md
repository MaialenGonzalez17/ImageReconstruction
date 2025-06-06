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

#### Métricas con Referencia (Full-Reference, FR)

Estas métricas comparan una imagen procesada con su versión original. Permiten cuantificar cuánto se ha degradado o mejorado una imagen tras aplicar una técnica de restauración. Se emplean en este proyecto para determinar qué métodos de mejora obtienen los mejores resultados en imágenes degradadas.

##### Métricas de Fidelidad

Evalúan la similitud a nivel de píxel entre la imagen original (referencial) y la imagen generada.

| Métrica | Descripción | Valor deseable |
|--------|-------------|----------------|
| **MSE** | Error cuadrático medio: mide la diferencia promedio entre píxeles. | Valores cercanos a 0 (< 3) |
| **PSNR** | Relación señal-ruido: compara la señal con el ruido de fondo. | Mayor a 30 dB |

##### Métricas Perceptuales

Tienen en cuenta la percepción visual humana, y evalúan cómo las diferencias afectan la calidad percibida.

| Métrica | Descripción | Valor deseable |
|--------|-------------|----------------|
| **SSIM** | Índice de similitud estructural: evalúa similitud estructural, luminancia y contraste. | Superior a 0.85 (rango 0–1) |
| **LPIPS** | Learned Perceptual Image Patch Similarity: mide la similitud perceptual usando redes neuronales. | Valores bajos (< 0.2) |

---

#### Métricas sin Referencia (No-Reference, NR)

Estas métricas analizan únicamente la imagen degradada o restaurada, sin necesidad de comparación con una imagen original. Son útiles cuando no se dispone de una referencia, y se utilizan a lo largo del proyecto para valorar la mejora obtenida tras el procesamiento.

##### Métricas Tradicionales

| Métrica | Descripción | Valor deseable |
|--------|-------------|----------------|
| **Entropía** | Mide la cantidad de información presente en la imagen. | Mayor a 7 (rango 0–8) |
| **Contraste** | Calcula la variación del brillo entre áreas. | Entre 40–70% (rango 0–100) |
| **Nitidez** | Indica la claridad de los bordes y detalles. | Entre 60–70% (rango 0–100) |
| **Colorido** | Mide la saturación y gama de colores. | Alrededor del 50% (rango 0–100) |

##### Métricas Basadas en Redes Neuronales

Utilizan modelos entrenados para simular la percepción humana de la calidad visual.

| Métrica | Descripción | Valor deseable |
|--------|-------------|----------------|
| **BRISQUE** | Analiza la imagen imitando la percepción del ojo humano. | Más bajo es mejor (óptimo → 0, ideal < 0.2) |
| **NIQE** | Compara la imagen con un modelo estadístico basado en imágenes naturales. | Más bajo es mejor (óptimo → 0, ideal < 0.2) |
| **NIMA** | Puntuación generada por una red neuronal en función de la percepción humana. | 1–4: Baja calidad, 4–7: Aceptable, 7–10: Alta calidad |

## Contenido

Realizar una mejora de la calidad de imagen es fundamental para lograr los mejores resultados visuales y cuantitativos posibles. Se han propuesto dos tipos de métodos de procesamiento para mejorar las imágenes médicas:   

- **[Algoritmo de basado en métodos tradiconales](https://gitlab.com/vicomtech/v6/projects/VISUALIZE_INNITIUS/practicas_mgonzalezp/-/tree/develop/TFG_mgonzalezp/Algoritmo%20basado%20en%20m%C3%A9todos%20tradicionales)**.

- **[Algoritmo basado en aprendizaje profundo](https://gitlab.com/vicomtech/v6/projects/VISUALIZE_INNITIUS/practicas_mgonzalezp/-/tree/develop/Algoritmo%20basado%20en%20aprendizaje%20profundo)**.

Para visualizar el funcionamiento de ambos algoritmos de forma dinámica, puede acceder a la siguiente interfaz de visualización de resultados:
- **[Interfaz para la visualización de resultados](https://gitlab.com/vicomtech/v6/projects/VISUALIZE_INNITIUS/practicas_mgonzalezp/-/tree/develop/Interfaz%20para%20la%20visualizaci%C3%B3n%20de%20resultados?ref_type=heads)**.


