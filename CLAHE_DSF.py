import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# Función para aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    """
    Aplica CLAHE a una imagen en escala de grises o en color (BGR).

    image: Imagen de entrada (escala de grises o color).
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    if len(image.shape) == 3:  # Si la imagen es en color (BGR)
        # Convertir a espacio de color LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)  # Aplicar CLAHE al canal L
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # Si la imagen es en escala de grises
        return clahe.apply(image)


# Función para aplicar la función de detención de difusión (DSF)
def diffusion_stopping_function(image, iterations=10, kappa=30):
    """
    Aplica un proceso de difusión de detención para suavizar la imagen mientras se conservan los bordes.

    image: Imagen en escala de grises.
    iterations: Número de iteraciones para el proceso de difusión.
    kappa: Parámetro de difusión.
    """
    img_float = np.float32(image)

    for _ in range(iterations):
        # Gradientes en las direcciones X e Y
        grad_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)

        # Magnitud del gradiente (para detectar los bordes)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Coeficiente de difusión basado en la magnitud del gradiente
        diffusion = 1.0 / (1.0 + (grad_magnitude / kappa) ** 2)

        # Aplicar el proceso de difusión
        img_float += diffusion * (cv2.Laplacian(img_float, cv2.CV_32F))

    return np.uint8(np.clip(img_float, 0, 255))


# Función para procesar la imagen y combinar CLAHE con DSF
def process_image(image_path):
    # Cargar la imagen original
    img = cv2.imread(image_path)

    # Convertir la imagen a escala de grises para el procesamiento
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar CLAHE a la imagen en color
    clahe_img = apply_clahe(img)

    # Aplicar la función de detención de difusión (DSF)
    dsf_img = diffusion_stopping_function(img_gray)

    # Combinar las imágenes procesadas (usamos un peso para cada una)
    combined_img = cv2.addWeighted(clahe_img, 0.5, cv2.cvtColor(dsf_img, cv2.COLOR_GRAY2BGR), 0.5, 0)

    return img, clahe_img, dsf_img, combined_img


# Mostrar las imágenes
image_path = "Public BDD (Intel)/0.jpg"  # Ruta de la imagen
original_img, clahe_img, dsf_img, combined_img = process_image(image_path)

# Mostrar las imágenes en una sola figura
plt.figure(figsize=(15, 10))

# Imagen original
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# Imagen con CLAHE
plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
plt.title("CLAHE Enhanced Image")
plt.axis('off')

# Imagen con DSF
plt.subplot(1, 4, 3)
plt.imshow(dsf_img, cmap='gray')
plt.title("DSF Smoothed Image")
plt.axis('off')

# Imagen combinada
plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
plt.title("Combined Image (CLAHE + DSF)")
plt.axis('off')

# Mostrar todas las imágenes
plt.show()