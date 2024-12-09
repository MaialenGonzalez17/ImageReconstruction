import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Función para calcular PSNR
def calculate_psnr(img1, img2):
    """
    Calcula el valor PSNR (Peak Signal-to-Noise Ratio) entre dos imágenes.
    """
    mse_value = calculate_mse(img1, img2)
    if mse_value == 0:
        return 100  # Las imágenes son idénticas
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse_value))
    return psnr

# Función para calcular MSE
def calculate_mse(img1, img2):
    """
    Calcula el valor MSE (Mean Squared Error) entre dos imágenes.
    """
    err = np.sum((img1 - img2) ** 2)
    mse = err / float(img1.shape[0] * img1.shape[1])
    return mse

# Función para calcular SSIM
def calculate_ssim(img1, img2):
    """
    Calcula el valor SSIM (Structural Similarity Index) entre dos imágenes.
    """
    # Verificamos si la imagen está en color o en escala de grises
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        # Convertir a escala de grises si la imagen tiene 3 canales (color)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1  # Ya está en escala de grises

    if len(img2.shape) == 3 and img2.shape[2] == 3:
        # Convertir a escala de grises si la imagen tiene 3 canales (color)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2  # Ya está en escala de grises

    # Calcular SSIM entre las dos imágenes en escala de grises
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

# Función para procesar y mostrar las métricas
def display_metrics(original_img, final_img):
    """
    Muestra las métricas PSNR, SSIM y MSE entre la imagen original y la imagen final procesada.
    """
    psnr_value = calculate_psnr(original_img, final_img)
    ssim_value = calculate_ssim(original_img, final_img)
    mse_value = calculate_mse(original_img, final_img)

    # Imprimir los resultados
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"MSE: {mse_value:.2f}")

# Función para procesar la imagen (aplicar técnicas de mejora o manipulación)
def process_image(image_path):
    """
    Procesa la imagen y devuelve la imagen original y la imagen procesada.
    """
    # Cargar la imagen
    img = cv2.imread(image_path)

    # Realizar alguna mejora o procesamiento de la imagen (ejemplo: aplicar un filtro bilateral)
    processed_img = cv2.bilateralFilter(img, 15, 75, 75)

    return img, processed_img

# Ruta de la imagen
image_path = "Public BDD (Intel)/0.jpg"  # Cambia la ruta de la imagen según tu archivo

# Procesar la imagen
original_img, final_img = process_image(image_path)

# Mostrar las métricas
display_metrics(original_img, final_img)

# Mostrar las imágenes
plt.figure(figsize=(10, 5))

# Imagen original
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# Imagen final procesada
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
plt.title("Processed Image")
plt.axis('off')

# Mostrar las imágenes
plt.show()
