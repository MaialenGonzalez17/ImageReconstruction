import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from obtain_enhancement_metrics import calculate_psnr, calculate_ssim, calculate_mse, calculate_entropy, calculate_contrast, calculate_sharpness

# Function to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Function to apply an Improved MSR (Multi-Scale Retinex) filter to an image
def improved_msr(image, scales=[3, 5, 7]):
    """
    Aplica el filtro Multi-Scale Retinex (MSR) de manera simplificada a una imagen en escala de grises.

    Parámetros:
        image: Imagen de entrada en escala de grises.
        scales: Lista de escalas para el desenfoque gaussiano (valores de sigma).

    Devuelve:
        Imagen procesada con mejora de contraste.
    """
    # Iniciar el acumulador de Retinex
    retinex = np.zeros_like(image, dtype=np.float32)

    # Aplicar desenfoque gaussiano y calcular Retinex para cada escala
    for scale in scales:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=scale)
        retinex += np.log1p(image) - np.log1p(blurred)

    # Promediar los resultados de las diferentes escalas
    retinex /= len(scales)

    # Normalizar y convertir a uint8
    return cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# Function to merge the individual R, G, B channels back into an RGB image
def merge_channels(r, g, b):
    """    Merges the three color channels (R, G, B) into an RGB image."""
    return cv2.merge([r, g, b])


def process_image(image_path):
    """
    Carga una imagen, aplica CLAHE e Improved MSR, y calcula las métricas PSNR, SSIM y MSE.

    Parameters:
        image_path: Ruta a la imagen que se va a procesar.

    Returns:
        Un diccionario con las métricas PSNR, SSIM y MSE.
    """
    # Cargar la imagen
    img = cv2.imread(image_path)

    # Convertir de BGR a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Separar los canales R, G, B
    r, g, b = cv2.split(img_rgb)

    # Aplicar CLAHE a cada canal
    r_clahe = apply_clahe(r)
    g_clahe = apply_clahe(g)
    b_clahe = apply_clahe(b)

    # Aplicar Improved MSR a cada canal
    r_msr = improved_msr(r_clahe)
    g_msr = improved_msr(g_clahe)
    b_msr = improved_msr(b_clahe)

    # Unir los canales R, G, B procesados en una imagen RGB
    processed_img = merge_channels(r_msr, g_msr, b_msr)

    # Convertir de nuevo a BGR para compararlo con la imagen original
    processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

    # Calcular las métricas PSNR, SSIM y MSE
    psnr_rgb = calculate_psnr(img, processed_img_bgr)
    ssim_rgb = calculate_ssim(img, processed_img_bgr)
    mse_rgb = calculate_mse(img, processed_img_bgr)
    entropy_gray = calculate_entropy(processed_img_bgr)
    contrast_gray = calculate_contrast(processed_img_bgr)
    sharpness_gray = calculate_sharpness(processed_img_bgr)

    # Devolver las métricas en un diccionario
    return {
        "psnr_rgb": psnr_rgb,
        "ssim_rgb": ssim_rgb,
        "mse_rgb": mse_rgb,
        "entropy_gray": entropy_gray,
        "contrast_gray": contrast_gray,
        "sharpness_gray": sharpness_gray,
    }

if __name__ == "__main__":
    folder_path = "Public BDD (Intel)/"
    results = []  # List to store results for each image

    # Iterate over all files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(folder_path, file_name)
            # Process the image and obtain results
            result = process_image(image_path)
            # Add the results to the list
            results.append(result)

    # Convert the list of results into a pandas DataFrame
    df = pd.DataFrame(results)

    # Save the results to a CSV file
    df.to_csv("image_metrics_results.csv", index=False)

    # Calculate the averages for PSNR, SSIM, and MSE
    mean_psnr_rgb = df["psnr_rgb"].mean()
    mean_ssim_rgb = df["ssim_rgb"].mean()
    mean_mse_rgb = df["mse_rgb"].mean()
    mean_entropy_gray = df["entropy_gray"].mean()
    mean_contrast_gray = df["contrast_gray"].mean()
    mean_sharpness_gray = df["sharpness_gray"].mean()

    # Display the results on the screen
    print("Mean PSNR :", mean_psnr_rgb)
    print("Mean SSIM :", mean_ssim_rgb)
    print("Mean MSE :", mean_mse_rgb)
    print("Mean Entropy :", mean_entropy_gray)
    print("Mean Contrast :", mean_contrast_gray)
    print("Mean Sharpness :", mean_sharpness_gray)
