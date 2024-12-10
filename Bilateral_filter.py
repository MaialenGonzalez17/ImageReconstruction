import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Importar las funciones de cálculo de PSNR, SSIM, MSE, Entropía, Contraste y Nitidez
from obtain_enhancement_metrics import calculate_psnr, calculate_ssim, calculate_mse, calculate_entropy, \
    calculate_contrast, calculate_sharpness


# Función para calcular y devolver las métricas PSNR, SSIM, MSE, Entropía, Contraste y Nitidez
def calculate_metrics(original_img, filtered_img):
    # Calcular las métricas PSNR, SSIM y MSE entre la imagen original y la filtrada
    psnr_rgb = calculate_psnr(original_img, filtered_img)
    ssim_rgb = calculate_ssim(original_img, filtered_img)
    mse_rgb = calculate_mse(original_img, filtered_img)

    # Convertir a escala de grises para calcular métricas específicas de imágenes en escala de grises
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    filtered_gray = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)

    entropy_gray = calculate_entropy(filtered_gray)
    contrast_gray = calculate_contrast(filtered_gray)
    sharpness_gray = calculate_sharpness(filtered_gray)

    # Devolver las métricas en un diccionario
    return {
        "psnr_rgb": psnr_rgb,
        "ssim_rgb": ssim_rgb,
        "mse_rgb": mse_rgb,
        "entropy_gray": entropy_gray,
        "contrast_gray": contrast_gray,
        "sharpness_gray": sharpness_gray,
    }

def process_image(image_path):
    # Leer la imagen usando OpenCV
    img = cv2.imread(image_path)

    # Verificar si la imagen se ha cargado correctamente
    if img is None:
        print("Error: No se pudo cargar la imagen.")
        return None

    # Convertir la imagen de BGR (por defecto en OpenCV) a RGB (para visualización)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Aplicar el filtro bilateral con d = 15, sigmaColor = sigmaSpace = 75
    bilateral = cv2.bilateralFilter(img, 15, 75, 75)

    # Convertir la imagen filtrada a RGB para una visualización coherente
    bilateral_rgb = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)

    # Calcular las métricas PSNR, SSIM y MSE
    psnr_rgb = calculate_psnr(img_rgb, bilateral_rgb)
    ssim_rgb = calculate_ssim(img_rgb, bilateral_rgb)
    mse_rgb = calculate_mse(img_rgb, bilateral_rgb)
    entropy_gray = calculate_entropy(bilateral_rgb)
    contrast_gray = calculate_contrast(bilateral_rgb)
    sharpness_gray = calculate_sharpness(bilateral_rgb)

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
