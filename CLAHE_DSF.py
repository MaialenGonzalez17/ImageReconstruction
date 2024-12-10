import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Importar las funciones de cálculo de PSNR, SSIM, MSE, Entropía, Contraste y Nitidez
from obtain_enhancement_metrics import calculate_psnr, calculate_ssim, calculate_mse, calculate_entropy, \
    calculate_contrast, calculate_sharpness


# Función para aplicar el DSF (Diffusion Stopping Function)
def diffusion_stopping_function(image, iterations=10, kappa=30):
    img_float = np.float32(image)
    for _ in range(iterations):
        grad_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        diffusion = 1.0 / (1.0 + (grad_magnitude / kappa) ** 2)
        img_float += diffusion * (cv2.Laplacian(img_float, cv2.CV_32F))
    return np.uint8(np.clip(img_float, 0, 255))

# Función para aplicar CLAHE
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])  # Solo aplicar CLAHE en el canal de luminancia (Y)
    img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_bgr


# Función para procesar la imagen
def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Error: No se pudo cargar la imagen {image_path}")

    # Convertir la imagen a escala de grises para el procesamiento de DSF
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar CLAHE (en el espacio de color YUV para mayor efectividad en el canal de luminancia)
    clahe_img = apply_clahe(img)  # Asegúrate de que `apply_clahe` ya realiza correctamente la conversión de BGR a YUV

    # Aplicar la función de detención de difusión (DSF)
    dsf_img = diffusion_stopping_function(img_gray)

    # Convertir la imagen DSF de vuelta a BGR
    dsf_img_bgr = cv2.cvtColor(dsf_img, cv2.COLOR_GRAY2BGR)

    # Si la imagen CLAHE ya es en escala de grises, no la conviertas a BGR. Usaremos directamente CLAHE en escala de grises.
    # Asegurarse de que ambas imágenes tengan el mismo tamaño
    if clahe_img.shape != dsf_img_bgr.shape:
        clahe_img = cv2.resize(clahe_img, (dsf_img_bgr.shape[1], dsf_img_bgr.shape[0]))

    # Combinar las imágenes procesadas
    combined_img = cv2.addWeighted(clahe_img, 0.5, dsf_img_bgr, 0.5, 0)

    # Calcular las métricas de calidad
    psnr_rgb = calculate_psnr(img, combined_img)
    ssim_rgb = calculate_ssim(img, combined_img)
    mse_rgb = calculate_mse(img, combined_img)
    entropy_gray = calculate_entropy(combined_img)
    contrast_gray = calculate_contrast(combined_img)
    sharpness_gray = calculate_sharpness(combined_img)

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
