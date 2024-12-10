import cv2
import numpy as np
import pandas as pd
import os

# Import PSNR, SSIM, MSE, entropy, contrast, and sharpness calculation functions
from obtain_enhancement_metrics import calculate_psnr, calculate_ssim, calculate_mse, calculate_entropy, calculate_contrast, calculate_sharpness

# Function to load the image in grayscale
def load_image_grayscale(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Function to load the image in RGB
def load_image_rgb(image_path):
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Function to apply anisotropic diffusion to each channel of an RGB image
""" La idea principal es suavizar más las áreas donde no hay mucha variación (zonas lisas) y suavizar menos donde hay cambios abruptos en la intensidad de los píxeles (los bordes). 
Esto ayuda a reducir el ruido mientras preserva las características importantes de la imagen."""

""" La función anisodiff_rgb toma una imagen RGB de entrada, comprueba si tiene 3 dimensiones (lo que indica que es una imagen RGB), y luego aplica la función anisodiff a cada canal 
(R, G y B) de la imagen por separado."""

def anisodiff_rgb(img, niter=1, kappa=50, gamma=0.1, step=(1., 1.), option=1):
    """ Aplica difusión anisotrópica a una imagen RGB procesando cada canal independientemente."""

    if img.ndim == 3:  # Verifica si la imagen tiene 3 dimensiones (es una imagen RGB)
        imgout = np.zeros_like(img, dtype='float32')  # Crea una imagen de salida vacía de tipo float32

        for i in range(3):  # Procesa cada uno de los tres canales (Rojo, Verde, Azul)
            imgout[:, :, i] = anisodiff(img[:, :, i], niter, kappa, gamma, step, option)

        return np.uint8(
            imgout)  # Convierte la imagen de salida a tipo uint8 (para guardar la imagen como imagen normal)
    else:
        raise ValueError("La imagen de entrada debe ser una imagen RGB de 3 dimensiones.")


def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1., 1.), option=1):
    """ Aplica difusión anisotrópica a una imagen en escala de grises."""

    img = img.astype('float32')  # Convierte la imagen a tipo float32 para los cálculos
    imgout = img.copy()  # Copia de la imagen original, que se actualizará

    for _ in range(niter):  # Aplica la difusión durante el número de iteraciones especificado
        # Calcula las diferencias (gradientes) en las direcciones de fila y columna
        deltaS = np.diff(imgout, axis=0, append=0)  # Diferencia en la dirección vertical (de arriba a abajo)
        deltaE = np.diff(imgout, axis=1, append=0)  # Diferencia en la dirección horizontal (de izquierda a derecha)

        # Calcula la "conductancia" (qué tanto difundir según el gradiente)
        if option == 1:
            # Expone un conducto más grande cuando el gradiente es pequeño (opción 1)
            gS = np.exp(-(deltaS / kappa) ** 2.) / step[0]  # Conductancia en la dirección S (Sur)
            gE = np.exp(-(deltaE / kappa) ** 2.) / step[1]  # Conductancia en la dirección E (Este)
        else:  # option == 2
            # Conduce menos en las áreas con gradientes grandes (opción 2)
            gS = 1. / (1 + (deltaS / kappa) ** 2.) / step[0]  # Conductancia en la dirección S (Sur)
            gE = 1. / (1 + (deltaE / kappa) ** 2.) / step[1]  # Conductancia en la dirección E (Este)

        # Actualiza la imagen con la suma de las variaciones controladas por la conductancia
        imgout += gamma * (np.diff(gS * deltaS, axis=0, prepend=0) + np.diff(gE * deltaE, axis=1, prepend=0))

    return imgout  # Devuelve la imagen modificada

# Main function to load, process the RGB image, and calculate metrics
def process_image(image_path, niter=1):
    """ Processes the image by loading it in grayscale and color, applying anisotropic diffusion,
    and calculates and returns various metrics. """
    # Load the images in grayscale and color
    original_gray = load_image_grayscale(image_path)
    original_rgb = load_image_rgb(image_path)

    # Apply anisotropic diffusion
    filtered_rgb = anisodiff_rgb(original_rgb, niter=niter)

    # Calculate PSNR, SSIM, and MSE for RGB images
    psnr_rgb = calculate_psnr(original_rgb, filtered_rgb)
    ssim_rgb = calculate_ssim(original_rgb, filtered_rgb)
    mse_rgb = calculate_mse(original_rgb, filtered_rgb)
    entropy_gray = calculate_entropy(filtered_rgb)
    contrast_gray = calculate_contrast(filtered_rgb)
    sharpness_gray = calculate_sharpness(filtered_rgb)

    return {
        "image": image_path,
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
            result = process_image(image_path, niter=5)
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
