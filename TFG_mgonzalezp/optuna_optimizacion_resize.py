import optuna
import cv2
import numpy as np
import os
from joblib import Parallel, delayed
from obtain_enhancement_metrics import calculate_psnr, calculate_ssim, calculate_mse

from img_CleanProgram_resize import (
    resize_image,
    apply_median_filter,
    apply_sharpening,
    reduce_saturation,
)

# Función para calcular las métricas
def calculate_metrics(input_image, target_image):
    # Convertir ambas imágenes a escala de grises si no lo están
    if input_image.ndim == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    if target_image.ndim == 3:
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    psnr_value = calculate_psnr(input_image, target_image)
    ssim_value = calculate_ssim(input_image, target_image)
    mae_value = calculate_mse(input_image, target_image)
    return psnr_value, ssim_value, mae_value


def preprocess_image(image, ksize, tilegridsize, cliplimit, scale):
    img_blurred = apply_median_filter(image, ksize)  # Aplica un filtro gaussiano
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilegridsize, tilegridsize))  # Corregir mayúsculas
    img_clahe = clahe.apply(img_blurred)  # Aplica CLAHE

    # Convertir la imagen a RGB si es de un solo canal
    if len(img_clahe.shape) == 2:
        img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)

    img_sharpened = apply_sharpening(img_clahe)
    saturation = reduce_saturation(img_sharpened, scale=scale)
    normalized = cv2.normalize(saturation, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


def process_image(img_path, ksize, tilegridsize, cliplimit, scale):
    input_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    target_image = cv2.imread(img_path.replace('input', 'target'), cv2.IMREAD_GRAYSCALE)

    if input_image is None or target_image is None:
        raise ValueError(f"Error al cargar las imágenes: {img_path}")

    processed_image = preprocess_image(input_image, ksize, tilegridsize, cliplimit, scale)
    return calculate_metrics(processed_image, target_image)


def objective(trial):
    ksize = trial.suggest_int("ksize", 3, 15, step=2)
    tilegridsize = trial.suggest_int("tilegridsize", 8, 32, step=2)
    cliplimit = trial.suggest_float("cliplimit", 1.0, 5.0)
    scale = trial.suggest_float("scale", 0.0, 2.0)

    # Lista de rutas de imágenes en la carpeta
    image_folder = 'Public BDD (Intel)/'  # Cambia esta ruta por la de tu carpeta
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.jpg')]

    # Procesar las imágenes en paralelo
    results = Parallel(n_jobs=-1)(
        delayed(process_image)(img_path, ksize, tilegridsize, cliplimit, scale) for img_path in image_paths)

    # Extraer los resultados
    psnr_values, ssim_values, mae_values = zip(*results)

    # Promediar las métricas
    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)
    mean_mae = np.mean(mae_values)

    # Optuna maximiza PSNR y SSIM, y minimiza MAE
    return -mean_psnr, -mean_ssim, mean_mae


# Ejecutar la optimización
study = optuna.create_study(directions=["maximize", "maximize", "minimize"])  # Maximizar PSNR y SSIM, minimizar MAE
study.optimize(objective, n_trials=20)

# Mostrar los tres mejores parámetros encontrados
best_trials = sorted(study.best_trials, key=lambda t: (t.values[0], t.values[1], -t.values[2]))[:3]
for trial in best_trials:
    print("PSNR:", -trial.values[0], "SSIM:", -trial.values[1], "MAE:", trial.values[2])
    print("Best Parameters:", trial.params)
