import optuna
import cv2
import numpy as np
import os

from joblib import Parallel, delayed
from Img_CleanProgram_1 import resize_image, apply_median_filter, clahe_lab, apply_sharpening, remove_shadows
from obtain_enhancement_metrics import calculate_psnr, calculate_ssim, calculate_mse


# Función para calcular las métricas
def calculate_metrics(input_image, target_image):
    psnr_value = calculate_psnr(input_image, target_image)
    ssim_value = calculate_ssim(input_image, target_image)
    mae_value = calculate_mse(input_image, target_image)
    return psnr_value, ssim_value, mae_value


# Función de preprocesado de imagen
def preprocess_image(image, ksize, tilegridsize, cliplimit, alpha, beta):
    #img_resize=resize_image(image)
    img_blurred = apply_median_filter(image, (ksize))  # Aplica un filtro gaussiano
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilegridsize, tilegridsize))  # Corregir mayúsculas
    img_clahe = clahe.apply(img_blurred)  # Aplica CLAHE
    img_adjusted = cv2.convertScaleAbs(img_clahe, alpha=alpha, beta=beta)  # Ajuste de brillo/contraste
    img_sharpened = apply_sharpening(img_adjusted)
    normalized = cv2.normalize(img_sharpened, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


# Función para procesar una sola imagen
def process_image(img_path, ksize, tilegridsize, cliplimit, alpha, beta):
    input_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    target_image = cv2.imread(img_path.replace('input', 'target'), cv2.IMREAD_GRAYSCALE)

    if input_image is None or target_image is None:
        raise ValueError(f"Error al cargar las imágenes: {img_path}")

    processed_image = preprocess_image(input_image, ksize, tilegridsize, cliplimit, alpha, beta)
    return calculate_metrics(processed_image, target_image)


# Función objetivo para Optuna
def objective(trial):
    ksize = trial.suggest_int("ksize", 3, 15, step=2)
    tilegridsize = trial.suggest_int("tilegridsize", 8, 32, step=2)
    cliplimit = trial.suggest_float("cliplimit", 1.0, 5.0)
    alpha = trial.suggest_float("alpha", 0.0, 2.0)
    beta = trial.suggest_float("beta", 0.0, 100.0)

    # Lista de rutas de imágenes en la carpeta
    image_folder = 'Public BDD (Intel)/'  # Cambia esta ruta por la de tu carpeta
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.jpg')]

    # Procesar las imágenes en paralelo
    results = Parallel(n_jobs=-1)(
        delayed(process_image)(img_path, ksize, tilegridsize, cliplimit, alpha, beta) for img_path in image_paths)

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
study.optimize(objective, n_trials=100)

# Mostrar los mejores parámetros encontrados
for trial in study.best_trials:
    print("PSNR:", -trial.values[0], "SSIM:", -trial.values[1], "MAE:", trial.values[2])
    print("Best Parameters:", trial.params)