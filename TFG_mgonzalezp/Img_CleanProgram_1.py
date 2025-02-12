import numpy as np
import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from obtain_enhancement_metrics import (
    calculate_1image_metrics,
    calculate_metrics,
    save_results_to_csv,
)

def resize_image(image, min_size=1024):
    """Redimensiona la imagen manteniendo la relación de aspecto para que el lado más corto sea min_size."""
    h, w = image.shape[:2]
    scale = min_size / min(h, w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def apply_median_filter(image, ksize=3):
    """Aplica un filtro de mediana para suavizar la imagen."""
    return cv2.medianBlur(image, ksize)

def clahe_lab(image):
    """Mejora el contraste de una imagen BGR usando CLAHE en el espacio de color LAB."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def remove_shadows(image, kernel_size=(5, 5), alpha=1.1, beta=0):
    """
    Elimina las sombras de una imagen ajustando el brillo y el contraste.

    Parámetros:
    - image: Imagen de entrada en formato BGR.
    - kernel_size: Tamaño del kernel para el filtro de paso alto.
    - alpha: Factor de contraste.
    - beta: Factor de brillo.

    Retorna:
    - Imagen con sombras reducidas.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, kernel_size, 0)
    high_pass = cv2.subtract(gray, blurred)
    high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)
    high_pass_bgr = cv2.cvtColor(high_pass, cv2.COLOR_GRAY2BGR)
    enhanced = cv2.addWeighted(image, alpha, high_pass_bgr, beta, 0)
    return enhanced


def apply_sharpening(image):
    """Agrega nitidez a la imagen utilizando un filtro de convolución."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Función de balance de blancos usando el algoritmo Gray World

def image_enhancement_pipeline(image_path, save_folder):
    """Procesa una imagen y calcula métricas de mejora."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se pudo cargar la imagen desde {image_path}")
            return None

        # Redimensionar la imagen manteniendo la relación de aspecto
        image = resize_image(image)

        # Aplicar las técnicas de mejora de la imagen
        denoised = apply_median_filter(image)
        clahe = clahe_lab(denoised)
        shadows_removed = remove_shadows(clahe)
        sharpened = apply_sharpening(shadows_removed)
        normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

        # Guardar la imagen mejorada
        filename = os.path.basename(image_path)
        save_path = os.path.join(save_folder, f"normalized_{filename.rsplit('.', 1)[0]}.png")
        cv2.imwrite(save_path, normalized)

        # Calcular métricas
        psnr, ssim, mse = calculate_metrics(image, normalized)
        entropy, sharpness, contrast = calculate_1image_metrics(normalized)

        return {
            "image_name": filename,
            "psnr_value": psnr,
            "ssim_value": ssim,
            "mse_value": mse,
            "entropy_gray": entropy,
            "contrast_gray": contrast,
            "sharpness_gray": sharpness,
        }
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None

def process_images_in_folder(folder_path, save_folder):
    """Procesa todas las imágenes de una carpeta y devuelve las rutas de las imágenes."""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    image_paths = [
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ]
    return image_paths

def process_images_in_parallel(image_paths, save_folder):
    """Procesa las imágenes de manera concurrente utilizando ThreadPoolExecutor."""
    results = []
    start_time = time.time()  # Inicia el temporizador para el cálculo de FPS
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(image_enhancement_pipeline, image_path, save_folder): image_path for image_path in
                   image_paths}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error procesando {futures[future]}: {e}")
    end_time = time.time()  # Finaliza el temporizador para el cálculo de FPS

    # Calcular FPS
    total_time = end_time - start_time
    fps = len(image_paths) / total_time if total_time > 0 else 0
    print(f"Tiempo total de procesamiento: {total_time:.2f} segundos")

    # Verificar si FPS cumple con el umbral requerido
    if fps >= 12:
        print(f"El sistema soporta {fps:.2f} FPS, cumpliendo el requisito de 12 FPS.")
    else:
        print(f"El sistema soporta {fps:.2f} FPS, lo cual está por debajo del requisito de 12 FPS.")

    return results

def calculate_stats(results):
    """Calcula la media y desviación estándar para cada métrica."""
    metrics = {
        "psnr_value": [],
        "ssim_value": [],
        "mse_value": [],
        "entropy_gray": [],
        "contrast_gray": [],
        "sharpness_gray": [],
    }

    for result in results:
        if result:
            for key in metrics:
                metrics[key].append(result[key])

    stats = {
        metric: {
            "mean": np.mean(values),
            "std_dev": np.std(values),
        }
        for metric, values in metrics.items()
    }
    return stats

if __name__ == "__main__":
    folder_path = "Public BDD (Intel)/Imagenes/"
    save_folder = "Image_enhancement/Optuna2/"
    output_csv = "Metricas/P2/Optuna_prueba.csv"

    # Procesar las imágenes
    image_paths = process_images_in_folder(folder_path, save_folder)
    results = process_images_in_parallel(image_paths, save_folder)

    # Calcular estadísticas y guardar resultados
    if results:
        stats = calculate_stats(results)
        for metric, values in stats.items():
            print(f"{metric} - Mean: {values['mean']:.2f}, Std Dev: {values['std_dev']:.2f}")
        save_results_to_csv(results, output_csv)
        print(f"Resultados guardados en {output_csv}")
    else:
        print("No se procesaron imágenes válidas.")
