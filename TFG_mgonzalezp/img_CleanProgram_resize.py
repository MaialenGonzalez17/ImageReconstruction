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

def resize_image(image, min_side=1024):
    """Resize image maintaining aspect ratio so the shortest side is 1024 pixels."""
    h, w = image.shape[:2]
    scale = min_side / min(h, w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def apply_median_filter(image, ksize=3):
    """Apply Median filter to smooth the image."""
    return cv2.medianBlur(image, ksize)


def clahe_lab(image):
    """Enhances the contrast of a BGR image using CLAHE in the LAB color space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.012, tileGridSize=(26, 26))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


#def cv2_white_balance(image):
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #h, s, v = cv2.split(hsv)
    #s = cv2.convertScaleAbs(s, alpha=(np.mean(h) / np.mean(s)), beta=0)
    #v = cv2.convertScaleAbs(v, alpha=(np.mean(h) / np.mean(v)), beta=0)
    #return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def apply_sharpening(image):
    """Sharpens the image using a convolution kernel."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def reduce_saturation(image, scale=0.9):
    """Reduces the saturation of an image by a specified scale."""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * scale, 0, 255).astype(np.uint8)
        hsv_saturated = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv_saturated, cv2.COLOR_HSV2BGR)
    except Exception as e:
        print(f"Error in reducing saturation: {e}")
        return image


def image_enhancement_pipeline(image_path, save_folder):
    """Processes an image and calculates metrics."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Couldn't load the image from {image_path}")
        return None

    image = resize_image(image)
    denoised = apply_median_filter(image)
    contrast_corrected = clahe_lab(denoised)
    #color_corrected = cv2_white_balance(contrast_corrected)
    reduced_saturation = reduce_saturation(contrast_corrected, scale=0.9)
    sharpened = apply_sharpening(reduced_saturation)
    normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

    filename = os.path.basename(image_path)
    save_path = os.path.join(save_folder, f"normalized_{filename.rsplit('.', 1)[0]}.png")
    cv2.imwrite(save_path, normalized)

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


def process_images_in_folder(folder_path, save_folder):
    """Processes all images in a folder and returns image paths."""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    image_paths = [
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ]

    return image_paths

# Parallelized processing function to handle multiple images simultaneously
def process_images_in_parallel(image_paths, save_folder):
    """Process images concurrently using ThreadPoolExecutor."""
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(image_enhancement_pipeline, image_path, save_folder): image_path for image_path in
                   image_paths}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")
    return results

def calculate_stats(results):
    """Calculates mean and standard deviation for each metric."""
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
    save_folder = "image_enhacement/Optuna/"
    output_csv = "Metricas/Optuna/Metricas_Optuna.csv"

    start_time = time.time()

    image_paths = process_images_in_folder(folder_path, save_folder)
    results = process_images_in_parallel(image_paths, save_folder)

    if results:
        stats = calculate_stats(results)
        for metric, values in stats.items():
            print(f"{metric} - Mean: {values['mean']:.2f}, Std Dev: {values['std_dev']:.2f}")
        save_results_to_csv(results, output_csv)
        print(f"Results saved to {output_csv}")
    else:
        print("No valid images were processed.")

    print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")
