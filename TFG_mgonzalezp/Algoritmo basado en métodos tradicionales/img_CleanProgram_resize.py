import numpy as np
import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from obtain_enhancement_metrics import (
    calculate_1image_metrics,
    calculate_metrics,
    save_results_to_csv,
    calculate_colorfulness
)


def resize_image(image, target_size=(1024, 1365)):
    """Resize the input image to the target dimensions using area interpolation.
    This helps standardize all images before further processing."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def apply_median_filter(image, ksize=3):
    """Apply a median filter to the image to reduce salt-and-pepper noise.
    This operation preserves edges while removing small pixel-level artifacts."""
    return cv2.medianBlur(image, ksize)


def clahe_lab(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel
    of the LAB color space to improve local contrast without amplifying noise."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.012, tileGridSize=(26, 26))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)


def apply_sharpening(image):
    """Sharpen the image by applying a convolution with a sharpening kernel.
    This enhances edges and fine details."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def reduce_saturation(image, scale=0.9):
    """Reduce the image saturation in the HSV color space by a specified factor.
    Useful to slightly desaturate overly vibrant colors for a more natural appearance."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * scale, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def image_enhancement_pipeline(image_path, save_folder):
    """Enhance a single image through resizing, denoising, contrast adjustment, desaturation,
    sharpening, and normalization. Then calculate various quality metrics.
    The final enhanced image is saved to disk."""

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Couldn't load the image from {image_path}")
        return None

    image = resize_image(image)
    denoised = apply_median_filter(image)
    contrast_corrected = clahe_lab(denoised)
    reduced_saturation = reduce_saturation(contrast_corrected, scale=0.9)
    sharpened = apply_sharpening(reduced_saturation)
    normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

    filename = os.path.basename(image_path)
    save_path = os.path.join(save_folder, f"{os.path.splitext(filename)[0]}.png")
    cv2.imwrite(save_path, normalized)

    psnr, ssim, mse = calculate_metrics(image, normalized)
    entropy, sharpness, contrast = calculate_1image_metrics(normalized)
    colorfulness = calculate_colorfulness(normalized)

    return {
        "image_name": filename,
        "psnr_value": psnr,
        "ssim_value": ssim,
        "mse_value": mse,
        "entropy_gray": entropy,
        "contrast_gray": contrast,
        "sharpness_gray": sharpness,
        "colorfulness": colorfulness,
    }


def process_images_in_folder(folder_path, save_folder):
    """Create the output folder if it doesn't exist, and return a list of image paths
    (only valid image formats) from the specified input folder."""
    os.makedirs(save_folder, exist_ok=True)
    return [
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ]


def process_images_in_parallel(image_paths, save_folder, max_workers=8):
    """Process multiple images in parallel using threading to speed up the pipeline.
    Each image is enhanced and its metrics are computed and stored."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(image_enhancement_pipeline, img, save_folder): img for img in image_paths}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")
    return results


def calculate_stats(results):
    """Given a list of results, compute the mean and standard deviation
    for each enhancement metric across all images."""
    if not results:
        return {}

    metrics = {key: np.array([res[key] for res in results], dtype=np.float32) for key in results[0] if
               key != "image_name"}

    return {metric: {"mean": np.mean(values), "std_dev": np.std(values)} for metric, values in metrics.items()}


if __name__ == "__main__":
    """Main entry point: defines folder paths, launches the enhancement and evaluation process,
    and saves the resulting metrics in a CSV file."""

    folder_path = "enter_here_your_input_folder_path"  # Example: "path/to/your/input/images"
    save_folder = "enter_here_your_output_folder_path"  # Example: "path/to/save/enhanced/images"
    output_csv = "enter_here_your_output_csv_path"  # Example: "path/to/save/results.csv"

    start_time = time.time()

    image_paths = process_images_in_folder(folder_path, save_folder)
    results = process_images_in_parallel(image_paths, save_folder, max_workers=8)

    total_time = time.time() - start_time
    num_images = len(results)
    avg_time_per_image = total_time / num_images if num_images > 0 else 0

    if results:
        stats = calculate_stats(results)
        for metric, values in stats.items():
            print(f"{metric} - Mean: {values['mean']:.2f}, Std Dev: {values['std_dev']:.2f}")
        save_results_to_csv(results, output_csv)
        print(f"Results saved to {output_csv}")
    else:
        print("No valid images were processed.")

    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Average Time per Image: {avg_time_per_image:.2f} seconds")
