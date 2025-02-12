import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor

# Metric calculations
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')  # Avoid division by zero
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1, img2, full=True)
    return score

def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def calculate_entropy(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return shannon_entropy(img)

def calculate_sharpness(img):
    """Calculates the normalized sharpness of an image using the Laplacian operator."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    variance = laplacian.var()

    # Normalize sharpness by the mean intensity of the image
    mean_intensity = np.mean(img)
    normalized_sharpness = variance / (mean_intensity + 1e-5)  # Avoid division by zero

    return normalized_sharpness


def calculate_contrast(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.std()

def calculate_1image_metrics(image):
    entropy = calculate_entropy(image)
    sharpness = calculate_sharpness(image)
    contrast = calculate_contrast(image)
    return round(entropy, 2), round(sharpness, 2), round(contrast, 2)

def calculate_metrics(original_img, modified_img):
    psnr = calculate_psnr(original_img, modified_img)
    ssim_value = calculate_ssim(original_img, modified_img)
    mse = calculate_mse(original_img, modified_img)
    return round(psnr, 2), round(ssim_value, 2), round(mse, 2)

# Function to calculate averages of metrics
def calculate_averages(results):
    num_images = len(results)
    return {
        "average_psnr": sum(r["psnr_value"] for r in results) / num_images,
        "average_ssim": sum(r["ssim_value"] for r in results) / num_images,
        "average_mse": sum(r["mse_value"] for r in results) / num_images,
        "average_entropy": sum(r["entropy_gray"] for r in results) / num_images,
        "average_contrast": sum(r["contrast_gray"] for r in results) / num_images,
        "average_sharpness": sum(r["sharpness_gray"] for r in results) / num_images,
    }

# File and plotting utilities
def generate_unique_csv_name(output_csv):
    base, ext = os.path.splitext(output_csv)
    counter = 1
    while os.path.exists(output_csv):
        output_csv = f"{base}_{counter}{ext}"
        counter += 1
    return output_csv

def save_results_to_csv(results, output_csv):
    output_csv = generate_unique_csv_name(output_csv)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
