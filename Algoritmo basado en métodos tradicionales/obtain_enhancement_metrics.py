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
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Avoid division by zero when images are identical
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate the Structural Similarity Index (SSIM) between two images."""
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1, img2, full=True)
    return score

def calculate_mse(img1, img2):
    """Calculate the Mean Squared Error (MSE) between two images."""
    return np.mean((img1 - img2) ** 2)

def calculate_entropy(img):
    """Calculate the Shannon entropy of an image (used to measure image information)."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return shannon_entropy(img)

def calculate_sharpness(img):
    """Calculate the sharpness of an image using the variance of the Laplacian operator."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    variance = laplacian.var()

    # Normalize by mean intensity to scale the sharpness
    mean_intensity = np.mean(img)
    normalized_sharpness = variance / (mean_intensity + 1e-5)  # Avoid division by zero

    return normalized_sharpness

def calculate_contrast(img):
    """Calculate the contrast of an image as the standard deviation of pixel intensities."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.std()

def calculate_1image_metrics(image):
    """Calculate entropy, sharpness, and contrast for a single image."""
    entropy = calculate_entropy(image)
    sharpness = calculate_sharpness(image)
    contrast = calculate_contrast(image)
    return round(entropy, 2), round(sharpness, 2), round(contrast, 2)

def calculate_colorfulness(image):
    """Calculate the colorfulness of the image based on RGB component statistics."""
    # Convert image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split into color channels
    R, G, B = cv2.split(image_rgb)

    # Compute the color differences
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)

    # Mean and standard deviation of the differences
    rg_mean = np.mean(rg)
    yb_mean = np.mean(yb)
    rg_std = np.std(rg)
    yb_std = np.std(yb)

    # Combine mean and std to estimate colorfulness
    colorfulness = np.sqrt(rg_mean ** 2 + yb_mean ** 2) + 0.3 * np.sqrt(rg_std ** 2 + yb_std ** 2)
    return colorfulness

def calculate_metrics(original_img, modified_img):
    """Calculate PSNR, SSIM and MSE between an original and a modified image."""
    psnr = calculate_psnr(original_img, modified_img)
    ssim_value = calculate_ssim(original_img, modified_img)
    mse = calculate_mse(original_img, modified_img)
    return round(psnr, 2), round(ssim_value, 2), round(mse, 2)

# Function to calculate averages of metrics

def calculate_averages(results):
    """Compute the average of each enhancement metric from a list of result dictionaries."""
    num_images = len(results)
    return {
        "average_psnr": sum(r["psnr_value"] for r in results) / num_images,
        "average_ssim": sum(r["ssim_value"] for r in results) / num_images,
        "average_mse": sum(r["mse_value"] for r in results) / num_images,
        "average_entropy": sum(r["entropy_gray"] for r in results) / num_images,
        "average_contrast": sum(r["contrast_gray"] for r in results) / num_images,
        "average_sharpness": sum(r["sharpness_gray"] for r in results) / num_images,
        "average_colorfulness": sum(r["colorfulness_gray"] for r in results) / num_images,
    }

# File and plotting utilities

def generate_unique_csv_name(output_csv):
    """Ensure that the CSV file name is unique by appending a counter if needed."""
    base, ext = os.path.splitext(output_csv)
    counter = 1
    while os.path.exists(output_csv):
        output_csv = f"{base}_{counter}{ext}"
        counter += 1
    return output_csv

def save_results_to_csv(results, output_csv):
    """Save the list of results to a CSV file with a unique name."""
    output_csv = generate_unique_csv_name(output_csv)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
