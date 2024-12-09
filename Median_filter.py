import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import structural_similarity as ssim

# Import PSNR, SSIM, and MSE calculation functions
from obtain_enhancement_metrics import calculate_psnr, calculate_ssim, calculate_mse

# Function to load the image in grayscale
def load_image_grayscale(image_path):
    """
    Loads an image from the specified path and converts it to grayscale.

    Parameters:
    - image_path: Path to the image.

    Returns:
    - Grayscale image as a numpy array.
    """
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Function to load the image in RGB color
def load_image_rgb(image_path):
    """
    Loads an image from the specified path and converts it to RGB format.

    Parameters:
    - image_path: Path to the image.

    Returns:
    - RGB image as a numpy array.
    """
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to apply median filter
def apply_median_filter(image, kernel_size=5):
    """
    Applies the median filter to the image with a specific kernel size.

    Parameters:
    - image: Input image as a numpy array.
    - kernel_size: Kernel size for the median filter (must be an odd number).

    Returns:
    - Image filtered with the median filter.
    """
    return cv2.medianBlur(image, kernel_size)

# Function to display original and filtered images (grayscale and color)
def show_images(original_gray, filtered_gray, original_rgb, filtered_rgb):
    """
    Displays the original and filtered images both in grayscale and RGB.

    Parameters:
    - original_gray: Original grayscale image.
    - filtered_gray: Filtered grayscale image.
    - original_rgb: Original RGB image.
    - filtered_rgb: Filtered RGB image.
    """
    plt.figure(figsize=(15, 10))

    # Show grayscale images
    plt.subplot(2, 2, 1)
    plt.title("Original Grayscale Image")
    plt.imshow(original_gray, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Median Filtered Grayscale Image")
    plt.imshow(filtered_gray, cmap='gray')
    plt.axis('off')

    # Show RGB images
    plt.subplot(2, 2, 3)
    plt.title("Original RGB Image")
    plt.imshow(original_rgb)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Median Filtered RGB Image")
    plt.imshow(filtered_rgb)
    plt.axis('off')

    plt.show()

# Main function to process the image
def process_image(image_path, kernel_size=5):
    """
    Processes the image by loading it in grayscale and color, applying the median filter,
    and displaying the results. Also calculates and prints the PSNR, SSIM, and MSE values.

    Parameters:
    - image_path: Path to the image.
    - kernel_size: Kernel size for the median filter (optional, default is 5).
    """
    # Load the images in grayscale and color
    original_gray = load_image_grayscale(image_path)
    original_rgb = load_image_rgb(image_path)

    # Apply the median filter to both versions of the image
    filtered_gray = apply_median_filter(original_gray, kernel_size)
    filtered_rgb = apply_median_filter(original_rgb, kernel_size)

    # Calculate PSNR, SSIM, and MSE for grayscale and RGB images
    psnr_gray = calculate_psnr(original_gray, filtered_gray)
    psnr_rgb = calculate_psnr(original_rgb, filtered_rgb)

    ssim_gray = calculate_ssim_safe(original_gray, filtered_gray)
    ssim_rgb = calculate_ssim_safe(original_rgb, filtered_rgb)

    mse_gray = calculate_mse(original_gray, filtered_gray)
    mse_rgb = calculate_mse(original_rgb, filtered_rgb)

    # Print PSNR, SSIM, and MSE values
    print(f"PSNR for Grayscale Image: {psnr_gray:.2f} dB")
    # print(f"PSNR for RGB Image: {psnr_rgb:.2f} dB")
    print(f"SSIM for Grayscale Image: {ssim_gray:.4f}")
    # print(f"SSIM for RGB Image: {ssim_rgb:.4f}")
    print(f"MSE for Grayscale Image: {mse_gray:.2f}")
    # print(f"MSE for RGB Image: {mse_rgb:.2f}")

    # Display the images
    show_images(original_gray, filtered_gray, original_rgb, filtered_rgb)

# Function to safely calculate SSIM
def calculate_ssim_safe(img1, img2):
    """
    Safely calculates the SSIM between two images, ensuring the correct number of channels.

    Parameters:
    - img1: First image.
    - img2: Second image.

    Returns:
    - SSIM value.
    """
    # Check if img1 and img2 are in color (3 channels) and convert to grayscale if necessary
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    return ssim(img1, img2)
if __name__ == "__main__":
    # Execute the process with an image
    image_path = "Public BDD (Intel)/0.jpg"
    process_image(image_path, kernel_size=5)
