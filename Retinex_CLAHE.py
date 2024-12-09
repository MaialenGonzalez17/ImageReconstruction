import cv2
import numpy as np
import matplotlib.pyplot as plt
from obtain_enhancement_metrics import calculate_psnr, calculate_ssim, calculate_mse

# Function to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    """
    Applies CLAHE to a grayscale image.

    image: Grayscale image.
    """
    # Create CLAHE object with clip limit of 3.0 and tile size of 8x8
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Function to apply an Improved MSR (Multi-Scale Retinex) filter to an image
def improved_msr(image):
    """
    Applies the Improved MSR (Multi-Scale Retinex) filter to an image.

    image: Input image (single color channel).
    """
    # Define different scale sizes for the MSR filter
    scales = [3, 5, 7]
    output = np.zeros_like(image, dtype=np.float32)  # Initialize the output as a float array

    # Loop over each scale and apply Gaussian blur for each scale
    for scale in scales:
        blurred = cv2.GaussianBlur(image, (scale, scale), 0)  # Apply Gaussian blur
        # Compute the MSR output for this scale
        output += np.log(image + 1.0) - np.log(blurred + 1.0)

    # Exponentiate the result and clip the output to be between 0 and 255
    output = np.exp(output) - 1.0
    return np.clip(output, 0, 255).astype(np.uint8)

# Function to merge the individual R, G, B channels back into an RGB image
def merge_channels(r, g, b):
    """
    Merges the three color channels (R, G, B) into an RGB image.

    r: Red channel.
    g: Green channel.
    b: Blue channel.
    """
    return cv2.merge([r, g, b])

# Function to calculate PSNR, SSIM, and MSE between the original and processed images
def display_metrics(original_img, processed_img):
    # Calculate PSNR, SSIM, and MSE
    psnr_value = calculate_psnr(original_img, processed_img)
    ssim_value = calculate_ssim(original_img, processed_img)
    mse_value = calculate_mse(original_img, processed_img)

    # Print the results
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"MSE: {mse_value:.2f}")

# Load the input image
image_path = "Public BDD (Intel)/0.jpg"  # Path to the image file
img = cv2.imread(image_path)  # Read the image using OpenCV

# Convert from BGR (OpenCV default) to RGB for processing
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Split the RGB image into individual channels
r, g, b = cv2.split(img_rgb)

# Apply CLAHE to the individual color channels
r_clahe = apply_clahe(r)  # CLAHE applied to the Red channel
g_clahe = apply_clahe(g)  # CLAHE applied to the Green channel
b_clahe = apply_clahe(b)  # CLAHE applied to the Blue channel

# Merge the processed channels into an RGB image
rgb_image = cv2.merge([r_clahe, g_clahe, b_clahe])

# Calculate and display PSNR, SSIM, and MSE between the original and processed images
display_metrics(img_rgb, rgb_image)

# Display the images in a 2x4 grid for comparison
plt.figure(figsize=(15, 10))

# Display the original Red channel (grayscale)
plt.subplot(2, 4, 1)
plt.imshow(r, cmap='gray')
plt.title("Original Red Channel")
plt.axis('off')

# Display the original Green channel (grayscale)
plt.subplot(2, 4, 2)
plt.imshow(g, cmap='gray')
plt.title("Original Green Channel")
plt.axis('off')

# Display the original Blue channel (grayscale)
plt.subplot(2, 4, 3)
plt.imshow(b, cmap='gray')
plt.title("Original Blue Channel")
plt.axis('off')

# Display the original RGB image (in grayscale)
plt.subplot(2, 4, 4)
plt.imshow(img_rgb, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Display the processed Red channel (after CLAHE)
plt.subplot(2, 4, 5)
plt.imshow(r_clahe, cmap='gray')
plt.title("Processed Red Channel (CLAHE)")
plt.axis('off')

# Display the processed Green channel (after CLAHE)
plt.subplot(2, 4, 6)
plt.imshow(g_clahe, cmap='gray')
plt.title("Processed Green Channel (CLAHE)")
plt.axis('off')

# Display the processed Blue channel (after CLAHE)
plt.subplot(2, 4, 7)
plt.imshow(b_clahe, cmap='gray')
plt.title("Processed Blue Channel (CLAHE)")
plt.axis('off')

# Display the final RGB image after merging the processed channels
plt.subplot(2, 4, 8)
plt.imshow(rgb_image)
plt.title("Processed Image (CLAHE)")
plt.axis('off')

# Show the plot
plt.show()
