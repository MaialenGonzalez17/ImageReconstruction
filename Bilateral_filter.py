import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import PSNR, SSIM, and MSE calculation functions
from obtain_enhancement_metrics import calculate_psnr, calculate_ssim, calculate_mse

# Function to calculate and display PSNR, SSIM, and MSE
def display_metrics(original_img, filtered_img):
    # Calculate PSNR, SSIM, and MSE
    psnr_value = calculate_psnr(original_img, filtered_img)
    ssim_value = calculate_ssim(original_img, filtered_img)
    mse_value = calculate_mse(original_img, filtered_img)

    # Print the results
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"MSE: {mse_value:.2f}")

if __name__ == "__main__":
    # Read the image using OpenCV
    img = cv2.imread("Public BDD (Intel)/0.jpg")

# Convert the image from BGR (OpenCV default) to RGB (for plotting)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply bilateral filter with d = 15, sigmaColor = sigmaSpace = 75.
bilateral = cv2.bilateralFilter(img, 15, 75, 75)

# Convert the filtered image to RGB for consistent plotting
bilateral_rgb = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)

# Calculate and display PSNR, SSIM, and MSE between original and filtered images
display_metrics(img_rgb, bilateral_rgb)

# Plot the original and filtered images side by side
plt.figure(figsize=(12, 6))

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')  # Hide axes

# Plot the filtered image
plt.subplot(1, 2, 2)
plt.imshow(bilateral_rgb)
plt.title("Bilateral Filtered Image")
plt.axis('off')  # Hide axes

# Show the plot
plt.show()

# Save the filtered image
cv2.imwrite('Cervix1_bilateral.jpg', bilateral)
