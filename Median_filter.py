import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Import PSNR, SSIM, MSE, entropy, contrast and sharpness calculation functions
from obtain_enhancement_metrics import calculate_psnr, calculate_ssim, calculate_mse, calculate_entropy, calculate_contrast, calculate_sharpness

# Function to load the image in grayscale
def load_image_grayscale(image_path):
    """ Loads an image from the specified path and converts it to grayscale. """
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Function to load the image in RGB color
def load_image_rgb(image_path):
    """ Loads an image from the specified path and converts it to RGB format """
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to apply median filter
def apply_median_filter(image, kernel_size=5):
    """ Applies the median filter """
    return cv2.medianBlur(image, kernel_size)

# Function to display original and filtered images (grayscale and color)
def show_images(original_gray, filtered_gray, original_rgb, filtered_rgb):

    # Show RGB images
    plt.subplot(1, 2, 1)
    plt.title("Original RGB Image")
    plt.imshow(original_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Median Filtered RGB Image")
    plt.imshow(filtered_rgb)
    plt.axis('off')

    plt.show()

# Main function to process the image
def process_image(image_path, kernel_size=5):
    """ Processes the image by loading it in grayscale and color, applying the median filter,
    and displaying the results. Also calculates and prints metrics values. """

    # Load the images in grayscale and color
    original_gray = load_image_grayscale(image_path)
    original_rgb = load_image_rgb(image_path)

    # Apply the median filter to both versions of the image
    filtered_rgb = apply_median_filter(original_rgb, kernel_size)

    # Calculate PSNR, SSIM, and MSE RGB images
    psnr_rgb = calculate_psnr(original_rgb, filtered_rgb)
    ssim_rgb = calculate_ssim(original_rgb, filtered_rgb)
    mse_rgb = calculate_mse(original_rgb, filtered_rgb)
    entropy_gray = calculate_entropy(filtered_rgb)
    contrast_gray = calculate_contrast(filtered_rgb)
    sharpness_gray = calculate_sharpness(filtered_rgb)


    # Print PSNR, SSIM, and MSE values
    return {
        "image": image_path,
        "psnr_rgb": psnr_rgb,
        "ssim_rgb": ssim_rgb,
        "mse_rgb": mse_rgb,
        "entropy_gray": entropy_gray,
        "contrast_gray": contrast_gray,
        "sharpness_gray": sharpness_gray,
    }

    # Display the images
    # show_images(original_gray, filtered_gray, original_rgb, filtered_rgb)

if __name__ == "__main__":
    folder_path = "Public BDD (Intel)/"
    results = []  # List to store results for each image

    # Iterate over all files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(folder_path, file_name)
            # Process the image and obtain results
            result = process_image(image_path, kernel_size=5)
            # Add the results to the list
            results.append(result)

    # Convert the list of results into a pandas DataFrame
    df = pd.DataFrame(results)

    # Save the results to a CSV file
    df.to_csv("image_metrics_results.csv", index=False)

    # Calculate the averages for PSNR, SSIM, and MSE
    mean_psnr_rgb = df["psnr_rgb"].mean()
    mean_ssim_rgb = df["ssim_rgb"].mean()
    mean_mse_rgb = df["mse_rgb"].mean()
    mean_entropy_gray = df["entropy_gray"].mean()
    mean_contrast_gray = df["contrast_gray"].mean()
    mean_sharpness_gray = df["sharpness_gray"].mean()

    # Display the results on the screen
    print("Mean PSNR :", mean_psnr_rgb)
    print("Mean SSIM :", mean_ssim_rgb)
    print("Mean MSE :", mean_mse_rgb)
    print("Mean Entropy :", mean_entropy_gray)
    print("Mean Contrast :", mean_contrast_gray)
    print("Mean Sharpness :", mean_sharpness_gray)
