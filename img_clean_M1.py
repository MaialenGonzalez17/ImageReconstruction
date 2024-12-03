import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load Image
def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Correct Image Color and Remove Noise
def correct_color_and_denoise(image):
    # Convert to LAB color space and equalize the luminance channel
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    l_channel = cv2.equalizeHist(l_channel)
    lab_image = cv2.merge([l_channel, a_channel, b_channel])
    corrected_image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2RGB)

    # Apply Gaussian smoothing to reduce noise
    denoised_image = cv2.GaussianBlur(corrected_image, (5, 5), 0)
    return denoised_image

# Step 3: Generate Lightmap
def generate_lightmap(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

# Step 4: Enhance Texture
def enhance_texture(denoised_image, lightmap):
    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2GRAY)
    lightmap_gray = cv2.cvtColor(lightmap, cv2.COLOR_RGB2GRAY)
    return cv2.subtract(gray_image, lightmap_gray)

# Step 5: Create Glandular Tube Masks (GTM)
def create_glandular_masks(enhanced_texture):
    _, gtm0 = cv2.threshold(enhanced_texture, 50, 255, cv2.THRESH_BINARY)  # GTM0
    gtm1 = cv2.erode(gtm0, None, iterations=2)  # GTM1
    gtm0_dilated = cv2.dilate(gtm0, None, iterations=2)  # Dilation
    return gtm0, gtm1, gtm0_dilated

# Step 6: Create GTM Overlay
def create_gtm_overlay(gtm0_dilated, gtm1):
    gtm2 = np.zeros_like(gtm0_dilated)
    gtm2[gtm0_dilated == 255] = 1  # Possible foreground (blue in overlay)
    gtm2[gtm1 == 255] = 2  # Definite foreground (yellow in overlay)
    return gtm2

# Step 7: Simulate Feature Point Sampling
def feature_point_sampling(mask, num_points=50):
    # Randomly sample points within the mask's foreground
    foreground_points = np.column_stack(np.where(mask > 0))
    sampled_points = foreground_points[np.random.choice(foreground_points.shape[0], num_points, replace=False)]
    return sampled_points

# Display Results
def display_results(results):
    num_images = len(results)  # Number of images to display
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))  # Adjust figure size based on the number of images

    # Iterate through results and plot each image in the corresponding subplot
    for idx, (title, img) in enumerate(results.items()):
        ax = axes[idx] if num_images > 1 else axes  # Handle single subplot case
        if len(img.shape) == 2:  # Grayscale
            ax.imshow(img, cmap="gray")
        else:  # Color
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()  # Automatically adjust subplot spacing
    plt.show()

# Main Pipeline
def process_image(image_path):
    input_image = load_image(image_path)
    denoised_image = correct_color_and_denoise(input_image)
    lightmap = generate_lightmap(denoised_image)
    enhanced_texture = enhance_texture(denoised_image, lightmap)
    gtm0, gtm1, gtm0_dilated = create_glandular_masks(enhanced_texture)
    gtm2 = create_gtm_overlay(gtm0_dilated, gtm1)
    sampled_points = feature_point_sampling(gtm0)

    # Display all results
    results = {
        "Original Image": input_image,
        "Corrected & Denoised": denoised_image,

    }
    display_results(results)

    print("Sampled Feature Points:", sampled_points)

# Run the pipeline
process_image("Cervix1.jpg")
