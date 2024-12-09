import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an endoscopic image
def load_image(image_path):
    """
    Loads an image from the specified path.
    """
    return cv2.imread(image_path)

# Temporal filtering: applies a median filter
def temporal_filtering(image, filter_size):
    """
    Applies temporal median filtering to the image using the specified filter size.
    """
    return cv2.medianBlur(image, filter_size)

# Color normalization: normalize the color channels
def color_normalization(image):
    """
    Normalizes the color of the image using LAB color space.
    """
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image_lab)
    l = cv2.equalizeHist(l)  # Equalize the luminance channel
    normalized_image = cv2.merge((l, a, b))
    return cv2.cvtColor(normalized_image, cv2.COLOR_LAB2BGR)

# Undistortion: applies distortion correction if calibration parameters are known
def undistort_image(image, camera_matrix, dist_coeffs):
    """
    Applies undistortion to the image using the provided camera matrix and distortion coefficients.
    """
    height, width = image.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1)
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

# Display images in a subplot
def display_images(titles, images):
    """
    Displays images in a single figure using subplots.
    """
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for i, (title, img) in enumerate(zip(titles, images)):
        ax = axes[i]
        ax.set_title(title)
        if len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap='gray')
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# Main function to process the image
def process_endoscopic_image(image_path, camera_matrix, dist_coeffs):
    """
    Processes the endoscopic image using temporal filtering, color normalization, and undistortion.
    Displays the results.
    """
    # Load the image
    image = load_image(image_path)

    # Temporal filtering with filter sizes 3 and 5
    temporal_filtered_3 = temporal_filtering(image, 3)
    temporal_filtered_5 = temporal_filtering(image, 5)

    # Color normalization
    color_normalized = color_normalization(image)

    # Undistortion (replace with actual camera matrix and distortion coefficients)
    undistorted_image = undistort_image(image, camera_matrix, dist_coeffs)

    # Display results
    titles = ["Original", "Temporal Filtered (Size 3)", "Temporal Filtered (Size 5)",
              "Color Normalized", "Undistorted"]
    images = [image, temporal_filtered_3, temporal_filtered_5, color_normalized, undistorted_image]
    display_images(titles, images)

# Example of using the functions
if __name__ == "__main__":
    # Provide the path to your endoscopic image
    image_path = "image/Cervix2.jpg"

    # Dummy camera matrix and distortion coefficients (replace with your own)
    camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])  # Example camera matrix
    dist_coeffs = np.array([-0.2, 0.1, 0, 0, 0])  # Example distortion coefficients

    # Process the image
    process_endoscopic_image(image_path, camera_matrix, dist_coeffs)
