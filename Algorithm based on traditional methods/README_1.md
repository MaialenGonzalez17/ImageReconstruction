# Description of the image enhancement code without AI

This script performs a complete image enhancement process based on classical image processing techniques, without using artificial intelligence. Below is a step-by-step explanation of how it works:

---

## Main functionality

1. **Loads and preprocesses images**.

   - Reads images from a specified folder.
   - Resizes each image to a fixed size of 1024x1365 pixels for uniformity. #resize to your liking 

2. **Image enhancement processes**.

   Several classic techniques are applied to improve visual quality:

   - **Median filtering** to reduce impulsive noise.
   - **CLAHE** adaptive histogram equalization in the LAB color space to improve local contrast without saturating colors.
   - **Saturation reduction** to soften vivid colors, increasing naturalness.
   - **Sharpening filtering** using a convolution kernel to enhance edges and details.
   - **Normalization** of pixel values to ensure optimal dynamic range.

3. **Saving and metrics calculation**.

   - Saves enhanced images to a target folder.
   - Calculates classic image quality metrics by comparing the original image with the enhanced image:
     - PSNR (Peak Signal-to-Noise Ratio).
     - SSIM (Structural Similarity Index)
     - MSE (Mean Squared Error)
   - Calculates non-reference metrics on the enhanced image:
     - Entropy (information/noise measure).
     - Sharpness
     - Contrast
     - Colorfulness (color vibrancy)

4. **Parallel execution**.

   - Process multiple images simultaneously using `ThreadPoolExecutor` to speed up processing.

5. **Statistical summary**

   - At the end, calculate the mean and standard deviation of all metrics obtained to globally evaluate the performance of the process.
   - Save the complete results in a CSV file for further analysis.

6. **Execution times**.

   - Displays the total processing time and average time per image.

---

## Summary of key functions

- `resize_image`: Resize image to 320x320.
- `apply_median_filter`: Apply median filter to remove noise.
- `clahe_lab`: Improve local contrast in LAB space with CLAHE.
- `reduce_saturation`: Decrease saturation to soften colors.
- `apply_sharpening`: Enhance details with sharpening filter.
- `image_enhancement_pipeline`: Applies the whole enhancement chain and calculates metrics.
- `process_images_in_folder`: Get valid image paths from a folder.
- `process_images_in_parallel`: Runs the enhancement in parallel.
- `calculate_stats`: Calculates means and standard deviations of metrics.

---

## Usage

Set input and output paths in:

````python
folder_path = "input_images_folder/"
save_folder = "enhanced_images_folder/"
output_csv = "results_path/results.csv"
