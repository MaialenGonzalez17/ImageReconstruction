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
````
# Metric Comparison: Degraded vs AI 

## 🔍 No-Reference Metrics (NR)

<div style="overflow-x: auto">

| Transformación              | Comparación | Entropía | Contraste (%) | Nitidez (%) | Colorido (%) | BRISQUE | NIQE  | NIMA  |
|----------------------------|-------------|----------|----------------|--------------|----------------|---------|-------|-------|
| Gaussian Blur              | Degraded    | 7.21     | 20.98          | **2.91**     | 44.35          | 0.72    | **0.64**  | 2.33  |
|                            | Restored    | **7.45** | **23.76**      | 0.76         | **45.76**      | **0.31**| 0.58  | **2.85**  |
| Gaussian Noise             | Degraded    | 7.64     | 20.67          | 100.00       | 54.32          | 0.72    | **0.40**  | 6.34  |
|                            | Restored    | **7.74** | **22.61**      | **100.00**   | **56.47**      | **0.30**| 0.42  | **6.46**  |
| Involuntary Motion         | Degraded    | 7.10     | 20.51          | 13.85        | 45.82          | 0.51    | **0.61**  | 2.41  |
|                            | Restored    | **7.37** | **22.66**      | **92.07**    | **46.10**      | **0.24**| 0.58  | **3.05**  |
| Lighting and Contrast      | Degraded    | 7.01     | 21.05          | 67.60        | 47.19          | 0.20    | **0.63**  | 2.87  |
|                            | Restored    | **7.33** | **23.71**      | **99.29**    | **48.94**      | **0.12**| 0.59  | **3.64**  |
| Uneven Lighting            | Degraded    | 7.11     | 26.72          | 88.38        | 42.89          | 0.17    | **0.67**  | 3.69  |
|                            | Restored    | **7.30** | **27.90**      | **99.81**    | **44.63**      | **0.12**| 0.64  | **4.34**  |
| Random Shadows             | Degraded    | 6.50     | 14.78          | 60.15        | 27.90          | 0.26    | **0.78**  | 2.15  |
|                            | Restored    | **7.00** | **19.32**      | **100.00**   | **35.89**      | **0.11**| 0.65  | **3.02**  |
| Specular Reflections       | Degraded    | 7.18     | 21.46          | 81.92        | 48.82          | 0.15    | **0.62**  | 2.86  |
|                            | Restored    | **7.42** | **23.27**      | **100.00**   | **48.99**      | **0.11**| 0.59  | **3.60**  |
| Condom Defects             | Degraded    | 7.14     | 21.53          | 91.09        | 47.56          | 0.10    | **0.60**  | 3.25  |
|                            | Restored    | **7.40** | **23.72**      | **100.00**   | **48.58**      | **0.07**| 0.56  | **4.40**  |




</div>

## 📊 Reference-Based Metrics (RB)

<div style="overflow-x: auto">

| Transformación              | Comparación             | PSNR  | SSIM  | MSE    | LPIPS |
|----------------------------|-------------------------|-------|-------|--------|--------|
| Desenfoque Gaussiano       | Original - Degradada    | **35.81** | **0.88** | **18.91** | 0.16  |
|                            | Original - Restaurada   | 29.27 | 0.82  | 77.72  | **0.13** |
| Ruido Gaussiano            | Original - Degradada    | **28.12** | 0.14  | **100.34** | 0.75  |
|                            | Original - Restaurada   | 28.10 | **0.20**  | 100.64 | **0.72** |
| Movimiento Involuntario    | Original - Degradada    | **36.82** | **0.89** | **16.05** | 0.07  |
|                            | Original - Restaurada   | 29.58 | 0.81  | 72.50  | **0.09** |
| Iluminación y contraste    | Original - Degradada    | 29.61 | **0.93** | 84.23  | **0.01** |
|                            | Original - Restaurada   | **29.98** | 0.81  | **83.93** | 0.07  |
| Iluminación desigual       | Original - Degradada    | **31.98** | **0.96** | **45.13** | **0.02** |
|                            | Original - Restaurada   | 29.47 | 0.81  | 74.94  | 0.08  |
| Sombras aleatorias         | Original - Degradada    | 28.45 | 0.82  | 95.35  | **0.09** |
|                            | Original - Restaurada   | **29.16** | **0.82**  | **82.06** | 0.11  |
| Reflejos especulares       | Original - Degradada    | **39.23** | **0.95** | **8.25**  | **0.07** |
|                            | Original - Restaurada   | 29.48 | 0.81  | 73.91  | 0.14  |
| Defectos del preservativo  | Original - Degradada    | **41.08** | **0.94** | **5.52**  | **0.10** |
|                            | Original - Restaurada   | 29.58 | 0.80  | 72.95  | 0.18  |



</div>
