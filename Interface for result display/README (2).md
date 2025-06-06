# Image Restoration Application with Streamlit

This script implements an interactive application with **Streamlit** to restore degraded images and videos, using two approaches: **traditional methods** and **artificial intelligence (AI)-based models**.

## Main functionalities

- Web interface for image and video restoration.
- Comparison between original, degraded and restored images.
- Display of quality metrics with and without reference.
- Use of a convolutional autoencoder (AI) model for restoration.
- Restoration alternative based on a traditional enhancement pipeline.
- Support for image or video input.

## Model loading

A pre-trained **autoencoder convolutional** model is loaded from a `checkpoint.pt` and GPU usage is set if available. The model is used in inference mode (`eval()`).

## Method Selection

The user can choose between two restoration methods:

- **Traditional**: processing using a classical image enhancement pipeline.
- **IA**: processing by a convolutional autoencoder.

## Processing Flow

### 1. Image Restoration

For each degraded image generated from an original one:

- A specific transformation is applied.
- The restored image is generated, either with AI or with the traditional method.
- The three versions are displayed in columns: Original, Degraded and Restored.
- Quality metrics are calculated and displayed:

  - **With reference**: SSIM, PSNR, MSE, LPIPS (compare to original image).
  - Without reference**: Entropy, Sharpness, Contrast, Color, BRISQUE, NIQE, NIMA (without the need of the original image).

- Metrics are presented by means of spider chart type radial graphs.

### 2. Video Restoration

For videos:

- The video frames are captured using OpenCV.
- For each frame, the same degradations are applied as for the images.
- The restored frames are processed in parallel and displayed using Streamlit placeholders.
- The display is limited to 100 frames by default.

## Display and Explanations

- **Explanations** are included for each degradation applied, obtained from a predefined dictionary.
- Metrics and results are clearly organized for easy interpretation by the user.

## Imported Files and Modules

The script depends on several custom files and modules:

- `Functions_app.py`: contains key functions such as image loading, transformations, metrics, and the traditional pipeline.
- design.py`: provides visual elements such as HTML header, CSS styles and explanatory content about basic knowledge or autoencoders.

## User Interface

The user interface is organized in three columns for each processing:

- **Column 1**: Original image.
- **Column 2**: Degraded image (with a specific transformation).
- **Column 3**: Restored image.

## Device

The device (CPU or GPU) is automatically determined using PyTorch:

````python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
