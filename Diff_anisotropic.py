import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import PSNR, SSIM, and MSE calculation functions
from obtain_enhancement_metrics import calculate_psnr, calculate_ssim, calculate_mse

# Function to load the image in RGB
def load_image_rgb(image_path):
    """
    Load an image in RGB format.
    """
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Function to apply anisotropic diffusion to each channel of an RGB image
def anisodiff_rgb(img, niter=1, kappa=50, gamma=0.1, step=(1., 1.), option=1, ploton=False):
    """
    Apply anisotropic diffusion to an RGB image by processing each channel independently.
    """
    if img.ndim == 3:
        # Process each channel (R, G, B) independently
        imgout = np.zeros_like(img, dtype='float32')

        for i in range(3):
            imgout[:, :, i] = anisodiff(img[:, :, i], niter=niter, kappa=kappa, gamma=gamma,
                                        step=step, option=option, ploton=False)

        return np.uint8(imgout)

    else:
        raise ValueError("Input image must be a 3D RGB image.")


# Function to apply anisotropic diffusion (as in previous code for grayscale)
def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1., 1.), option=1, ploton=False):
    """
    Apply anisotropic diffusion to a grayscale image.
    """
    img = img.astype('float32')
    imgout = img.copy()

    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    if ploton:
        import pylab as pl
        from time import sleep
        fig = pl.figure(figsize=(20, 5.5), num="Anisotropic Diffusion")
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
        ax1.imshow(img, interpolation='nearest')
        ih = ax2.imshow(imgout, interpolation='nearest', animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")
        fig.canvas.draw()

    for ii in range(niter):
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        if option == 1:
            gS = np.exp(-(deltaS / kappa) ** 2.) / step[0]
            gE = np.exp(-(deltaE / kappa) ** 2.) / step[1]
        elif option == 2:
            gS = 1. / (1 + (deltaS / kappa) ** 2.) / step[0]
            gE = 1. / (1 + (deltaE / kappa) ** 2.) / step[1]

        E = gE * deltaE
        S = gS * deltaS

        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        imgout += gamma * (NS + EW)

        if ploton:
            iterstring = "Iteration %i" % (ii + 1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()

    return imgout


# Main function to load, process the RGB image, and calculate metrics
def process_image_rgb(image_path, niter=5):
    # Load the RGB image
    img_rgb = load_image_rgb(image_path)

    # Apply anisotropic diffusion
    diffused_img_rgb = anisodiff_rgb(img_rgb, niter=niter, kappa=50, gamma=0.1, option=1, ploton=False)

    # Calculate PSNR, SSIM, and MSE between original and diffused images
    psnr_value = calculate_psnr(img_rgb, diffused_img_rgb)
    ssim_value = calculate_ssim(img_rgb, diffused_img_rgb)
    mse_value = calculate_mse(img_rgb, diffused_img_rgb)

    # Print PSNR, SSIM, and MSE values
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"MSE: {mse_value:.2f}")

    # Display the original and diffused images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original RGB Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(diffused_img_rgb)
    plt.title(f"Diffused RGB Image (Iterations: {niter})")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    # Example: Process an RGB image with anisotropic diffusion and calculate metrics
    image_path = "Public BDD (Intel)/0.jpg"  # Replace with your image path
    process_image_rgb(image_path, niter=10)
