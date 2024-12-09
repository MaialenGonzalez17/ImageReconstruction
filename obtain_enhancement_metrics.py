import os
import cv2
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as ssim
#from image_enhancement_pipeline import image_enhancement_pipeline


# The Peak Signal-to-Noise Ratio (PSNR) is a metric commonly used to evaluate the quality of an image after compression or enhancement.

def get_all_jpg_paths(directory):
    # Find all .jpg files in the directory and subdirectories
    jpg_files = glob.glob(f"{directory}/**/*.jpg", recursive=True)
    return jpg_files


def calculate_psnr(img1, img2):
    # Ensure the images are the same shape
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Compute Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite if MSE is zero

    # Define max pixel value (255 for 8-bit images)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr


def calculate_ssim(img1, img2):
    # Convert to grayscale for SSIM calculation
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score


def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def calculate_entropy(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    entropy = shannon_entropy(img_gray)
    return entropy


def calculate_sharpness(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return laplacian_var


def calculate_contrast(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = img_gray.std()
    return contrast


# Calculate quality metrics
def calculate_1image_metrics(image):
    entropy_value = calculate_entropy(image)
    sharpness_value = calculate_sharpness(image)
    contrast_value = calculate_contrast(image)

    return round(entropy_value, 2), round(sharpness_value, 2), round(contrast_value, 2)


def calculate_metrics(original_img, modified_img):
    psnr_value = calculate_psnr(original_img, modified_img)
    ssim_value = calculate_ssim(original_img, modified_img)
    mse_value = calculate_mse(original_img, modified_img)

    return round(psnr_value, 2), round(ssim_value, 2), round(mse_value, 2)


def create_csv_individual_metrics(filepaths, database_name, origin='org', output_path='/home/VICOMTECH/pnmartinez/raginia'):
    # filenames = file_dir #os.listdir(file_dir)
    # if database_name == 'innitius':
    #     filenames = [i for i in filenames if '.bmp' in i]

    data = []
    for img_path in tqdm(filepaths):
        org_img = cv2.imread(img_path)

        try:
            if origin == 'enh':
                img = image_enhancement_pipeline(org_img)
            else:
                img = org_img.copy()

            ent, sharp, cntrst = calculate_1image_metrics(img)

            if database_name == 'innitius':
                img_name = img_path.split('/')[-1]
            else:
                img_name = f"{img_path.split('/')[-2]}_{img_path.split('/')[-1]}"

            data.append([img_name, ent, sharp, cntrst])
        except Exception as e:
            print(f"Problems with case {img_path}. Skipping")
            continue

    df = pd.DataFrame(data, columns=['file', 'entropy', 'sharpness', 'contrast'])
    df.to_csv(f'{output_path}/results_{database_name}_{origin}.csv', index=False)
    print('File created!')


def create_csv_comparison_metrics(filepaths, database_name, output_path='/home/VICOMTECH/pnmartinez/raginia'):
    # filenames = file_dir #os.listdir(file_dir)
    # if database_name == 'innitius':
    #     filenames = [i for i in filenames if '.bmp' in i]

    data = []
    for img_path in tqdm(filepaths):
        org_img = cv2.imread(img_path)

        try:
            enh_img = image_enhancement_pipeline(org_img)

            psnr_value, ssim_value, mse_value = calculate_metrics(org_img, enh_img)

            if database_name == 'innitius':
                img_name = img_path.split('/')[-1]
            else:
                img_name = f"{img_path.split('/')[-2]}_{img_path.split('/')[-1]}"

            data.append([img_name, psnr_value, ssim_value, mse_value])
        except Exception as e:
            print(f"Problems with case {img_path}. Skipping")
            continue

    df = pd.DataFrame(data, columns=['file', 'psnr', 'ssim', 'mse'])
    df.to_csv(f'/{output_path}/raginia/results_comparison_{database_name}.csv', index=False)
    print('File created!')


if __name__ == "__main__":

    database_name = 'intel'
    images_origin = 'enh'

    if database_name == 'innitius':

        file_dir = '/gpfs-cluster/proiektuak/DI02/VISUALIZE_INNITIUS/raw_data/TandaMedidasCervix/'
        filenames = os.listdir(file_dir)
        filenames = [i for i in filenames if '.bmp' in i]

        filepaths = [os.path.join(file_dir, i) for i in filenames]

    else:
        filepaths = get_all_jpg_paths('/gpfs-cluster/proiektuak/DI02/DATA/Intel-CervicalCancer/')
        filepaths = [i for i in filepaths if 'test' not in i]

    print(f"**** Working with database {database_name} and type of images {images_origin}")
    # create_csv_individual_metrics(filepaths, database_name, images_origin)
    create_csv_comparison_metrics(filepaths, database_name)