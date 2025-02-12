import os
from PIL import Image
from datetime import datetime

# Path to the main folder
main_folder_path = 'Public BDD (Intel)/'

# Function to get the format, dimensions, size, and modification date of images
def get_image_info(image_path):
    try:
        with Image.open(image_path) as img:
            format = img.format
            dimensions = img.size
            channels = len(img.getbands())  # Number of color channels (RGB, etc.)
            return format, dimensions, channels
    except Exception as e:
        return None, None, None

# Function to get the size and modification date of the file
def get_file_info(file_path):
    size = os.path.getsize(file_path)  # Size in bytes
    modification_date = os.path.getmtime(file_path)  # Last modification date
    return size, datetime.fromtimestamp(modification_date).strftime('%Y-%m-%d %H:%M:%S')

# Recursive function to traverse all folders and subfolders
def analyze_folder(path):
    # Traverse all items in the current folder
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)

        # Check if it's a folder
        if os.path.isdir(folder_path):
            print(f"\nFolder: {folder_path}")
            num_images = 0
            images_info = []

            # Traverse the files within the folder
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)

                # If the file is an image (this can be adjusted based on the formats)
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif')):
                    num_images += 1
                    format, dimensions, channels = get_image_info(file_path)
                    size, modification_date = get_file_info(file_path)
                    images_info.append((file, format, dimensions, channels, size, modification_date))

            # Display the number of images and their details
            print(f"Number of images: {num_images}")
            for info in images_info:
                file, format, dimensions, channels, size, modification_date = info
                print(f"- {file} | Format: {format} | Dimensions: {dimensions} | "
                      f"Channels: {channels} | Size: {size} bytes | Modification date: {modification_date}")

            # Recursive call to explore subfolders
            analyze_folder(folder_path)

# Call the function to analyze the main folder
analyze_folder(main_folder_path)
