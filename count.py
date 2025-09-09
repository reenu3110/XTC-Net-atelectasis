import os
from PIL import Image  # Make sure to install the Pillow library if not installed: pip install Pillow

def count_images_in_folder(folder_path):
    # List all files in the folder
    all_files = os.listdir(folder_path)

    # Filter only image files (you can extend the list of valid extensions)
    image_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Count the number of image files
    num_images = len(image_files)

    return num_images

# Replace 'your_folder_path' with the path to your folder
folder_path = "/home/213112002/Atelectasis/Dataset/dataset-clahe/Atelectasis/"
num_images = count_images_in_folder(folder_path)

print(f'The number of images in the folder "{folder_path}" is: {num_images}')