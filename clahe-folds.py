import os
import cv2
import shutil

# Base folder path containing the folds
base_folder_path = "/home/213112002/Atelectasis/Dataset/Dataset_folds/"

# Output base folder path to save the CLAHE-enhanced images
output_base_folder_path = "/home/213112002/Atelectasis/Dataset-new/clahe-folds-new-new/"

# Clear and recreate the output base folder
if os.path.exists(output_base_folder_path):
    shutil.rmtree(output_base_folder_path)
os.makedirs(output_base_folder_path)

# Create a CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def process_images_in_folder(input_folder, output_folder):
    """
    Apply CLAHE to all images in the input folder and save them to the output folder.
    """
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            relative_path = os.path.relpath(file_path, base_folder_path)
            output_path = os.path.join(output_base_folder_path, relative_path)

            # Create the output folder structure if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                # Load and process the image
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    clahe_image = clahe.apply(image)
                    cv2.imwrite(output_path, clahe_image)
                    print(f"Processed and saved: {output_path}")
                else:
                    print(f"Skipped {file_path}: Unable to read as an image.")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        elif os.path.isdir(file_path):
            # Process subdirectories recursively
            sub_output_folder = os.path.join(output_folder, filename)
            process_images_in_folder(file_path, sub_output_folder)


# Process each fold
for fold in os.listdir(base_folder_path):
    fold_path = os.path.join(base_folder_path, fold)
    if os.path.isdir(fold_path):  # Only process directories
        output_fold_path = os.path.join(output_base_folder_path, fold)
        process_images_in_folder(fold_path, output_fold_path)

print("CLAHE processing for all folds completed.")
