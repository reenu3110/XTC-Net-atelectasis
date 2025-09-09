import os
import cv2

# Folder path containing the input images
folder_path = "/home/213112002/Atelectasis/Dataset/external/chestxray_binary/Normal/"

# Output folder path to save the CLAHE-enhanced images
output_folder_path = "/home/213112002/Atelectasis/Dataset/external/chestxray_binary/clahe/Normal/"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Create a CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Iterate over each image in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.png','.jpeg')):
        # Load the image
        image = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)

        # Apply CLAHE to the image
        clahe_image = clahe.apply(image)

        # Save the CLAHE-enhanced image
        output_filename = os.path.join(output_folder_path, filename)
        cv2.imwrite(output_filename, clahe_image)

print("CLAHE processing completed.")