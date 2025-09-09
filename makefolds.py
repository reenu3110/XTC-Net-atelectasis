import os
import shutil
from sklearn.model_selection import StratifiedKFold

# Path to the directory containing all images with labels
dataset_dir = r"/home/213112002/Atelectasis/Dataset-new/clahe/Atelectasis/"

# Output directory where folds will be created
output_dir = r"/home/213112002/Atelectasis/Dataset-new/clahe-fold/"

# Number of folds
num_folds = 5

# Create a list to store image filenames
data = []

# Walk through the dataset directory and collect image filenames
for img_filename in os.listdir(dataset_dir):
    img_path = os.path.join(dataset_dir, img_filename)
    data.append(img_path)

# Calculate the number of samples for each split (70:10:20 ratio)
num_samples = len(data)
num_val = int(0.1 * num_samples)
num_test = int(0.2 * num_samples)
num_train = num_samples - num_val - num_test

# Create an empty list to store the folds
folds = []

# Create the folds
for fold_num in range(num_folds):
    fold_dir = os.path.join(output_dir, f"fold_{fold_num + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    val_start = fold_num * num_val
    val_end = (fold_num + 1) * num_val
    test_start = val_end
    test_end = test_start + num_test

    fold_data = {
        'train': [],
        'val': [],
        'test': []
    }

    for i, img_path in enumerate(data):
        if val_start <= i < val_end:
            fold_data['val'].append(img_path)
        elif test_start <= i < test_end:
            fold_data['test'].append(img_path)
        else:
            fold_data['train'].append(img_path)

    folds.append(fold_data)

# Now, you have a list "folds" containing each fold with train, val, and test data

# Iterate through folds to organize the data into your desired folder structure
for fold_num, fold_data in enumerate(folds):
    fold_dir = os.path.join(output_dir, f"fold_{fold_num + 1}")

    for split_name, data_list in fold_data.items():
        split_dir = os.path.join(fold_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for img_path in data_list:
            img_filename = os.path.basename(img_path)
            dst_path = os.path.join(split_dir, img_filename)
            shutil.copy(img_path, dst_path)

print("Data organization and fold creation completed.")