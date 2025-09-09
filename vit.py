import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification
from transformers import ViTFeatureExtractor
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support


class AtelectasisDataset(Dataset):
    def __init__(self, root_dir, feature_extractor):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.classes = ['Atelectasis', 'Normal']
        self.data = []
        for idx, cls in enumerate(self.classes):
            path = os.path.join(root_dir, cls)
            for img in os.listdir(path):
                self.data.append((os.path.join(path, img), idx))

        # Specify mean and std based on the feature extractor's requirements
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = torchvision.io.read_image(img_path)

        # Convert the input image tensor to a float tensor
        image = image.float()

        # Calculate mean and std for a single-channel image
        mean_single_channel = torch.tensor([0.485])  # Replace with your actual mean
        std_single_channel = torch.tensor([0.229])  # Replace with your actual std

        # Normalize the single-channel grayscale image
        image = (image - mean_single_channel) / std_single_channel

        # Clip values to be within the range [0, 1]
        image = torch.clamp(image, 0, 1)

        # Expand dimensions to make it a 3-channel image (assuming grayscale)
        image = image.expand(3, -1, -1)

        # Process the image with the feature extractor
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values[0]
        return image, label

root_dir = "/home/213112002/Atelectasis/Dataset/"
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
dataset = AtelectasisDataset(root_dir, feature_extractor)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ViTClassifier, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit.classifier = nn.Linear(self.vit.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.vit(pixel_values=x).logits
        return x

model = ViTClassifier(num_classes=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
train_accuracies = []
num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f'Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}')


model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")
print(classification_report(all_labels, all_preds, target_names=['Atelectasis', 'Normal']))

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print(conf_matrix)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.show()


df = pd.DataFrame({
    'Model': ['ViTClassifier'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1]
})
df.to_csv('model_performance.csv', index=False)

np.savetxt("confusion_matrix.csv", conf_matrix, delimiter=",", fmt='%d')

torch.save(model.state_dict(), 'Atelectasis_disease_classifier.pth')

import seaborn as sns

# Assuming `conf_matrix` is your confusion matrix obtained from the evaluation step
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Atelectasis', 'Normal'], yticklabels=['Atelectasis', 'Normal'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()


model_path = "/home/213112002/Atelectasis/Atelectasis_disease_classifier.pth"
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mod

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy and other metrics if needed
accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy: {accuracy}')

# Print the classification report
print(classification_report(all_labels, all_preds, target_names=['Atelectasis', 'Normal']))


import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming all_preds and all_labels are numerical indices
class_names = ['Atelectasis', 'Normal']
all_preds_folder_names = [class_names[pred_index] for pred_index in all_preds]
all_labels_folder_names = [class_names[label_index] for label_index in all_labels]

# Calculate classification metrics
accuracy = accuracy_score(all_labels_folder_names, all_preds_folder_names)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels_folder_names, all_preds_folder_names, average='macro')

# Print metrics to console
print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}')

# Create a DataFrame and save it to a CSV file
report_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [accuracy, precision, recall, f1]
})
report_df.to_csv('/kaggle/working/classification_report.csv', index=False)

# Calculate and plot the confusion matrix
conf_matrix = confusion_matrix(all_labels_folder_names, all_preds_folder_names)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Atelectasis', 'Normal'], yticklabels=['Atelectasis', 'Normal'])
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('/kaggle/working/confusion_matrix.png')
plt.show()


import matplotlib.pyplot as plt

# Assuming train_losses and train_accuracies contain loss and accuracy for each epoch
epochs = range(1, num_epochs + 1)

# Plotting training loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Training Loss', color='red')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_loss_accuracy.png')
plt.show()
