import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from keras.layers import Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Reshape, Flatten, MultiHeadAttention
from PIL import Image
import logging
from pathlib import Path

# Supress info, warnings and error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setup logging
logging.basicConfig(level=logging.INFO)

# Function to prepare data generators
def dataGen():
    try:
        batch = 32
        train_datagen = ImageDataGenerator(
            rescale=1./255, 
            validation_split=0.1, 
            rotation_range=40, 
            width_shift_range=0.3, 
            height_shift_range=0.3, 
            shear_range=0.3,
            zoom_range=0.3, 
            horizontal_flip=True, 
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        size = (224, 224)
        train_path = Path("/home/213112002/Atelectasis/Dataset/clahe/training/")
        test_path = Path("/home/213112002/Atelectasis/Dataset/clahe/testing/")
        
        if not train_path.exists() or not test_path.exists():
            logging.error("Training or Testing directory does not exist.")
            return None
        
        train_generator = train_datagen.flow_from_directory(
            str(train_path), 
            target_size=size,  
            batch_size=batch,
            subset="training",
            class_mode='categorical'
        )
        
        valid_generator = train_datagen.flow_from_directory(
            str(train_path), 
            target_size=size,  
            batch_size=batch,
            subset="validation",
            class_mode='categorical'
        )
        
        test_generator = test_datagen.flow_from_directory(
            str(test_path), 
            target_size=size,  
            batch_size=batch,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, test_generator, valid_generator
    
    except Exception as e:
        logging.error(f"Error in dataGen: {e}")
        return None

# Function to build model
def build_Xception_with_transformer(IMAGE_SIZE, channels, num_classes):
    base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, channels))
    input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, channels))
    x = Conv2D(3, (3, 3), padding='same')(input)
    x = base_model(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(16 * num_classes, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(x)
    x = Dropout(0.5)(x)

    num_heads = 8
    transformer_block = MultiHeadAttention(num_heads=num_heads, key_dim=16)(x, x)
    transformer_block = Dense(128, activation='relu')(transformer_block)
    
    capsule_dim = 8
    capsules = Reshape(target_shape=(-1, capsule_dim))(transformer_block)
    digit_caps1 = Dense(capsule_dim * num_classes, activation='relu')(capsules)
    
    output = Dense(num_classes, activation='sigmoid')(digit_caps1)
    output = Flatten()(output)
    output = Dense(num_classes, activation='sigmoid')(output)
    
    model = tf.keras.Model(inputs=input, outputs=output)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                  
    return model

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Other callbacks
annealer = ReduceLROnPlateau(
    monitor='val_accuracy', 
    factor=0.70, 
    patience=5, 
    verbose=1, 
    min_lr=1e-4
)
checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)

# Plot model architecture
model = build_Xception_with_transformer(224, 3, 2)
plot_model(model, to_file='convnet.png', show_shapes=True, show_layer_names=True)

# Configure GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100

# Generate data
train_generator, test_generator, valid_generator = dataGen()

# Check data generation
if train_generator and test_generator and valid_generator:
    print("Data generators created successfully.")
else:
    print("Error in creating data generators.")

# Train the model
hist = model.fit(
    train_generator,
    epochs=EPOCHS,
    verbose=1,
    validation_data=valid_generator, 
    callbacks=[annealer, checkpoint, early_stopping]
)

# Save the model
model.save("model_capsul.h5")

# Evaluate the model
test_generator.reset()
Y_pred1 = model.predict(test_generator)
Y_pred = np.argmax(Y_pred1, axis=1)
Y_true = test_generator.classes

final_loss, final_accuracy = model.evaluate(test_generator)
print(f"\nFinal Loss: {final_loss}, Final Accuracy: {final_accuracy}")

# Plot accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Confusion matrix and classification report
true_label, pred_label = Y_pred, Y_true
acc = accuracy_score(true_label, pred_label)
target_names = ['Atelectasis', 'Normal']
Gmean = geometric_mean_score(true_label, pred_label, average='weighted')
result = sensitivity_specificity_support(true_label, pred_label, average='macro')

print(f"Sensitivity: {100 * result[0]:.2f}%", f"Specificity: {100 * result[1]:.2f}%", f"Accuracy: {100 * acc:.2f}%", f"Gmean: {100 * Gmean:.2f}%\n")

report = classification_report(true_label, pred_label, target_names=target_names, digits=4)
print(report)

with open('report.txt', 'w') as f:
    f.write(report)

matrix = confusion_matrix(true_label, pred_label)
matrix = matrix.astype('float')
df_cm = pd.DataFrame(matrix, target_names, target_names)
fig, ax = plt.subplots(figsize=(5,4))
sns.set(font_scale=1.6)
sns.heatmap(df_cm, annot=True, annot_kws={"size": 12}, ax=ax, cmap="YlOrBr", fmt='g', cbar=False)
plt.ylabel('Actual', fontsize=15, fontweight='bold')
plt.xlabel('Predicted', fontsize=15, fontweight='bold')
plt.ioff()
plt.savefig("confusion_mat.eps", format='eps', bbox_inches='tight')
plt.savefig("confusion_mat.png", format='png', bbox_inches='tight')
plt.show()

print(model.summary())