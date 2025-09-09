import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image
from tf_explain.core.integrated_gradients import IntegratedGradients

# Load your trained model
model = tf.keras.models.load_model("model_capsul.h5")

# Choose a sample X-ray image for interpretation
image_path = "/home/213112002/Atelectasis/Dataset/clahe/testing/Atelectasis/Atelectasis (1049).jpg"

# Preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Initialize Integrated Gradients explainer
explainer = IntegratedGradients()

# Explain model prediction on the input image
attributions = explainer.explain((img_array, None), model, class_index=1)

# Plot the heatmap of attributions
plt.imshow(attributions[0], cmap="viridis", alpha=0.8)
plt.colorbar()
plt.imshow(img, alpha=0.5)
plt.axis('off')
plt.show()