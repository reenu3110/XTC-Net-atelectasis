
import numpy as np
import innvestigate
import matplotlib.pyplot as plt

# Load your trained model
model = build_Xception_with_transformer(IMAGE_SIZE, channels, num_classes)
model.load_weights("/home/213112002/Atelectasis/model.h5")

# Create an analyzer for the model
analyzer = innvestigate.create_analyzer("lrp.epsilon", model)

# Choose an input image
input_image = "/home/213112002/Atelectasis/Dataset/training/Atelectasis/Atelectasis (987).jpg"  # Provide your input image here

# Preprocess the input image (e.g., resize, normalize) to match the model's input requirements

# Perform LRP on the input image
analysis = analyzer.analyze(input_image)

# Visualize the relevance scores overlaid on the input image
plt.imshow(input_image)  # Display the input image
plt.imshow(analysis[0], cmap="jet", alpha=0.5)  # Overlay the relevance scores
plt.colorbar()  # Add a color bar to indicate relevance
plt.show()