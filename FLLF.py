import cv2
import numpy as np

def generate_gaussian_pyramid(image, num_levels):
    gaussian_pyramid = [image]
    for i in range(num_levels - 1):
        blurred = cv2.GaussianBlur(gaussian_pyramid[-1], (2*i+1, 2*i+1), 0)
        gaussian_pyramid.append(blurred)
    return gaussian_pyramid

def generate_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        expanded = cv2.resize(gaussian_pyramid[i + 1], gaussian_pyramid[i].shape[::-1])
        laplacian_pyramid.append(gaussian_pyramid[i] - expanded)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

def apply_bilateral_filter(image, d, r):
    return cv2.bilateralFilter(image, -1, d, r)

def generate_modified_laplacian_pyramid(laplacian_pyramid, d, r):
    modified_laplacian_pyramid = []
    for level in laplacian_pyramid:
        modified_laplacian_pyramid.append(apply_bilateral_filter(level, d, r))
    return modified_laplacian_pyramid

def reconstruct_enhanced_image(gaussian_pyramid, modified_laplacian_pyramid):
    enhanced_image = np.zeros_like(gaussian_pyramid[0])
    for i in range(len(modified_laplacian_pyramid)):
        resized_ml = cv2.resize(modified_laplacian_pyramid[i], enhanced_image.shape[::-1])
        enhanced_image += resized_ml
    return np.clip(gaussian_pyramid[0] + enhanced_image, 0, 255).astype(np.uint8)

# Parameters
d = 10  # Spatial distance for bilateral filter
r = 0.1  # Range similarity for bilateral filter
num_levels = 5  # Number of levels in the Laplacian pyramid

# Load the input image
image = cv2.imread("input_image.jpg", cv2.IMREAD_GRAYSCALE)

# Generate Gaussian pyramid
gaussian_pyramid = generate_gaussian_pyramid(image, num_levels)

# Generate Laplacian pyramid
laplacian_pyramid = generate_laplacian_pyramid(gaussian_pyramid)

# Generate modified Laplacian pyramid
modified_laplacian_pyramid = generate_modified_laplacian_pyramid(laplacian_pyramid, d, r)

# Reconstruct enhanced image
enhanced_image = reconstruct_enhanced_image(gaussian_pyramid, modified_laplacian_pyramid)

# Display and save the enhanced image
cv2.imshow("Enhanced Image", enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("enhanced_image.jpg", enhanced_image)