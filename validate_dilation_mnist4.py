import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras

def dilate_digit(img_np, max_kernel=2):
    """
    Apply morphological dilation to a grayscale image (0–255).
    """
    k_size = np.random.randint(1, max_kernel + 1)
    kernel = np.ones((k_size, k_size), np.uint8)
    dilated = cv2.dilate(img_np, kernel, iterations=1)
    return np.clip(dilated, 0, 255).astype(np.uint8)

# Load MNIST using Keras (same as your baseline)
print("Loading MNIST data...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Find indices of 4s in training data
fours_indices = np.where(y_train == 4)[0][:8]  # Get 8 examples of 4s

# Extract the images (already 28x28, values 0-255)
fours = x_train[fours_indices]

print(f"Found {len(fours)} examples of digit '4' for testing")

# Apply dilation and plot comparisons
fig, axes = plt.subplots(len(fours), 2, figsize=(6, len(fours) * 1.5))

for i, img in enumerate(fours):
    # Apply dilation
    dilated_img = dilate_digit(img, max_kernel=2)
    
    # Plot original
    axes[i, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axes[i, 0].set_title(f"Original 4 (#{i})")
    axes[i, 0].axis('off')
    
    # Plot dilated
    axes[i, 1].imshow(dilated_img, cmap='gray', vmin=0, vmax=255)
    axes[i, 1].set_title(f"Dilated 4 (#{i})")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig('dilation_validation.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nValidation complete. Check the before/after comparisons:")
print("1. Do the dilated 4s show thicker strokes?")
print("2. Do any top triangular areas look more closed/loop-like?") 
print("3. Could any dilated versions be mistaken for 9s?")
print("\nIf yes to these questions, the augmentation is working as intended.")