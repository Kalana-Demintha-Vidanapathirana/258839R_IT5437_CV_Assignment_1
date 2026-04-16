import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.5):
    """
    Apply image sharpening using unsharp masking.
    """
    blurred = cv2.GaussianBlur(img, kernel_size, sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return blurred, sharpened


# Base directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Paths
img_path = BASE_DIR / "data" / "raw" / "emma.jpg"   # change if needed
out_dir = BASE_DIR / "results" / "q09"
out_dir.mkdir(parents=True, exist_ok=True)

# Load image
img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

if img_bgr is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Apply sharpening
blurred_rgb, sharpened_rgb = unsharp_mask(img_rgb, kernel_size=(5, 5), sigma=1.0, amount=1.5)

# Save outputs
cv2.imwrite(str(out_dir / "q09_original.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(out_dir / "q09_blurred.png"), cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(out_dir / "q09_sharpened.png"), cv2.cvtColor(sharpened_rgb, cv2.COLOR_RGB2BGR))

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(blurred_rgb)
plt.title("Blurred Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sharpened_rgb)
plt.title("Sharpened Image")
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"Saved outputs to: {out_dir}")