import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Base directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Paths
img_dir = BASE_DIR / "data" / "raw"
out_dir = BASE_DIR / "results" / "q08"
out_dir.mkdir(parents=True, exist_ok=True)

img_path = img_dir / "emma_corrupted.jpg"   # change to .jpg if needed

# Load image
img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")

# (a) Gaussian smoothing
gaussian_smoothed = cv2.GaussianBlur(img, (5, 5), sigmaX=1.0, sigmaY=1.0)

# (b) Median filtering
median_filtered = cv2.medianBlur(img, 5)

# Save outputs
cv2.imwrite(str(out_dir / "q08_original_corrupted.png"), img)
cv2.imwrite(str(out_dir / "q08_gaussian_smoothed.png"), gaussian_smoothed)
cv2.imwrite(str(out_dir / "q08_median_filtered.png"), median_filtered)

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Corrupted Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gaussian_smoothed, cmap="gray")
plt.title("Gaussian Smoothing")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(median_filtered, cmap="gray")
plt.title("Median Filtering")
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"Saved outputs to: {out_dir}")