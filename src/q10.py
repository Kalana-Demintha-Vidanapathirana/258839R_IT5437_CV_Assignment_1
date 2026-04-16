import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def manual_bilateral_filter(img, kernel_size=5, sigma_s=3.0, sigma_r=25.0):
    """
    Manual bilateral filter for grayscale images.
    """
    if img.ndim != 2:
        raise ValueError("Input image must be grayscale.")

    pad = kernel_size // 2
    padded = np.pad(img, pad, mode="reflect")
    output = np.zeros_like(img, dtype=np.float32)

    # Spatial Gaussian
    ax = np.arange(-pad, pad + 1)
    xx, yy = np.meshgrid(ax, ax)
    spatial = np.exp(-(xx**2 + yy**2) / (2 * sigma_s**2))

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            region = padded[y:y + kernel_size, x:x + kernel_size]
            center = padded[y + pad, x + pad]

            range_weight = np.exp(-((region - center) ** 2) / (2 * sigma_r**2))
            weights = spatial * range_weight

            output[y, x] = np.sum(weights * region) / np.sum(weights)

    return np.clip(output, 0, 255).astype(np.uint8)


# Base directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Paths
img_path = BASE_DIR / "data" / "raw" / "einstein.png"
out_dir = BASE_DIR / "results" / "q10"
out_dir.mkdir(parents=True, exist_ok=True)

# Load grayscale image
img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")

# Part (b): Gaussian smoothing
gaussian_result = cv2.GaussianBlur(img, (5, 5), sigmaX=3.0, sigmaY=3.0)

# Part (c): OpenCV bilateral filter
opencv_bilateral = cv2.bilateralFilter(img, d=5, sigmaColor=25.0, sigmaSpace=3.0)

# Part (d): Manual bilateral filter
manual_bilateral = manual_bilateral_filter(img, kernel_size=5, sigma_s=3.0, sigma_r=25.0)

# Save outputs
cv2.imwrite(str(out_dir / "q10_original.png"), img)
cv2.imwrite(str(out_dir / "q10_gaussian.png"), gaussian_result)
cv2.imwrite(str(out_dir / "q10_opencv_bilateral.png"), opencv_bilateral)
cv2.imwrite(str(out_dir / "q10_manual_bilateral.png"), manual_bilateral)

# Display results
plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gaussian_result, cmap="gray")
plt.title("Part (b): Gaussian Blur")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(opencv_bilateral, cmap="gray")
plt.title("Part (c): OpenCV Bilateral Filter")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(manual_bilateral, cmap="gray")
plt.title("Part (d): Manual Bilateral Filter")
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"Saved outputs to: {out_dir}")