import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def histogram_equalize(img):
    """
    Perform manual histogram equalization on a grayscale image.
    """
    # Histogram
    hist = np.bincount(img.flatten(), minlength=256)

    # Cumulative distribution function
    cdf = hist.cumsum()

    # First non-zero value in the CDF
    cdf_min = cdf[np.nonzero(cdf)][0]
    total_pixels = img.size

    # Build lookup table
    lut = np.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # Map original pixels through LUT
    equalized = lut[img]

    return equalized


# ---- paths ----
img_path = Path("../data/raw/runway.png")
out_dir = Path("../results/q03")
out_dir.mkdir(parents=True, exist_ok=True)

# ---- load image in grayscale ----
img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")

# ---- apply manual histogram equalization ----
eq_img = histogram_equalize(img)

# ---- save output ----
cv2.imwrite(str(out_dir / "runway_equalized.png"), eq_img)

# ---- display results ----
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(eq_img, cmap="gray")
plt.title("Equalized Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# ---- plot histograms ----
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(img.ravel(), bins=256, range=[0, 256], color="gray")
plt.title("Original Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(eq_img.ravel(), bins=256, range=[0, 256], color="gray")
plt.title("Equalized Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()