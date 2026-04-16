import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def equalize_foreground_only(gray, mask):
    """
    Apply histogram equalization only to the foreground region.
    mask should be a binary image with foreground = 255, background = 0
    """
    result = gray.copy()

    fg_pixels = gray[mask == 255]

    if fg_pixels.size == 0:
        return result

    # Histogram of foreground only
    hist = np.bincount(fg_pixels, minlength=256)

    # CDF
    cdf = hist.cumsum()
    nonzero = np.nonzero(cdf)[0]

    if len(nonzero) == 0:
        return result

    cdf_min = cdf[nonzero[0]]
    total_fg = fg_pixels.size

    if total_fg == cdf_min:
        return result

    # LUT
    lut = np.round((cdf - cdf_min) / (total_fg - cdf_min) * 255)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # Apply only on foreground
    result[mask == 255] = lut[gray[mask == 255]]

    return result


# Base directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Paths
img_path = BASE_DIR / "data" / "raw" / "woman_window.jpg"   # change filename if needed
out_dir = BASE_DIR / "results" / "q04"
out_dir.mkdir(parents=True, exist_ok=True)

# Load image
img_bgr = cv2.imread(str(img_path))

if img_bgr is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")

# Convert to grayscale
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Otsu thresholding
otsu_thresh, mask = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# We want the woman + room as foreground.
# If Otsu selects the bright outside instead, invert the mask.
# Check which side is darker on average.
mean_fg = gray[mask == 255].mean() if np.any(mask == 255) else 255
mean_bg = gray[mask == 0].mean() if np.any(mask == 0) else 0

if mean_fg > mean_bg:
    mask = cv2.bitwise_not(mask)

# Equalize only foreground
eq_foreground = equalize_foreground_only(gray, mask)

# Save outputs
cv2.imwrite(str(out_dir / "q04_gray.png"), gray)
cv2.imwrite(str(out_dir / "q04_otsu_mask.png"), mask)
cv2.imwrite(str(out_dir / "q04_foreground_equalized.png"), eq_foreground)

# Display
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title(f"Otsu Foreground Mask\nThreshold = {otsu_thresh:.2f}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(eq_foreground, cmap="gray")
plt.title("Foreground Equalized")
plt.axis("off")

plt.tight_layout()
plt.show()

# Histograms
fg_original = gray[mask == 255]
fg_equalized = eq_foreground[mask == 255]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(fg_original, bins=256, range=[0, 256], color="gray")
plt.title("Foreground Histogram (Original)")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(fg_equalized, bins=256, range=[0, 256], color="gray")
plt.title("Foreground Histogram (Equalized)")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

print(f"Otsu threshold value: {otsu_thresh:.2f}")
print(f"Saved grayscale image to: {out_dir / 'q04_gray.png'}")
print(f"Saved mask to: {out_dir / 'q04_otsu_mask.png'}")
print(f"Saved equalized result to: {out_dir / 'q04_foreground_equalized.png'}")