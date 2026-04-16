import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting


def gaussian_kernel(size, sigma):
    """
    Generate a normalized Gaussian kernel using NumPy.
    """
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    return kernel


# Base directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Paths
img_path = BASE_DIR / "data" / "raw" / "woman_window.jpg"   # change if needed
out_dir = BASE_DIR / "results" / "q05"
out_dir.mkdir(parents=True, exist_ok=True)

# Load image in grayscale
img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")

# Part (a): 5x5 Gaussian kernel, sigma=2
kernel_5x5 = gaussian_kernel(5, 2)

print("Normalized 5x5 Gaussian Kernel (sigma = 2):")
print(kernel_5x5)
print("Kernel sum:", kernel_5x5.sum())

# Part (b): 51x51 Gaussian kernel for 3D visualization
kernel_51x51 = gaussian_kernel(51, 2)

x = np.arange(-(51 // 2), 51 // 2 + 1)
y = np.arange(-(51 // 2), 51 // 2 + 1)
xx, yy = np.meshgrid(x, y)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(xx, yy, kernel_51x51, cmap="viridis")

ax.set_title("3D Surface Plot of 51x51 Gaussian Kernel (sigma = 2)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Kernel Value")

plt.tight_layout()
plt.show()

# Part (c): Manual Gaussian smoothing using filter2D
smoothed_manual = cv2.filter2D(img, -1, kernel_5x5)

# Part (d): OpenCV GaussianBlur
smoothed_cv = cv2.GaussianBlur(img, (5, 5), sigmaX=2, sigmaY=2)

# Save results
cv2.imwrite(str(out_dir / "q05_original_gray.png"), img)
cv2.imwrite(str(out_dir / "q05_manual_gaussian.png"), smoothed_manual)
cv2.imwrite(str(out_dir / "q05_opencv_gaussian.png"), smoothed_cv)

# Display images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Grayscale")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(smoothed_manual, cmap="gray")
plt.title("Manual Gaussian Smoothing")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(smoothed_cv, cmap="gray")
plt.title("OpenCV GaussianBlur")
plt.axis("off")

plt.tight_layout()
plt.show()

# Difference between manual and OpenCV results
difference = np.abs(smoothed_manual.astype(np.int16) - smoothed_cv.astype(np.int16))

print("Maximum pixel difference:", difference.max())
print("Mean pixel difference:", difference.mean())

plt.figure(figsize=(6, 6))
plt.imshow(difference, cmap="gray")
plt.title("Absolute Difference: Manual vs OpenCV")
plt.axis("off")
plt.show()

print(f"Saved outputs to: {out_dir}")