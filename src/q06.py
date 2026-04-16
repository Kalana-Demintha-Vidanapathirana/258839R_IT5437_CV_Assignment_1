import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


def gaussian(size, sigma):
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    g = (1 / (2 * np.pi * sigma**2)) * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return g, xx, yy


def dog_kernels(size, sigma):
    """
    Compute derivative-of-Gaussian kernels in x and y directions.
    """
    g, xx, yy = gaussian(size, sigma)

    gx = -(xx / sigma**2) * g
    gy = -(yy / sigma**2) * g

    # Normalize by sum of absolute values for stable comparison
    gx = gx / np.sum(np.abs(gx))
    gy = gy / np.sum(np.abs(gy))

    return gx, gy, xx, yy


# Base directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Paths
img_path = BASE_DIR / "data" / "raw" / "rice.png"  
out_dir = BASE_DIR / "results" / "q06"
out_dir.mkdir(parents=True, exist_ok=True)

# Load grayscale image
img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")

# (b) 5x5 DoG kernels for sigma=2
gx_5, gy_5, _, _ = dog_kernels(5, 2)

print("5x5 Derivative of Gaussian Kernel in x-direction (sigma=2):")
print(gx_5)
print()
print("5x5 Derivative of Gaussian Kernel in y-direction (sigma=2):")
print(gy_5)

# (c) 51x51 DoG kernel for 3D visualization
gx_51, gy_51, xx_51, yy_51 = dog_kernels(51, 2)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(xx_51, yy_51, gx_51, cmap="viridis")

ax.set_title("3D Surface Plot of 51x51 DoG Kernel (x-direction, sigma=2)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Kernel Value")

plt.tight_layout()
plt.show()

# (d) Apply DoG kernels
grad_x = cv2.filter2D(img.astype(np.float32), -1, gx_5)
grad_y = cv2.filter2D(img.astype(np.float32), -1, gy_5)

grad_mag = np.sqrt(grad_x**2 + grad_y**2)

# Normalize for display/save
grad_x_disp = cv2.normalize(np.abs(grad_x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
grad_y_disp = cv2.normalize(np.abs(grad_y), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
grad_mag_disp = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# (e) Sobel comparison
sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

sobel_x_disp = cv2.normalize(np.abs(sobel_x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
sobel_y_disp = cv2.normalize(np.abs(sobel_y), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
sobel_mag_disp = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Save outputs
cv2.imwrite(str(out_dir / "q06_grad_x_dog.png"), grad_x_disp)
cv2.imwrite(str(out_dir / "q06_grad_y_dog.png"), grad_y_disp)
cv2.imwrite(str(out_dir / "q06_grad_mag_dog.png"), grad_mag_disp)
cv2.imwrite(str(out_dir / "q06_sobel_x.png"), sobel_x_disp)
cv2.imwrite(str(out_dir / "q06_sobel_y.png"), sobel_y_disp)
cv2.imwrite(str(out_dir / "q06_sobel_mag.png"), sobel_mag_disp)

# Display results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Grayscale")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(grad_x_disp, cmap="gray")
plt.title("DoG Gradient X")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(grad_y_disp, cmap="gray")
plt.title("DoG Gradient Y")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(grad_mag_disp, cmap="gray")
plt.title("DoG Gradient Magnitude")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(sobel_x_disp, cmap="gray")
plt.title("Sobel X")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(sobel_y_disp, cmap="gray")
plt.title("Sobel Y")
plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(grad_mag_disp, cmap="gray")
plt.title("DoG Gradient Magnitude")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sobel_mag_disp, cmap="gray")
plt.title("Sobel Gradient Magnitude")
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"Saved outputs to: {out_dir}")