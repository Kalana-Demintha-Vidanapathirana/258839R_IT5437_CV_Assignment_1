import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def homomorphic_filter(img, gamma_l=0.6, gamma_h=1.8, c=1.0, d0=30):
    """
    Apply homomorphic filtering to a grayscale image.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image.
    gamma_l : float
        Low-frequency gain. Should be less than 1 to suppress illumination.
    gamma_h : float
        High-frequency gain. Should be greater than 1 to enhance reflectance.
    c : float
        Controls the sharpness of the filter transition.
    d0 : float
        Cutoff frequency.

    Returns
    -------
    filtered_img : np.ndarray
        Homomorphic filtered output image.
    h : np.ndarray
        Frequency-domain homomorphic filter.
    """
    img = img.astype(np.float32)

    # Log transform
    img_log = np.log1p(img)

    # FFT and shift
    img_fft = np.fft.fft2(img_log)
    img_fft_shift = np.fft.fftshift(img_fft)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # Distance matrix
    y, x = np.ogrid[:rows, :cols]
    d2 = (y - crow) ** 2 + (x - ccol) ** 2

    # Homomorphic high-pass filter
    h = (gamma_h - gamma_l) * (1 - np.exp(-c * d2 / (d0 ** 2))) + gamma_l

    # Apply filter in frequency domain
    filtered_shift = h * img_fft_shift

    # Inverse FFT
    filtered_ishift = np.fft.ifftshift(filtered_shift)
    img_ifft = np.fft.ifft2(filtered_ishift)

    # Inverse log transform
    img_exp = np.expm1(np.real(img_ifft))

    # Normalize to displayable range
    filtered_img = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)
    return filtered_img.astype(np.uint8), h


def main():
    # Base directory = project root
    base_dir = Path(__file__).resolve().parents[1]

    # Input/output paths
    img_path = base_dir / "data" / "raw" / "woman_window.jpg"
    out_dir = base_dir / "results" / "q12"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load grayscale image
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    # Apply homomorphic filtering
    filtered_img, H = homomorphic_filter(
        img,
        gamma_l=0.6,
        gamma_h=1.8,
        c=1.0,
        d0=30
    )

    # Save outputs
    cv2.imwrite(str(out_dir / "q12_original_gray.png"), img)
    cv2.imwrite(str(out_dir / "q12_homomorphic_filtered.png"), filtered_img)

    # Display original and filtered image
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_img, cmap="gray")
    plt.title("Homomorphic Filtering Result")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Display the filter
    plt.figure(figsize=(6, 6))
    plt.imshow(H, cmap="gray")
    plt.title("Homomorphic Filter H(u,v)")
    plt.axis("off")
    plt.show()

    print(f"Saved original image to: {out_dir / 'q12_original_gray.png'}")
    print(f"Saved filtered image to: {out_dir / 'q12_homomorphic_filtered.png'}")


if __name__ == "__main__":
    main()