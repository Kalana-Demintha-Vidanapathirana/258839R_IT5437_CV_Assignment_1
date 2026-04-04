from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def gamma_transform(image_norm: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to a normalized image in [0, 1]."""
    return np.power(image_norm, gamma)


def to_uint8(image_norm: np.ndarray) -> np.ndarray:
    """Convert normalized image in [0, 1] to uint8 in [0, 255]."""
    return np.clip(image_norm * 255, 0, 255).astype(np.uint8)


def main():
    input_path = Path(r"D:\MSc in AI\1YQ3\CV\Assignment_1\data\raw\highlights_and_shadows.jpg")
    output_dir = Path(r"D:\MSc in AI\1YQ3\CV\Assignment_1\results\q02")
    output_dir.mkdir(parents=True, exist_ok=True)

    gamma = 0.8

    img_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L_original, a, b = cv2.split(img_lab)

    L_norm = L_original.astype(np.float32) / 255.0
    L_corrected_norm = gamma_transform(L_norm, gamma)
    L_corrected = to_uint8(L_corrected_norm)

    img_lab_corrected = cv2.merge((L_corrected, a, b))
    img_bgr_corrected = cv2.cvtColor(img_lab_corrected, cv2.COLOR_LAB2BGR)
    img_corrected_rgb = cv2.cvtColor(img_bgr_corrected, cv2.COLOR_BGR2RGB)

    cv2.imwrite(str(output_dir / "original_bgr.png"), img_bgr)
    cv2.imwrite(str(output_dir / "corrected_bgr.png"), img_bgr_corrected)
    cv2.imwrite(str(output_dir / "original_L.png"), L_original)
    cv2.imwrite(str(output_dir / "corrected_L.png"), L_corrected)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original Color Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_corrected_rgb)
    axes[0, 1].set_title(f"Gamma Corrected in L Channel (γ = {gamma})")
    axes[0, 1].axis("off")

    axes[1, 0].hist(L_original.ravel(), bins=256, range=(0, 256))
    axes[1, 0].set_title("Histogram of Original L Channel")
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].hist(L_corrected.ravel(), bins=256, range=(0, 256))
    axes[1, 1].set_title(f"Histogram of Corrected L Channel (γ = {gamma})")
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(output_dir / "q02_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Q2 - LAB Gamma Correction")
    print("Image: highlights_and_shadows.jpg")
    print(f"Gamma value (γ): {gamma}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()