from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def gamma_transform(image_norm: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(image_norm, gamma)


def contrast_stretch(image_norm: np.ndarray, r1: float = 0.2, r2: float = 0.8) -> np.ndarray:
    out = np.zeros_like(image_norm, dtype=np.float32)
    out[image_norm < r1] = 0.0

    mask = (image_norm >= r1) & (image_norm <= r2)
    out[mask] = (image_norm[mask] - r1) / (r2 - r1)

    out[image_norm > r2] = 1.0
    return out


def to_uint8(image_norm: np.ndarray) -> np.ndarray:
    return np.clip(image_norm * 255, 0, 255).astype(np.uint8)


def main():
    input_path = Path(r"D:\MSc in AI\1YQ3\CV\Assignment_1\data\raw\runway.png")
    output_dir = Path(r"D:\MSc in AI\1YQ3\CV\Assignment_1\results\q01")
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    img_norm = img.astype(np.float32) / 255.0

    gamma_05 = to_uint8(gamma_transform(img_norm, 0.5))
    gamma_2 = to_uint8(gamma_transform(img_norm, 2.0))
    contrast_img = to_uint8(contrast_stretch(img_norm, 0.2, 0.8))

    cv2.imwrite(str(output_dir / "original.png"), img)
    cv2.imwrite(str(output_dir / "gamma_0_5.png"), gamma_05)
    cv2.imwrite(str(output_dir / "gamma_2_0.png"), gamma_2)
    cv2.imwrite(str(output_dir / "contrast_stretch.png"), contrast_img)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gamma_05, cmap="gray")
    axes[0, 1].set_title("Gamma Correction (γ = 0.5)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(gamma_2, cmap="gray")
    axes[1, 0].set_title("Gamma Correction (γ = 2)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(contrast_img, cmap="gray")
    axes[1, 1].set_title("Contrast Stretching")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()