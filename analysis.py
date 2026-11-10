import os, random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from config import DATA_DIR, OUT_DIR

DATA = DATA_DIR / "train"


def load_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

def fft_spectrum(img):
    f = np.fft.fftshift(np.fft.fft2(img))
    return np.log1p(np.abs(f))

def noise_residual(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    return img - blur

def lbp_hist(img, P=8, R=1):
    lbp = local_binary_pattern(img.astype(np.uint8), P, R, method="uniform")
    hist, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2), density=True)
    return hist

def sample_paths(cls, n=500):
    folder = DATA / cls
    files = random.sample(os.listdir(folder), n)
    return [str(folder / f) for f in files]

if __name__ == "__main__":
    results = {}
    for cls in ["REAL", "FAKE"]:
        imgs = [load_gray(p) for p in sample_paths(cls)]
        results[cls] = {
            "fft": np.mean([fft_spectrum(i) for i in imgs], axis=0),
            "noise_std": [noise_residual(i).std() for i in imgs],
            "lbp": np.mean([lbp_hist(i) for i in imgs], axis=0),
        }

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for r, cls in enumerate(["REAL", "FAKE"]):
        ax[r, 0].imshow(results[cls]["fft"], cmap="inferno")
        ax[r, 0].set_title(f"{cls}: mean FFT spectrum")
        ax[r, 1].hist(results[cls]["noise_std"], bins=30)
        ax[r, 1].set_title(f"{cls}: noise residual std")
        ax[r, 2].bar(range(10), results[cls]["lbp"])
        ax[r, 2].set_title(f"{cls}: LBP texture histogram")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "analysis.png", dpi=150)
    plt.show()