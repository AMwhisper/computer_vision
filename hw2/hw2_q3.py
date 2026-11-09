import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, feature

rng = np.random.default_rng(5639)


def load_image(path="cameraman.tif"):
    img = io.imread(path).astype(np.float64)
    return (img - img.min()) / (img.max() - img.min())


def gradients(img, method):
    op = {"sobel": ndimage.sobel, "prewitt": ndimage.prewitt}[method]
    Gx = op(img, axis=1, mode="reflect")
    Gy = op(img, axis=0, mode="reflect")
    return Gx, Gy, np.sqrt(Gx**2 + Gy**2)


def canny(img):
    return feature.canny(img)


def show_row(images, titles, fname, suptitle):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4.4))
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=max(im.max(), 1e-12) if im.dtype != bool else 1)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(f"figures/{fname}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def scale(g):
    return np.clip(g / np.percentile(g, 99.5), 0, 1)


def main():
    I = load_image()

    Gx, Gy, G = gradients(I, "sobel")
    show_row([I, scale(np.abs(Gx)), scale(np.abs(Gy)), scale(G)],
             ["Original", "Sobel $|G_x|$", "Sobel $|G_y|$", "Sobel $G=\\sqrt{G_x^2+G_y^2}$"],
             "q3a_sobel_components.png", "Problem 3(a): Sobel gradient components")

    _, _, Gp = gradients(I, "prewitt")
    E = canny(I)
    show_row([I, scale(Gp), scale(G), E.astype(float)],
             ["Original", "Prewitt magnitude", "Sobel magnitude", "Canny edges (default)"],
             "q3b_clean.png", "Problem 3(b): Edge detectors on the original image")

    In = I + rng.normal(0.0, 0.05, I.shape)
    _, _, Gpn = gradients(In, "prewitt")
    _, _, Gn = gradients(In, "sobel")
    En = canny(In)
    show_row([np.clip(In, 0, 1), scale(Gpn), scale(Gn), En.astype(float)],
             ["Noisy ($\\sigma$=0.05)", "Prewitt magnitude", "Sobel magnitude", "Canny edges (default)"],
             "q3d_noisy.png", "Problem 3(d): Edge detectors on the noisy image")

    flat = (slice(20, 120), slice(20, 120))
    edges = G > 0.5 * G.max()
    print("Flat-region mean gradient / mean strong-edge gradient (lower = cleaner)")
    for name, gc, gn in [("Prewitt", Gp, Gpn), ("Sobel", G, Gn)]:
        print(f"  {name:7s}: clean={gc[flat].mean() / gc[edges].mean():.4f}, "
              f"noisy={gn[flat].mean() / gn[edges].mean():.4f}")
    print(f"Canny edge pixels: clean={E.sum()}, noisy={En.sum()}, "
          f"in flat region clean={E[flat].sum()}, noisy={En[flat].sum()}")
    print(f"Canny: fraction of clean edges preserved in noisy = "
          f"{np.mean(ndimage.binary_dilation(En, iterations=1)[E]):.3f}")


if __name__ == "__main__":
    main()
