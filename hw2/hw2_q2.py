import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io

rng = np.random.default_rng(5639)


def load_image(path="cameraman.tif"):
    img = io.imread(path).astype(np.float64)
    return (img - img.min()) / (img.max() - img.min())


def gaussian_kernel(size, sigma):
    r = size // 2
    x = np.arange(-r, r + 1)
    g = np.exp(-x**2 / (2 * sigma**2))
    k = np.outer(g, g)
    return k / k.sum()


def box_filter(img, size=5):
    return ndimage.uniform_filter(img, size=size, mode="reflect")


def gaussian_filter(img, size=5, sigma=1.0):
    return ndimage.convolve(img, gaussian_kernel(size, sigma), mode="reflect")


def median_filter(img, size=5):
    return ndimage.median_filter(img, size=size, mode="reflect")


def add_gaussian_noise(img, sigma=0.05):
    return img + rng.normal(0.0, sigma, img.shape)


def add_salt_pepper(img, density=0.05):
    noisy = img.copy()
    u = rng.random(img.shape)
    noisy[u < density / 2] = 0.0
    noisy[(u >= density / 2) & (u < density)] = 1.0
    return noisy


def psnr(ref, img):
    mse = np.mean((ref - img) ** 2)
    return 10 * np.log10(1.0 / mse)


def show_row(images, titles, ref, fname, suptitle):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4.4))
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{t}\nPSNR = {psnr(ref, im):.2f} dB", fontsize=11)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(f"figures/{fname}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def show_zoom(images, titles, fname, rows=slice(60, 260), cols=slice(180, 380)):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4.2))
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(np.clip(im[rows, cols], 0, 1), cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"figures/{fname}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    I = load_image()

    In = add_gaussian_noise(I, 0.05)
    res_a = [In, box_filter(In), gaussian_filter(In, 5, 1.0), median_filter(In)]
    titles = ["Noisy (Gaussian, $\\sigma$=0.05)", "5x5 Box", "5x5 Gaussian ($\\sigma$=1)", "5x5 Median"]
    show_row(res_a, titles, I, "q2a_gaussian_noise.png", "Problem 2(a): Gaussian noise")
    show_zoom(res_a, titles, "q2a_zoom.png")

    Isp = add_salt_pepper(I, 0.05)
    print(f"(c) fraction of corrupted pixels: {np.mean(Isp != I):.4f}")
    res_c = [Isp, box_filter(Isp), gaussian_filter(Isp, 5, 1.0), median_filter(Isp)]
    titles = ["Noisy (salt & pepper, d=0.05)", "5x5 Box", "5x5 Gaussian ($\\sigma$=1)", "5x5 Median"]
    show_row(res_c, titles, I, "q2c_salt_pepper.png", "Problem 2(c): Salt-and-pepper noise")
    show_zoom(res_c, titles, "q2c_zoom.png")

    sigmas = [0.6, 1.0, 2.0]
    res_e = [In] + [gaussian_filter(In, 7, s) for s in sigmas]
    titles = ["Noisy (Gaussian, $\\sigma$=0.05)"] + [f"7x7 Gaussian, $\\sigma$={s}" for s in sigmas]
    show_row(res_e, titles, I, "q2e_gaussian_sigma.png", "Problem 2(e): 7x7 Gaussian filter, varying $\\sigma$")
    show_zoom(res_e, titles, "q2e_zoom.png")

    print("\nPSNR (dB) w.r.t. clean image")
    for name, res in [("(a) Gaussian noise", res_a), ("(c) Salt & pepper", res_c)]:
        print(f"  {name}: noisy={psnr(I, res[0]):.2f}, box={psnr(I, res[1]):.2f}, "
              f"gauss={psnr(I, res[2]):.2f}, median={psnr(I, res[3]):.2f}")
    print("  (e) " + ", ".join(f"sigma={s}: {psnr(I, r):.2f}" for s, r in zip(sigmas, res_e[1:])))

    flat = (slice(20, 120), slice(20, 120))
    print("\nResidual std in flat sky patch / mean Sobel gradient magnitude of clean-image edges")
    edge_mask = np.hypot(ndimage.sobel(I, 1), ndimage.sobel(I, 0)) > 0.5
    for name, im in zip(["noisy", "box", "gauss", "median"], res_a):
        g = np.hypot(ndimage.sobel(im, 1), ndimage.sobel(im, 0))
        print(f"  (a) {name:6s}: flat std={np.std(im[flat] - I[flat]):.4f}, edge grad={g[edge_mask].mean():.3f}")
    for s, im in zip(sigmas, res_e[1:]):
        g = np.hypot(ndimage.sobel(im, 1), ndimage.sobel(im, 0))
        print(f"  (e) sigma={s}: flat std={np.std(im[flat] - I[flat]):.4f}, edge grad={g[edge_mask].mean():.3f}")

    print("\nCentre row of normalized 7x7 Gaussian kernels:")
    for s in sigmas:
        k = gaussian_kernel(7, s)
        print(f"  sigma={s}: centre weight={k[3, 3]:.4f}, row = {np.round(k[3] / k[3].sum(), 4)}")


if __name__ == "__main__":
    main()
