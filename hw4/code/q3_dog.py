"""Q3: Gaussian / DoG scale space and 26-neighbour extremum search for I1 and I3."""
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "..", "figs")
I1 = np.array([[0, 0, 0, 0, 0], [0, 0, 200, 0, 0], [0, 200, 200, 200, 0],
               [200] * 5, [0] * 5], float)
I3 = np.kron(I1, np.ones((5, 5)))

J = 30                                     # sigma_0 ... sigma_29 (0.5 ... 6.17)
sigmas = 0.5 * 2 ** (np.arange(J) / 8)

def scale_space(I):
    L = np.stack([gaussian_filter(I, s, mode="reflect", truncate=4.0) for s in sigmas])
    D = L[1:] - L[:-1]                     # D_j = L(sigma_{j+1}) - L(sigma_j)
    return L, D

def find_extrema(D, thr=0.0):
    """Strict 26-neighbour extrema; scale index j in 1..J-3, all spatial pixels
    that have a full 3x3 neighbourhood (border pixels excluded)."""
    out = []
    S, H, W = D.shape
    for j in range(1, S - 1):
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                v = D[j, r, c]
                if abs(v) <= thr:
                    continue
                cube = D[j - 1:j + 2, r - 1:r + 2, c - 1:c + 2].ravel()
                others = np.delete(cube, 13)
                if v > others.max() or v < others.min():
                    out.append((r + 1, c + 1, j, sigmas[j], v,
                                "max" if v > 0 else "min"))
    return out

if __name__ == "__main__":
    res = {}
    for name, I in [("I1", I1), ("I3", I3)]:
        L, D = scale_space(I)
        ext = find_extrema(D, thr=1e-6)
        res[name] = (L, D, ext)
        print(f"\n{name}: {len(ext)} extrema  (row, col one-based; DoG index j; sigma_j; D)")
        for e in ext:
            print("  (%2d,%2d)  j=%2d  sigma=%.3f  D=%+.3f  %s" % e)

    # tip-pixel DoG responses vs scale
    _, D1, _ = res["I1"]; _, D3, _ = res["I3"]
    print("\nDoG at tip vs sigma_j")
    for j in range(J - 1):
        print("j=%2d sigma=%.3f  D1(2,3)=%+8.3f  D3(8,13)=%+8.3f" % (j, sigmas[j], D1[j, 1, 2], D3[j, 7, 12]))

    show = [0, 8, 16, 24, 29]
    fig, ax = plt.subplots(2, len(show), figsize=(12, 5))
    for i, j in enumerate(show):
        for k, n in enumerate(["I1", "I3"]):
            ax[k, i].imshow(res[n][0][j], cmap="gray", vmin=0, vmax=200)
            ax[k, i].set_title(f"{n}, $\\sigma$={sigmas[j]:.2f}", fontsize=9); ax[k, i].axis("off")
    plt.tight_layout(); plt.savefig(f"{FIG}/q3_gauss.png", dpi=150); plt.close()

    show = [0, 6, 12, 18, 24, 28]
    fig, ax = plt.subplots(2, len(show), figsize=(13, 4.8))
    for i, j in enumerate(show):
        for k, n in enumerate(["I1", "I3"]):
            Dj = res[n][1][j]; m = np.abs(Dj).max() + 1e-9
            ax[k, i].imshow(Dj, cmap="seismic", vmin=-m, vmax=m)
            ax[k, i].set_title(f"{n}, $D_{{{j}}}$ ($\\sigma_j$={sigmas[j]:.2f})", fontsize=8); ax[k, i].axis("off")
    plt.tight_layout(); plt.savefig(f"{FIG}/q3_dog.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(sigmas[:-1], D1[:, 1, 2], "o-", ms=3, label="I1 at P=(2,3)")
    ax.plot(sigmas[:-1], D3[:, 7, 12], "s-", ms=3, label="I3 at P'=(8,13)")
    ax.set_xscale("log"); ax.set_xlabel(r"$\sigma_j$"); ax.set_ylabel(r"$D_j$ at tip"); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/q3_tip_curve.png", dpi=150); plt.close()
