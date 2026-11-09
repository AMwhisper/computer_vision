"""Q4: DLT homography, outlier injection, adaptive RANSAC, warping and MSE.

Run:  python q4_homography.py            (uses the 20 recorded correspondences)
      python q4_homography.py --pick     (re-select 20 points interactively with ginput)
"""
import os, sys
import numpy as np
import cv2
import matplotlib
if "--pick" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "figs")
I1 = cv2.imread(os.path.join(ROOT, "001.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float64)
I2 = cv2.imread(os.path.join(ROOT, "002.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float64)

# (u1, v1) <-> (u2, v2); u = column, v = row (pixels).
# Points 1-4 are the black markers; the rest are window / sill corners.
PTS = np.array([
    [307.1, 233.1, 245.0, 228.2], [407.0, 230.1, 351.0, 229.7],
    [310.1, 280.9, 245.0, 268.8], [409.7, 274.2, 350.9, 268.7],
    [ 59.0, 128.0,  40.0, 151.0], [268.0,  20.0, 217.0,  51.0],
    [180.0, 264.0, 130.0, 250.0], [209.0, 168.0, 157.0, 178.0],
    [311.0, 142.0, 254.0, 153.0], [246.0, 107.0, 192.0, 127.0],
    [154.0, 134.0, 112.0, 152.0], [435.0, 117.0, 396.0, 124.0],
    [312.0,  79.0, 258.0,  98.0], [380.0, 146.0, 328.0, 154.0],
    [251.0, 234.0, 192.0, 230.0], [348.0,  43.0, 299.0,  62.0],
    [ 84.0,  87.0,  58.0, 120.0], [337.0, 197.0, 277.0, 200.0],
    [ 95.0, 251.0,  66.0, 239.0], [266.0, 279.0, 204.0, 265.0]])


def pick_points(n=20):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(I1, cmap="gray"); ax[1].imshow(I2, cmap="gray")
    out = []
    for i in range(n):
        fig.suptitle(f"Point {i + 1}/{n}: click in I1 (left) then I2 (right)")
        (p,) = plt.ginput(1, timeout=0); ax[0].plot(*p, "r+")
        (q,) = plt.ginput(1, timeout=0); ax[1].plot(*q, "r+")
        out.append([*p, *q])
    plt.close(fig)
    return np.array(out)


# ---------------------------------------------------------------- DLT
def build_A(p1, p2):
    A = []
    for (u1, v1), (u2, v2) in zip(p1, p2):
        A.append([u1, v1, 1, 0, 0, 0, -u2 * u1, -u2 * v1, -u2])
        A.append([0, 0, 0, u1, v1, 1, -v2 * u1, -v2 * v1, -v2])
    return np.array(A)


def dlt(p1, p2):
    """min ||A h||^2 s.t. ||h|| = 1  ->  right singular vector of smallest singular value."""
    _, _, Vt = np.linalg.svd(build_A(p1, p2))
    H = Vt[-1].reshape(3, 3)
    return H / np.linalg.norm(H) * np.sign(H[2, 2])   # sum h_ij^2 = 1


def project(H, p):
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:]


def reproj_err(H, p1, p2):
    return np.linalg.norm(project(H, p1) - p2, axis=1)


# ---------------------------------------------------------------- RANSAC
def collinear3(p):
    for i in range(4):
        q = np.delete(p, i, axis=0)
        if abs(np.cross(q[1] - q[0], q[2] - q[0])) < 1e-6:
            return True
    return False


def ransac(p1, p2, s=4, tau=3.0, p=0.99, rng=None, max_iter=10000):
    """Adaptive RANSAC: N is updated from the current best inlier ratio."""
    rng = np.random.default_rng() if rng is None else rng
    n = len(p1)
    N, it = np.inf, 0
    best_in = np.zeros(n, bool)
    while it < N and it < max_iter:
        idx = rng.choice(n, s, replace=False)
        it += 1
        if collinear3(p1[idx]) or collinear3(p2[idx]):
            continue
        H = dlt(p1[idx], p2[idx])
        inl = reproj_err(H, p1, p2) < tau
        if inl.sum() > best_in.sum():
            best_in = inl
            eps = 1 - inl.sum() / n                       # outlier ratio
            N = 0 if eps == 0 else np.log(1 - p) / np.log(1 - (1 - eps) ** s)
    H = dlt(p1[best_in], p2[best_in])                     # re-estimate on all inliers
    return H, best_in, it


# ---------------------------------------------------------------- warping / MSE
def warp(I, H, shape):
    return cv2.warpPerspective(I, H, (shape[1], shape[0]), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def mse(A, B):
    R = (slice(100, 251), slice(100, 351))                 # v in [100,250], u in [100,350]
    return np.mean((A[R] - B[R]) ** 2)


def show_pairs(p1, p2, colors, fname, labels=None, title=None, legend=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].imshow(I1, cmap="gray"); ax[1].imshow(I2, cmap="gray")
    for i in range(len(p1)):
        for a, q in ((ax[0], p1[i]), (ax[1], p2[i])):
            a.plot(*q, "o", ms=7, mfc="none", mec=colors[i], mew=1.8)
            a.text(q[0] + 4, q[1] - 4, str(i + 1), color=colors[i], fontsize=8, weight="bold")
    ax[0].set_title("$I_1$"); ax[1].set_title("$I_2$")
    for a in ax: a.axis("off")
    if legend:
        for c, l in legend: ax[1].plot([], [], "o", mfc="none", mec=c, mew=1.8, label=l)
        ax[1].legend(loc="lower right", fontsize=8)
    if title: fig.suptitle(title)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, fname), dpi=150); plt.close(fig)


if __name__ == "__main__":
    np.set_printoptions(suppress=False, precision=5, linewidth=120)
    pts = pick_points() if "--pick" in sys.argv else PTS
    p1, p2 = pts[:, :2], pts[:, 2:]
    n = len(p1)

    # (a)
    cmap = plt.get_cmap("tab20")
    show_pairs(p1, p2, [cmap(i) for i in range(n)], "q4a_corr.png")

    # (b)
    H_dlt = dlt(p1, p2)
    e = reproj_err(H_dlt, p1, p2)
    print("H_DLT =\n", H_dlt, "\nreproj err: mean %.3f  max %.3f" % (e.mean(), e.max()))

    # (c)(i) replace 25 % of targets with random pixels in I2
    rng = np.random.default_rng(5639)
    out_idx = np.sort(rng.choice(n, n // 4, replace=False))
    p2o = p2.copy()
    p2o[out_idx] = np.c_[rng.uniform(0, I2.shape[1] - 1, len(out_idx)),
                         rng.uniform(0, I2.shape[0] - 1, len(out_idx))]
    print("outlier indices (1-based):", out_idx + 1)
    print("random targets:\n", np.round(p2o[out_idx], 1))
    col = ["red" if i in out_idx else "lime" for i in range(n)]
    show_pairs(p1, p2o, col, "q4c_outliers.png",
               legend=[("lime", "original"), ("red", "injected outlier")])

    # (c)(ii)
    H_out = dlt(p1, p2o)
    print("H_outlier =\n", H_out)

    # (c)(iii)-(iv)
    H_rs, inl, iters = ransac(p1, p2o, rng=np.random.default_rng(0))
    print("RANSAC iterations:", iters, " inliers:", inl.sum(), " outliers:", n - inl.sum())
    print("RANSAC outlier indices (1-based):", np.where(~inl)[0] + 1)
    print("H_RANSAC =\n", H_rs)
    show_pairs(p1, p2o, ["lime" if b else "red" for b in inl], "q4c_ransac.png",
               legend=[("lime", "RANSAC inlier"), ("red", "RANSAC outlier")])

    # (d)
    W = {k: warp(I1, H, I2.shape) for k, H in
         [("DLT", H_dlt), ("outlier", H_out), ("RANSAC", H_rs)]}
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for a, (t, im) in zip(ax, [("Ground truth $I_2$", I2), (r"$\hat I_{2,\mathrm{DLT}}$", W["DLT"]),
                               (r"$\hat I_{2,\mathrm{outlier}}$", W["outlier"]),
                               (r"$\hat I_{2,\mathrm{RANSAC}}$", W["RANSAC"])]):
        a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(t)
        a.add_patch(plt.Rectangle((100, 100), 250, 150, fill=False, ec="yellow", lw=1))
        a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "q4d_warps.png"), dpi=150); plt.close(fig)
    for k in W:
        print("MSE %-8s %.2f" % (k, mse(I2, W[k])))
