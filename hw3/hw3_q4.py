import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "figures"

P = np.array([
    (5, 11), (19, -6), (2, 5), (7, 15), (14, 21),
    (8, 6), (-9, 12), (9, 19), (6, 13), (3, -7),
    (4, 9), (10, 21), (3, 7), (-3, -3), (-9, 13),
    (8, 16), (10, 3), (2, 7), (7, 12), (1, 3),
], dtype=float)

THRESH = 1.5
CONF = 0.99
N_MAX = 100
S = 2
SEED = 0


def tls_fit(pts):
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T, bias=True)
    evals, evecs = np.linalg.eigh(cov)
    a, b = evecs[:, 0]
    if b < 0 or (b == 0 and a < 0):
        a, b = -a, -b
    c = -(a * mean[0] + b * mean[1])
    return np.array([a, b, c]), mean, cov, evals


def line_from_two(p1, p2):
    d = p2 - p1
    n = np.array([-d[1], d[0]]) / np.hypot(*d)
    return np.array([n[0], n[1], -n @ p1])


def distances(line, pts):
    return np.abs(pts @ line[:2] + line[2])


def required_iters(w):
    if w >= 1.0:
        return 1
    return math.ceil(math.log(1 - CONF) / math.log(1 - w ** S))


def ransac(pts, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(pts)
    best = None
    limit = N_MAX
    history = []
    it = 0
    while it < limit:
        it += 1
        i, j = rng.choice(n, size=2, replace=False)
        line = line_from_two(pts[i], pts[j])
        d = distances(line, pts)
        inl = d <= THRESH
        k = int(inl.sum())
        sse = float(np.sum(d[inl] ** 2))
        improved = best is None or k > best["k"]
        tie_better = best is not None and k == best["k"] and sse < best["sse"]
        rec = dict(it=it, idx=(i, j), line=line, inl=inl, k=k, sse=sse,
                   improved=improved, tie=tie_better, limit_before=limit)
        if improved or tie_better:
            best = rec
        if improved:
            w = k / n
            limit = min(required_iters(w), N_MAX)
        rec["limit_after"] = limit
        history.append(rec)
    return best, history


def line_xy(line, xlim):
    a, b, c = line
    if abs(b) > 1e-9:
        x = np.array(xlim)
        return x, -(a * x + c) / b
    return np.full(2, -c / a), np.array([-30, 30])


XLIM = (-12, 22)
YLIM = (-10, 25)


def setup(ax, title):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=10)


def plot_iteration(ax, rec, pts):
    inl = rec["inl"]
    ax.scatter(*pts[~inl].T, c="tab:red", marker="x", s=40, label=f"outliers ({(~inl).sum()})")
    ax.scatter(*pts[inl].T, c="tab:blue", s=30, label=f"inliers ({inl.sum()})")
    i, j = rec["idx"]
    ax.scatter(*pts[[i, j]].T, s=180, facecolors="none", edgecolors="k", linewidths=2,
               label="sampled points")
    ax.plot(*line_xy(rec["line"], XLIM), "g-", lw=1.5, label="candidate line")
    a, b, c = rec["line"]
    for s in (-1, 1):
        ax.plot(*line_xy((a, b, c + s * THRESH), XLIM), "g--", lw=0.8)
    setup(ax, f"Iteration {rec['it']}: {rec['k']} inliers")
    ax.legend(fontsize=7, loc="upper left")


def main():
    OUT.mkdir(exist_ok=True)

    line_a, mean, cov, evals = tls_fit(P)
    print("(a) TLS on all 20 points")
    print(f"  mean = {mean}, covariance =\n{cov}\n  eigenvalues = {evals}")
    print(f"  line: {line_a[0]:.4f} x + {line_a[1]:.4f} y + {line_a[2]:.4f} = 0, "
          f"slope = {-line_a[0] / line_a[1]:.4f}, intercept = {-line_a[2] / line_a[1]:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*P.T, c="k", s=25, label="data")
    ax.plot(*line_xy(line_a, XLIM), "m-", lw=2, label="TLS (all points)")
    setup(ax, "(a) Orthogonal least squares, all 20 points")
    ax.legend()
    fig.savefig(OUT / "q4a_tls_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    best, hist = ransac(P)
    print("\n(b) RANSAC log")
    for r in hist:
        flag = " <- new best" if r["improved"] else (" <- tie, smaller SSE" if r["tie"] else "")
        print(f"  it {r['it']:2d}: pts {r['idx']}, inliers {r['k']:2d}, SSE {r['sse']:.3f}, "
              f"limit {r['limit_after']}{flag}")
    n_in = best["k"]
    print(f"  final: inliers = {n_in}, outliers = {len(P) - n_in}, w = {n_in / len(P):.2f}, "
          f"iterations = {len(hist)}, best found at iteration {best['it']}")
    print(f"  outlier points: {P[~best['inl']].tolist()}")

    shown = [hist[0]]
    for r in hist:
        if r["improved"] and r["it"] not in [s["it"] for s in shown]:
            shown.append(r)
    rest = [r for r in hist if r["it"] not in [s["it"] for s in shown]]
    if rest:
        shown.append(max(rest, key=lambda r: (not r["tie"] and r["k"] < best["k"], -r["k"])))
    if best["it"] not in [s["it"] for s in shown]:
        shown.append(best)
    shown = sorted(shown, key=lambda r: r["it"])
    cols = 2 if len(shown) == 4 else min(len(shown), 3)
    rows = math.ceil(len(shown) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.2 * rows), squeeze=False)
    for ax, r in zip(axes.flat, shown):
        plot_iteration(ax, r, P)
        if r["it"] == best["it"]:
            ax.set_title(ax.get_title() + "  (final best)", fontsize=10, fontweight="bold")
    for ax in axes.flat[len(shown):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "q4b_ransac_iterations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plotted iterations: {[r['it'] for r in shown]}")

    line_c, mean_c, cov_c, evals_c = tls_fit(P[best["inl"]])
    print("\n(c) TLS on RANSAC inliers")
    print(f"  mean = {mean_c}, eigenvalues = {evals_c}")
    print(f"  line: {line_c[0]:.4f} x + {line_c[1]:.4f} y + {line_c[2]:.4f} = 0, "
          f"slope = {-line_c[0] / line_c[1]:.4f}, intercept = {-line_c[2] / line_c[1]:.4f}")
    for name, ln in [("all-points TLS", line_a), ("inlier TLS", line_c)]:
        d_in = distances(ln, P[best["inl"]])
        print(f"  {name}: RMS perpendicular distance of inliers = {np.sqrt(np.mean(d_in ** 2)):.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(*P[best["inl"]].T, c="tab:blue", s=30, label="RANSAC inliers")
    ax.scatter(*P[~best["inl"]].T, c="tab:red", marker="x", s=50, label="RANSAC outliers")
    ax.plot(*line_xy(line_a, XLIM), "m--", lw=2, label="(a) TLS, all points")
    ax.plot(*line_xy(line_c, XLIM), "g-", lw=2, label="(c) TLS, inliers only")
    setup(ax, "(c) TLS fit: all points vs. RANSAC inliers")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "q4c_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
