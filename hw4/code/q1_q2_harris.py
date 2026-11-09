"""Q1 & Q2: Harris response computations."""
import numpy as np
np.set_printoptions(suppress=True, precision=4)
k = 0.04

def harris(M):
    return np.linalg.det(M) - k * np.trace(M) ** 2

# ---------- Q1 ----------
s3 = np.sqrt(3)
Ms = {'M1': [[1, 0], [0, 1]], 'M2': [[100, 0], [0, 1]], 'M3': [[1, 0], [0, 100]],
      'M4': [[100, 0], [0, 50]], 'M5': [[87.5, 12.5 * s3], [12.5 * s3, 62.5]]}
print("Q1")
for n, M in Ms.items():
    M = np.array(M, float)
    w, V = np.linalg.eigh(M)
    print(n, "det=%.2f tr=%.2f R=%.2f" % (np.linalg.det(M), np.trace(M), harris(M)),
          "eig=", w, "vecs(cols)=\n", V)

# ---------- Q2 ----------
w = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0

def grads(I):
    P = np.pad(I.astype(float), 1)               # zero padding
    Ix = (P[1:-1, 2:] - P[1:-1, :-2]) / 2
    Iy = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2
    return Ix, Iy

def second_moment(I, r, c):                       # r, c one-based
    Ix, Iy = grads(I)
    sl = (slice(r - 2, r + 1), slice(c - 2, c + 1))
    gx, gy = Ix[sl], Iy[sl]
    M = np.array([[np.sum(w * gx * gx), np.sum(w * gx * gy)],
                  [np.sum(w * gx * gy), np.sum(w * gy * gy)]])
    return gx, gy, M

I1 = np.array([[0, 0, 0, 0, 0], [0, 0, 200, 0, 0], [0, 200, 200, 200, 0],
               [200] * 5, [0] * 5])
I2 = I1 // 2
I3 = np.kron(I1, np.ones((5, 5), int))

for name, I, (r, c) in [('I1', I1, (2, 3)), ('I2', I2, (2, 3)), ('I3', I3, (8, 13))]:
    gx, gy, M = second_moment(I, r, c)
    print("\nQ2", name, "Ix=\n", gx, "\nIy=\n", gy, "\nM=\n", M,
          "\ndet=%.2f tr=%.2f R=%.2f" % (np.linalg.det(M), np.trace(M), harris(M)),
          "eig=", np.linalg.eigvalsh(M))
