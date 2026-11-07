"""
CCA step by step on a tiny example you can follow by hand.

3 photos, each only 4 pixels, reduced to 1 number per photo (d = 1).
Prints every matrix and every move, so you can see what "training" does.

Run (from the repository root):  python examples/cca_toy_step_by_step.py
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

np.set_printoptions(precision=3, suppress=True)
names = ["A", "B", "C"]

# 1. The data: one row per photo, one column per pixel (brightness 0-1)
X = np.array([[0.2, 0.4, 0.6, 0.8],    # photo A
              [0.3, 0.5, 0.6, 0.9],    # photo B (looks like A)
              [0.9, 0.7, 0.1, 0.2]])   # photo C (very different)
print("STEP 1. Data matrix X (3 photos x 4 pixels):\n", X, "\n")

# 2. Real distances between every pair of photos (computed ONCE, never change)
Dx = squareform(pdist(X))
print("STEP 2. Real distance matrix Dx (3 x 3), Dx[i, j] = distance between photo i and photo j:\n", Dx, "\n")

# 3. Starting map: 1 number per photo (here: a simple guess)
Y = np.array([0.0, 0.1, 0.2])
print("STEP 3. Starting map Y (1 number per photo):", Y)
print("        Map distances now: A-B =", round(abs(Y[0]-Y[1]), 3), " A-C =", round(abs(Y[0]-Y[2]), 3),
      " B-C =", round(abs(Y[1]-Y[2]), 3), "\n")

# 4. Training = repeat: pick a photo i, keep it still, move every other photo j
alpha, lam = 0.5, 10.0          # big step, big circle (everything counts), like epoch 1
for epoch in range(1, 4):
    print(f"=========== EPOCH {epoch}  (alpha = {alpha}, lambda = {lam}) ===========")
    for i in range(3):
        print(f"  Hold photo {names[i]} still (Y = {Y[i]:.3f}). Move the others:")
        for j in range(3):
            if j == i:
                continue
            dy = abs(Y[j] - Y[i])                       # distance on the map now
            if dy >= lam:
                print(f"    {names[j]}: outside the circle, not moved"); continue
            direction = np.sign(Y[j] - Y[i]) or 1.0     # which side of i is j on
            move = alpha * (Dx[i, j] - dy) * direction  # = alpha*(Dx-Dy)/Dy*(y_j - y_i)
            print(f"    {names[j]}: real dist {Dx[i, j]:.3f}, map dist {dy:.3f} -> "
                  f"{'too close, push away' if Dx[i, j] > dy else 'too far, pull in'}: "
                  f"move by {move:+.3f}  ({Y[j]:.3f} -> {Y[j] + move:.3f})")
            Y[j] += move
    print("  Map after this epoch Y =", Y)
    print("  Map distances: A-B =", round(abs(Y[0]-Y[1]), 3), "(real", round(Dx[0, 1], 3), ")",
          " A-C =", round(abs(Y[0]-Y[2]), 3), "(real", round(Dx[0, 2], 3), ")",
          " B-C =", round(abs(Y[1]-Y[2]), 3), "(real", round(Dx[1, 2], 3), ")\n")
    alpha /= 2                     # smaller steps each epoch

print("RESULT: each photo is now ONE number instead of 4 pixels.")
print("A and B (similar photos) are close on the map, C (different photo) is far away.")
print("That is all 'training' means: moving these numbers until map distances match real distances.")
