import numpy as np
import matplotlib.pyplot as plt

points = np.load("/home/shourya/rl/scripts/ee_reachable_points.npy")

plt.figure(figsize=(6,6))
plt.scatter(points[:,0], points[:,1], s=1, alpha=0.3)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Top-down view of reachable workspace (X-Y)")
plt.axis("equal")
plt.savefig("/home/shourya/rl/scripts/workspace_topdown.png")
print("Saved plot to workspace_topdown.png")
