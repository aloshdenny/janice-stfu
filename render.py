import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# labels and values from your similarity calculation

labels = list(similarities.keys())
values = np.array(list(similarities.values()))

N = len(labels)

# --------------------------------------------------
# Close loop
# --------------------------------------------------

angles = np.linspace(
    0,
    2*np.pi,
    N,
    endpoint=False
)

angles_closed = np.r_[angles, angles[0]]
values_closed = np.r_[values, values[0]]

# --------------------------------------------------
# Smooth curve
# --------------------------------------------------

cs = CubicSpline(
    angles_closed,
    values_closed,
    bc_type="periodic"
)

angles_smooth = np.linspace(
    0,
    2*np.pi,
    1000
)

values_smooth = cs(angles_smooth)

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(
    figsize=(14,14),
    subplot_kw={"polar": True}
)

# smoother outline

ax.plot(
    angles_smooth,
    values_smooth,
    linewidth=3
)

ax.fill(
    angles_smooth,
    values_smooth,
    alpha=0.25
)

# --------------------------------------------------
# Category labels
# --------------------------------------------------

ax.set_xticks(angles)
ax.set_xticklabels([])

for angle, label in zip(angles, labels):

    rotation = np.degrees(angle)

    ax.text(
        angle,
        1.12,                 # outside outer ring
        label,
        fontsize=13,
        ha="center",
        va="center",
        rotation=rotation,
        rotation_mode="anchor"
    )

# --------------------------------------------------
# Radial grid
# --------------------------------------------------

ax.set_ylim(0, 1.15)

ax.set_yticks(np.linspace(0.1, 1.0, 10))

ax.set_yticklabels([
    f"{x:.1f}"
    for x in np.linspace(0.1,1.0,10)
])

ax.grid(
    True,
    linestyle="--",
    alpha=0.6
)

ax.set_title(
    "TRIBE v2 Cortical Representation Similarity to Porn",
    fontsize=20,
    pad=50
)

plt.tight_layout()

plt.savefig(
    "porn_similarity_radar_publication.png",
    dpi=400,
    bbox_inches="tight"
)

plt.show()