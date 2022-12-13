import pathlib

import imageio.v2 as imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt

src = pathlib.Path("input") / "scene0042_01_00695.jpg"
tgt = pathlib.Path("output") / (src.stem + "_result.jpg")
print(f"{src} -> {tgt}")

img = imageio.imread(src)

border_width = 10
x_min = border_width
y_min = border_width
x_max = img.shape[1] - border_width
y_max = img.shape[0] - border_width

fig, ax = plt.subplots()
ax.imshow(img)

# Create a Rectangle patch
rect = patches.Rectangle(
    (x_min, y_min),
    x_max - x_min,
    y_max - y_min,
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)

# Add the patch to the Axes
ax.add_patch(rect)

fig.savefig(tgt, dpi=300)


import numpy as np

border_width = 10
img = imageio.imread(src)
mask = np.ones_like(img)
mask[:border_width, :, :] = 0
mask[-border_width:, :, :] = 0
mask[:, :border_width, :] = 0
mask[:, -border_width:, :] = 0
mask = mask[:, :, 0]

imageio.imwrite("mask.png", mask)
