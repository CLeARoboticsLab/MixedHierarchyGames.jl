import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

files = [
    "9.png",
    "8.png",
    "7.png",
    "6.png",
    "5.png",
    "4.png",
    "3.png",
    "2.png",
    "1.png",
]

fig, ax = plt.subplots(figsize=(15, 8))

# Blend all images at once using weighted averaging so they're on the "same level"
imgs = [mpimg.imread(f).astype(np.float32) for f in files]
if len(imgs) == 0:
    raise RuntimeError("No images found to blend.")

# Ensure all images have the same HxW
h0, w0 = imgs[0].shape[:2]
imgs = [img if img.shape[:2] == (h0, w0) else np.resize(img, (h0, w0, img.shape[2])) for img in imgs]

# Build weights (later images get much higher weight). Tune gamma as desired.
n = len(imgs)
# Compute weight by the numeric index in the filename so 9.png > 8.png > ... > 0.png,
# regardless of list order.
gamma = 0.7  # increase to bias even more toward higher indices
nums = [int(os.path.splitext(os.path.basename(f))[0]) for f in files]
weights = np.array([(num + 1) ** gamma for num in nums], dtype=np.float32)
weights = weights / weights.sum()
print(weights)

# Use RGB channels for blending; ignore alpha if present
rgb_stack = []
for im in imgs:
    if im.shape[-1] == 4:
        rgb_stack.append(im[..., :3])
    else:
        rgb_stack.append(im[..., :3] if im.ndim == 3 else np.stack([im, im, im], axis=-1))
rgb_stack = np.stack(rgb_stack, axis=0)  # [N, H, W, 3]

blended = np.tensordot(weights, rgb_stack, axes=(0, 0))  # [H, W, 3]
blended = np.clip(blended, 0.0, 1.0)

ax.imshow(blended)

ax.axis("off")
plt.tight_layout()
plt.savefig("final_overlay.png", dpi=1000)
