import numpy as np
from PIL import Image
import os
from pathlib import Path

img_path = Path('runs/run_20260611_220844/viz/viz03_label_windows_exploration.png')
out_dir = img_path.parent / 'viz03_exploration'
out_dir.mkdir(exist_ok=True)

img = Image.open(img_path).convert('RGB')
arr = np.array(img)
# Find horizontal lines that are pure white (255, 255, 255)
# A row is white if all pixels in it are white
row_means = np.mean(arr, axis=(1, 2))
is_white = row_means > 253

# Find continuous blocks of non-white rows
blocks = []
in_block = False
start = 0
for i, w in enumerate(is_white):
    if not w and not in_block:
        in_block = True
        start = i
    elif w and in_block:
        in_block = False
        if i - start > 100: # only keep blocks larger than 100px
            blocks.append((start, i))
if in_block:
    if len(is_white) - start > 100:
        blocks.append((start, len(is_white)))

print(f"Found {len(blocks)} individual panels.")
for idx, (start, end) in enumerate(blocks):
    # Add a 50px white margin
    s = max(0, start - 50)
    e = min(len(is_white), end + 50)
    cropped = img.crop((0, s, img.width, e))
    cropped.save(out_dir / f"viz03_exploration_{idx+1}.png")
print("Done slicing!")
