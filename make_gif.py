#!/usr/bin/env python3
# Stitch frame_*.pgm into wave.gif. Needs Pillow: pip install pillow
import glob, sys
from PIL import Image

frames = sorted(glob.glob("frame_*.pgm"))
if not frames:
    sys.exit("no frame_*.pgm files in cwd")

imgs = [Image.open(f) for f in frames]
imgs[0].save("wave.gif", save_all=True, append_images=imgs[1:],
             duration=200, loop=0)
print(f"wrote wave.gif ({len(imgs)} frames)")
