#!/usr/bin/env python
"""READMEに貼る用のGIFを作る (ffmpegなし: imageio + PIL の適応パレット量子化)。"""
import sys
import numpy as np
import imageio
from PIL import Image


def to_gif(src, dst, stride=5, width=640, colors=96, t0=None, t1=None,
           src_fps=60):
    rd = imageio.get_reader(src)
    frames = []
    for i, fr in enumerate(rd):
        t = i / src_fps
        if t0 is not None and t < t0:
            continue
        if t1 is not None and t > t1:
            break
        if i % stride:
            continue
        im = Image.fromarray(fr)
        h = int(im.height * width / im.width)
        im = im.resize((width, h), Image.LANCZOS)
        frames.append(im.convert("RGB").quantize(colors=colors,
                                                 method=Image.MEDIANCUT))
    dur = int(1000 * stride / src_fps)
    frames[0].save(dst, save_all=True, append_images=frames[1:],
                   duration=dur, loop=0, optimize=True)
    import os
    print(f"{dst}: {len(frames)}コマ {width}px {dur}ms/コマ "
          f"{os.path.getsize(dst)/1e6:.1f}MB", flush=True)


if __name__ == "__main__":
    to_gif("outputs/vf_composite.mp4", "docs/flight_connectome.gif",
           stride=5, width=640, colors=96)
    to_gif("outputs/seq_composite.mp4", "docs/sequence.gif",
           stride=8, width=560, colors=80)
