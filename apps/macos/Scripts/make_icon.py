#!/usr/bin/env python3
"""Generate the OnIt app icon.

Writes Assets/AppIcon.iconset/*.png and converts it to Assets/AppIcon.icns
with `iconutil`. Run from anywhere; paths are resolved relative to this file.

    python3 Scripts/make_icon.py

The mark: a dark rounded square, a light speech bubble, and an orange spark
inside it — the harness talks, and what it says is agent work. Pure geometry,
no font dependencies, so it renders identically on any machine.
"""

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "Assets"
ICONSET = ASSETS / "AppIcon.iconset"
ICNS = ASSETS / "AppIcon.icns"

# The iconset sizes Apple requires, (name, pixels).
SIZES = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]

# Palette, matched to the app's Theme/Palette.swift feel.
BG_TOP = (34, 37, 52)
BG_BOTTOM = (22, 24, 34)
BUBBLE = (238, 238, 243)
SPARK = (255, 140, 60)


def vertical_gradient(size: int, top: tuple, bottom: tuple) -> Image.Image:
    """A rounded-rect filled with a two-stop vertical gradient."""
    scale = 4  # supersample the gradient, then downscale with the mask
    big = Image.new("RGB", (size * scale, size * scale))
    draw = ImageDraw.Draw(big)
    for y in range(size * scale):
        t = y / (size * scale - 1)
        color = tuple(round(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        draw.line([(0, y), (size * scale, y)], fill=color)
    return big.resize((size, size), Image.LANCZOS)


def draw_mark(size: int) -> Image.Image:
    """One icon, at any pixel size."""
    icon = vertical_gradient(size, BG_TOP, BG_BOTTOM).convert("RGBA")

    # macOS icons sit with a margin; Apple's grid puts the mark at ~80%.
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=round(size * 0.22), fill=255
    )
    icon.putalpha(mask)

    draw = ImageDraw.Draw(icon)

    # The speech bubble: a rounded rect with a small tail, lower-left.
    bubble_left = size * 0.18
    bubble_top = size * 0.24
    bubble_right = size * 0.82
    bubble_bottom = size * 0.68
    tail = [
        (size * 0.30, bubble_bottom - 1),
        (size * 0.24, size * 0.80),
        (size * 0.42, bubble_bottom - 1),
    ]
    draw.rounded_rectangle(
        (bubble_left, bubble_top, bubble_right, bubble_bottom),
        radius=round(size * 0.10),
        fill=BUBBLE,
    )
    draw.polygon(tail, fill=BUBBLE)

    # The spark: a six-point asterisk at the bubble's center. The arms reach
    # about half the bubble's height — a mark, not a fill — and the stroke is
    # ~5% of the icon so it stays legible down at 16 px.
    cx, cy = (bubble_left + bubble_right) / 2, (bubble_top + bubble_bottom) / 2
    arm = size * 0.11
    thick = max(2, round(size * 0.055))
    for angle in (45, 135):
        # Rotate a horizontal bar around the center to draw the diagonals.
        temp = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        ImageDraw.Draw(temp).line(
            [(cx - arm, cy), (cx + arm, cy)], fill=SPARK, width=thick
        )
        temp = temp.rotate(angle, center=(cx, cy), resample=Image.BICUBIC)
        icon.alpha_composite(temp)
    draw.line([(cx, cy - arm), (cx, cy + arm)], fill=SPARK, width=thick)

    return icon


def main() -> int:
    ICONSET.mkdir(parents=True, exist_ok=True)
    for name, pixels in SIZES:
        draw_mark(pixels).save(ICONSET / name)
        print(f"  {name}  {pixels}px")

    result = subprocess.run(
        ["iconutil", "-c", "icns", str(ICONSET), "-o", str(ICNS)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"iconutil failed: {result.stderr}", file=sys.stderr)
        return 1
    print(f"==> {ICNS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
