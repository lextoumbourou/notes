# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""
Asset-synthesis montage for the OpenGame article.

Shows a sample of the assets OpenGame generated for its Marvel platformer demo
(four characters, a tileset, two effects, a level background). The assets live
inside the published demo zip, so this script downloads + caches it, extracts
the chosen assets, and composes a labelled montage with a TRANSPARENT
background (so it sits cleanly on the light article page).

Run:  uv run asset_example.py        (or: python3 asset_example.py)
Writes: ../../notes/_media/opengame-asset-example.png
"""
import os
import zipfile
import urllib.request
from PIL import Image, ImageDraw, ImageFont

ZIP_URL = ("https://github.com/leigest519/OpenGame/raw/main/"
           "assets/downloads/demo_platformer_marvel.zip")
CACHE_ZIP = "/tmp/opengame_demo_platformer_marvel.zip"
ASSET_DIR = "demo_platformer_marvel/dist/assets/assets"

# (filename in zip, label) in display order: 2 rows x 4 cols
ITEMS = [
    ("ironman_idle_01.png",   "ironman"),
    ("thor_idle_01.png",      "thor"),
    ("hulk_idle_01.png",      "hulk"),
    ("thanos_idle_01.png",    "thanos (boss)"),
    ("city_tiles.png",        "city_tiles (tileset)"),
    ("mjolnir_projectile.png", "mjolnir_projectile"),
    ("repulsor_blast.png",    "repulsor_blast (fx)"),
    ("level1_bg.png",         "level1_bg (background)"),
]

COLS, ROWS = 4, 2
CELL, PAD, LABEL_H = 380, 18, 34
LABEL_COL = (55, 65, 81, 255)   # dark slate, readable on a light page


def ensure_zip():
    if not os.path.exists(CACHE_ZIP):
        print("downloading demo zip (~490 MB, cached at %s) ..." % CACHE_ZIP)
        urllib.request.urlretrieve(ZIP_URL, CACHE_ZIP)
    return CACHE_ZIP


def load_assets():
    imgs = {}
    with zipfile.ZipFile(ensure_zip()) as z:
        for fn, _ in ITEMS:
            with z.open(f"{ASSET_DIR}/{fn}") as f:
                imgs[fn] = Image.open(f).convert("RGBA").copy()
    return imgs


def font(size):
    for p in ["/System/Library/Fonts/SFNSMono.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    imgs = load_assets()
    fnt = font(20)
    W = COLS * CELL + (COLS + 1) * PAD
    H = ROWS * (CELL + LABEL_H) + (ROWS + 1) * PAD
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))   # transparent
    draw = ImageDraw.Draw(canvas)

    for i, (fn, label) in enumerate(ITEMS):
        r, c = divmod(i, COLS)
        x = PAD + c * (CELL + PAD)
        y = PAD + r * (CELL + LABEL_H + PAD)
        im = imgs[fn]
        inset = 24
        mw, mh = CELL - 2 * inset, CELL - 2 * inset
        s = min(mw / im.width, mh / im.height)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        canvas.alpha_composite(im, (x + (CELL - im.width) // 2,
                                    y + (CELL - im.height) // 2))
        tw = draw.textlength(label, font=fnt)
        draw.text((x + (CELL - tw) // 2, y + CELL + 7), label,
                  fill=LABEL_COL, font=fnt)

    out = os.path.join(os.path.dirname(__file__), "..", "..", "notes",
                       "_media", "opengame-asset-example.png")
    canvas.save(out, "PNG")
    print("saved", os.path.normpath(out), canvas.size)


if __name__ == "__main__":
    main()
