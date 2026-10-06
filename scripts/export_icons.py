#!/usr/bin/env python3
"""
Render the 4K Master (`orb_4k.png`) and macOS & iOS App Icons Directly
from the Procedural WebGL2 Shader (`index.html`) via Headless Chrome + Metal GPU.
"""
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HTML_URL = f"file://{ROOT}/index.html"


def render_webgl_shader(query: str, out_path: Path, w: int, h: int) -> Image.Image:
    t0 = time.time()
    cmd = [
        CHROME,
        "--headless=new",
        "--use-gl=angle",
        "--use-angle=metal",
        "--allow-file-access-from-files",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--window-size={w},{h}",
        f"--screenshot={out_path}",
        f"{HTML_URL}?{query}",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    im = Image.open(out_path).convert("RGB")
    if im.size != (w, h):
        im = im.resize((w, h), Image.Resampling.LANCZOS)
        im.save(out_path, optimize=True)
    arr = np.array(im)
    print(
        f"[WebGL2 Shader] {out_path.name} ({w}x{h}) rendered in {time.time() - t0:.2f}s "
        f"| corner max: {arr[:20, :20].max()} | peak: {arr.max()}"
    )
    return im


def main() -> None:
    macos_dir = ROOT / "icons" / "macos"
    ios_dir = ROOT / "icons" / "ios" / "AppIcon.appiconset"
    macos_dir.mkdir(parents=True, exist_ok=True)
    ios_dir.mkdir(parents=True, exist_ok=True)

    # 1. Native 4K Master (3840 x 4126)
    render_webgl_shader("clean=1&res=4k&t=0.35", ROOT / "orb_4k.png", 3840, 4126)

    # 2. macOS 1254x1254 Master (`orb.png` & `Orb.icon/Assets/orb.png`)
    macos_master = macos_dir / "orb_1254_master.png"
    render_webgl_shader("clean=1&w=1254&h=1254&r=0.388&t=0.35", macos_master, 1254, 1254)
    shutil.copyfile(macos_master, ROOT / "orb.png")
    orb_icon_asset = macos_dir / "Orb.icon" / "Assets" / "orb.png"
    if orb_icon_asset.parent.exists():
        shutil.copyfile(macos_master, orb_icon_asset)

    # 3. iOS 2048x2048 Super-Sampled Master -> 13 AppIcon.appiconset PNGs
    with tempfile.TemporaryDirectory() as tmp:
        ios_2k_path = Path(tmp) / "ios_shader_2048.png"
        ios_2k_pil = render_webgl_shader("clean=1&w=2048&h=2048&r=0.382&t=0.35", ios_2k_path, 2048, 2048)
        for sz in [1024, 180, 167, 152, 120, 87, 80, 76, 60, 58, 40, 29, 20]:
            resized = ios_2k_pil.resize((sz, sz), Image.Resampling.LANCZOS)
            resized.save(ios_dir / f"icon-{sz}.png", optimize=True)

    print("All 4K, macOS, and iOS icons exported successfully!")


if __name__ == "__main__":
    main()
