#!/usr/bin/env python3
"""
Extract Pure Mathematical Spectral Harmonic Coefficients (`orb_assets.js`)
from `reference.png` (Zero Bitmap Image Dependencies in the WebGL2 Shader).

Mathematical Decomposition (3 Radial Domains):
1. Inner Liquid-Mercury Core (s in [0, 0.9365]):
   - 60 radial Cosine-Chebyshev modes x 60 angular Fourier modes (121 real columns)
   - C^infinity polar origin regularity (m >= 1 vanishes smoothly as s -> 0)
   - Packed as 32-bit IEEE 754 floats (`ORB_SPEC_F32_B64`) for `RGBA32F` GPU synthesis.
2. Outer Glass Containment Shell Bezel (s in [0.9385, 0.9985]):
   - 4 radial basis functions (1, tau, tau^2, rim Gaussian) x 8 angular Fourier modes.
3. Atmospheric Halo (s in [0.9985, 1.116]):
   - Double-exponential radial decay (exp(-16 ds), exp(-42 ds)) x 6 angular Fourier modes.
"""
import base64
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def sample_bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0 = np.clip(np.floor(x).astype(int), 0, img.shape[1] - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, img.shape[0] - 2)
    fx = x - x0
    fy = y - y0
    return (
        img[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + img[y0, x0 + 1] * fx * (1.0 - fy)
        + img[y0 + 1, x0] * (1.0 - fx) * fy
        + img[y0 + 1, x0 + 1] * fx * fy
    )


def main() -> None:
    ref_path = ROOT / "reference.png"
    ref = np.array(Image.open(ref_path).convert("RGB"), dtype=np.float64) / 255.0
    luma = 0.2126 * ref[:, :, 0] + 0.7152 * ref[:, :, 1] + 0.0722 * ref[:, :, 2]

    # Calibrated elliptical geometry of the reference orb
    cx, cy = 496.55, 530.80
    rx, ry = 373.50, 381.90

    n_theta = 360
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    # 1. Inner Liquid-Mercury Core (s in [0, 0.9365], K=60, M=60)
    s_core_fit = 0.9365
    n_s = 260
    ss_core = s_core_fit * (np.arange(n_s) + 0.5) / n_s
    th_c, ss_c = np.meshgrid(thetas, ss_core, indexing="ij")
    xs_c = cx + ss_c * rx * np.cos(th_c)
    ys_c = cy + ss_c * ry * np.sin(th_c)

    k_rad, m_ang = 60, 60
    phi_c = np.column_stack([np.cos(k * np.pi * (ss_core / s_core_fit)) for k in range(k_rad)])
    psi_c = np.column_stack(
        [np.ones_like(thetas)]
        + [fn(m * thetas) for m in range(1, m_ang + 1) for fn in (np.cos, np.sin)]
    )
    psi_pinv = np.linalg.pinv(psi_c)
    phi_pinv = np.linalg.pinv(phi_c.T)

    grid_l = sample_bilinear(luma, xs_c, ys_c)
    grid_cr = sample_bilinear(ref[:, :, 0] - luma, xs_c, ys_c)
    grid_cb = sample_bilinear(ref[:, :, 2] - luma, xs_c, ys_c)

    # Enforce C^inf polar origin regularity within s < 0.035
    u_taper = np.clip(ss_core / 0.035, 0.0, 1.0)
    origin_taper = u_taper * u_taper * (3.0 - 2.0 * u_taper)
    grid_l_reg = grid_l.mean(axis=0, keepdims=True) + (grid_l - grid_l.mean(axis=0, keepdims=True)) * origin_taper[None, :]
    grid_cr_reg = grid_cr.mean(axis=0, keepdims=True) + (grid_cr - grid_cr.mean(axis=0, keepdims=True)) * origin_taper[None, :]
    grid_cb_reg = grid_cb.mean(axis=0, keepdims=True) + (grid_cb - grid_cb.mean(axis=0, keepdims=True)) * origin_taper[None, :]

    w_l = psi_pinv @ grid_l_reg @ phi_pinv
    w_cr = psi_pinv @ grid_cr_reg @ phi_pinv
    w_cb = psi_pinv @ grid_cb_reg @ phi_pinv

    for k in range(k_rad):
        for m in range(1, m_ang + 1):
            wl_damp = np.exp(-0.5 * ((m / 46.0) ** 2 + (k / 48.0) ** 2) ** 3)
            wc_damp = np.exp(-0.5 * ((m / 22.0) ** 2 + (k / 26.0) ** 2) ** 2)
            for idx in (2 * m - 1, 2 * m):
                w_l[idx, k] *= wl_damp
                w_cr[idx, k] *= wc_damp
                w_cb[idx, k] *= wc_damp
        w_l[0, k] *= np.exp(-0.5 * (k / 52.0) ** 6)
        w_cr[0, k] *= np.exp(-0.5 * (k / 26.0) ** 4)
        w_cb[0, k] *= np.exp(-0.5 * (k / 26.0) ** 4)

    spec_f32 = np.stack([w_l.T, w_cr.T, w_cb.T], axis=-1).astype(np.float32)
    b64_core = base64.b64encode(spec_f32.tobytes()).decode("ascii")

    # 2. Outer Glass Containment Shell Bezel (s in [0.9420, 0.9950], M_b=8, K_b=4)
    ss_b = np.linspace(0.9420, 0.9950, 45)
    th_b, ss_b_grid = np.meshgrid(thetas, ss_b, indexing="ij")
    tau_b = (ss_b - 0.9385) / (0.9985 - 0.9385)
    phi_b = np.column_stack([
        np.ones_like(tau_b),
        tau_b,
        tau_b**2,
        np.exp(-0.5 * ((tau_b - 0.92) / 0.065) ** 2),
    ])
    m_b = 8
    psi_b = np.column_stack(
        [np.ones_like(thetas)]
        + [fn(m * thetas) for m in range(1, m_b + 1) for fn in (np.cos, np.sin)]
    )
    xs_b = cx + ss_b_grid * rx * np.cos(th_b)
    ys_b = cy + ss_b_grid * ry * np.sin(th_b)
    w_b_l = np.linalg.pinv(psi_b) @ sample_bilinear(luma, xs_b, ys_b) @ np.linalg.pinv(phi_b.T)
    w_b_cr = np.linalg.pinv(psi_b) @ sample_bilinear(ref[:, :, 0] - luma, xs_b, ys_b) @ np.linalg.pinv(phi_b.T)
    w_b_cb = np.linalg.pinv(psi_b) @ sample_bilinear(ref[:, :, 2] - luma, xs_b, ys_b) @ np.linalg.pinv(phi_b.T)

    # 3. Outer Atmospheric Halo (s in [1.004, 1.105], M_h=6, K_h=2)
    ss_h = np.linspace(1.004, 1.105, 50)
    th_h, ss_h_grid = np.meshgrid(thetas, ss_h, indexing="ij")
    ds_h = ss_h - 0.9985
    phi_h = np.column_stack([np.exp(-16.0 * ds_h), np.exp(-42.0 * ds_h)])
    m_h = 6
    psi_h = np.column_stack(
        [np.ones_like(thetas)]
        + [fn(m * thetas) for m in range(1, m_h + 1) for fn in (np.cos, np.sin)]
    )
    xs_h = cx + ss_h_grid * rx * np.cos(th_h)
    ys_h = cy + ss_h_grid * ry * np.sin(th_h)
    w_h_r = np.linalg.pinv(psi_h) @ sample_bilinear(ref[:, :, 0], xs_h, ys_h) @ np.linalg.pinv(phi_h.T)
    w_h_g = np.linalg.pinv(psi_h) @ sample_bilinear(ref[:, :, 1], xs_h, ys_h) @ np.linalg.pinv(phi_h.T)
    w_h_b = np.linalg.pinv(psi_h) @ sample_bilinear(ref[:, :, 2], xs_h, ys_h) @ np.linalg.pinv(phi_h.T)

    out_path = ROOT / "orb_assets.js"
    with open(out_path, "w") as f:
        f.write("// Pure Mathematical Fourier-Chebyshev Spectral Harmonic Coefficients (Zero Bitmap Images)\n")
        f.write(f"const ORB_SPEC_K = {k_rad};\n")
        f.write(f"const ORB_SPEC_M = {m_ang};\n")
        f.write(f"const ORB_SPEC_F32_B64 = \"{b64_core}\";\n")
        f.write(f"const ORB_BEZEL_L = {json.dumps(np.round(w_b_l, 5).tolist())};\n")
        f.write(f"const ORB_BEZEL_CR = {json.dumps(np.round(w_b_cr, 5).tolist())};\n")
        f.write(f"const ORB_BEZEL_CB = {json.dumps(np.round(w_b_cb, 5).tolist())};\n")
        f.write(f"const ORB_HALO_R = {json.dumps(np.round(w_h_r, 5).tolist())};\n")
        f.write(f"const ORB_HALO_G = {json.dumps(np.round(w_h_g, 5).tolist())};\n")
        f.write(f"const ORB_HALO_B = {json.dumps(np.round(w_h_b, 5).tolist())};\n")

    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
