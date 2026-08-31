"""
Classical (non-learned) components used both to build the external test sets and
as comparison metrics:

* Otsu's thresholding (OS)                       - Otsu 1979
* Minimum error thresholding (METS)              - Kittler & Illingworth 1986
* Kapur maximum entropy thresholding (KMES)      - Kapur, Sahoo & Wong 1985
* Dice similarity index (DSI)
* AGIC - average gradient of the illumination component (Xie et al. 2016)
* LOE  - lightness order error (Wang et al. 2013)

All routines operate on 8-bit grey level images and use only numpy/PIL so they
can be run without a deep-learning stack.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Thresholding
# --------------------------------------------------------------------------- #
def _hist(gray: np.ndarray) -> np.ndarray:
    h = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    return h


def otsu_threshold(gray: np.ndarray) -> int:
    h = _hist(gray)
    p = h / max(h.sum(), EPS)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom <= 0] = EPS
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = -1.0
    return int(np.argmax(sigma_b))


def minimum_error_threshold(gray: np.ndarray) -> int:
    """Kittler & Illingworth minimum error thresholding."""
    h = _hist(gray)
    p = h / max(h.sum(), EPS)
    g = np.arange(256, dtype=np.float64)

    P1 = np.cumsum(p)
    P2 = 1.0 - P1
    S1 = np.cumsum(p * g)
    S2 = S1[-1] - S1
    SS1 = np.cumsum(p * g * g)
    SS2 = SS1[-1] - SS1

    best_t, best_J = otsu_threshold(gray), np.inf
    for t in range(1, 255):
        p1, p2 = P1[t], P2[t]
        if p1 <= EPS or p2 <= EPS:
            continue
        m1 = S1[t] / p1
        m2 = S2[t] / p2
        v1 = SS1[t] / p1 - m1 * m1
        v2 = SS2[t] / p2 - m2 * m2
        if v1 <= EPS or v2 <= EPS:
            continue
        J = (1.0 + 2.0 * (p1 * np.log(np.sqrt(v1)) + p2 * np.log(np.sqrt(v2)))
             - 2.0 * (p1 * np.log(p1) + p2 * np.log(p2)))
        if np.isfinite(J) and J < best_J:
            best_J, best_t = J, t
    return int(best_t)


def kapur_threshold(gray: np.ndarray) -> int:
    """Kapur / Sahoo / Wong maximum entropy thresholding."""
    h = _hist(gray)
    p = h / max(h.sum(), EPS)
    P = np.cumsum(p)
    pl = np.where(p > 0, p, EPS)
    Hcum = -np.cumsum(p * np.log(pl))
    Htot = Hcum[-1]

    best_t, best_E = 0, -np.inf
    for t in range(1, 255):
        p1, p2 = P[t], 1.0 - P[t]
        if p1 <= EPS or p2 <= EPS:
            continue
        E = (np.log(p1) + Hcum[t] / p1) + (np.log(p2) + (Htot - Hcum[t]) / p2)
        if np.isfinite(E) and E > best_E:
            best_E, best_t = E, t
    return int(best_t)


# --------------------------------------------------------------------------- #
# Segmentation + DSI
# --------------------------------------------------------------------------- #
def segment_lesion(gray: np.ndarray, threshold: int) -> np.ndarray:
    """On macro-photographs the lesion is darker than the surrounding skin, so the
    lesion is the sub-threshold class.  This polarity is fixed a priori and never
    chosen using the ground truth."""
    return gray <= threshold


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / s)


def mean_dsi(image_path, gt_path) -> dict:
    """Mean DSI over Otsu / minimum-error / Kapur segmentations (the GISR-Net target)."""
    img = Image.open(image_path).convert("L")
    gt = Image.open(gt_path).convert("L")
    if gt.size != img.size:
        gt = gt.resize(img.size, Image.NEAREST)
    gray = np.asarray(img, dtype=np.uint8)
    mask = np.asarray(gt, dtype=np.uint8) > 127

    d_os = dice(segment_lesion(gray, otsu_threshold(gray)), mask)
    d_me = dice(segment_lesion(gray, minimum_error_threshold(gray)), mask)
    d_ka = dice(segment_lesion(gray, kapur_threshold(gray)), mask)
    return {
        "DSI_OS": d_os,
        "DSI_METS": d_me,
        "DSI_KMES": d_ka,
        "DSI": float(np.mean([d_os, d_me, d_ka])),
    }


# --------------------------------------------------------------------------- #
# AGIC  (Xie et al., IEEE TBME 2016)  -- Eq. (5) and (6) of the manuscript
# --------------------------------------------------------------------------- #
def agic(image_path, patch: int = 32) -> float:
    img = Image.open(image_path).convert("L")
    g = np.asarray(img, dtype=np.float64)
    H, W = g.shape
    nh, nw = max(H // patch, 2), max(W // patch, 2)
    # block means on an nh x nw grid
    ys = np.linspace(0, H, nh + 1).astype(int)
    xs = np.linspace(0, W, nw + 1).astype(int)
    mu = np.zeros((nh, nw))
    for i in range(nh):
        for j in range(nw):
            mu[i, j] = g[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean()

    grads = []
    for i in range(nh):
        for j in range(nw):
            neigh = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii, jj = i + di, j + dj
                    if 0 <= ii < nh and 0 <= jj < nw:
                        neigh.append(mu[ii, jj])
            if not neigh:
                continue
            grads.append(np.max(np.abs(mu[i, j] - np.asarray(neigh))) / max(mu[i, j], EPS))
    return float(np.mean(grads))


# --------------------------------------------------------------------------- #
# LOE  (Wang et al., IEEE TIP 2013)  -- Eq. (1)-(4) of the manuscript
# --------------------------------------------------------------------------- #
def _lightness(path) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return a.max(axis=2)


def loe(degraded_path, corrected_path) -> float:
    Ld = _lightness(degraded_path)
    Lw = _lightness(corrected_path)
    if Ld.shape != Lw.shape:
        Lw = np.asarray(
            Image.open(corrected_path).convert("RGB").resize(
                (Ld.shape[1], Ld.shape[0]), Image.BILINEAR),
            dtype=np.float64).max(axis=2)

    H, W = Ld.shape
    r = 50.0 / min(H, W)                      # decimation ratio, per Wang et al.
    h, w = max(int(round(H * r)), 8), max(int(round(W * r)), 8)
    yi = np.linspace(0, H - 1, h).astype(int)
    xi = np.linspace(0, W - 1, w).astype(int)
    dd = Ld[np.ix_(yi, xi)].ravel()
    dw = Lw[np.ix_(yi, xi)].ravel()

    Gd = dd[:, None] > dd[None, :]
    Gw = dw[:, None] > dw[None, :]
    rlod = np.logical_xor(Gd, Gw).sum(axis=1)
    return float(rlod.mean())
