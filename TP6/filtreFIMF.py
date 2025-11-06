import numpy as np
import cv2
import time
import os

def _as_uint8(img):
    orig_dtype = img.dtype
    img_min, img_max = img.min(), img.max()
    if img_max == img_min:
        scale = 1.0
        img_u8 = np.zeros_like(img, dtype=np.uint8)
    else:
        scale = 255.0 / float(img_max - img_min)
        img_u8 = np.clip((img - img_min) * scale, 0, 255).astype(np.uint8)
    return img_u8, orig_dtype, img_min, img_max, scale

def _from_uint8(img_u8, orig_dtype, img_min, img_max, scale):
    if np.issubdtype(orig_dtype, np.integer) and orig_dtype != np.uint8:
        inv = img_u8.astype(np.float32) / 255.0
        out = (inv * (img_max - img_min) + img_min).round().astype(orig_dtype)
    elif np.issubdtype(orig_dtype, np.floating):
        inv = img_u8.astype(np.float32) / 255.0
        out = (inv * (img_max - img_min) + img_min).astype(orig_dtype)
    else:
        out = img_u8
    return out

def fimf(image, radius=2):
    """
    Fast Isotropic Median Filter (FIMF-like).
    - radius=1 ~ noyau 3x3 circulaire
    - radius=2 ~ noyau 5x5 circulaire
    """
    if radius < 0:
        raise ValueError("radius doit être >= 0")
    img = np.asarray(image)
    if img.ndim == 2:
        return _fimf_single(img, radius)
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        chans = []
        for c in range(img.shape[2]):
            chans.append(_fimf_single(img[..., c], radius))
        return np.stack(chans, axis=2)
    else:
        raise ValueError("Image doit être 2D (gris) ou 3D (H,W,C).")

def _fimf_single(gray, radius):
    if radius == 0:
        return gray.copy()

    t0 = time.perf_counter()
    # Prépare uint8 pour chemins rapides
    img_u8, orig_dtype, img_min, img_max, scale = _as_uint8(gray.astype(np.float32))

    # 1) scikit-image (noyau circulaire, rapide)
    try:
        from skimage.morphology import disk
        from skimage.filters.rank import median as rank_median
        fp = disk(radius)
        out_u8 = rank_median(img_u8, footprint=fp, mode='reflect')
        out = _from_uint8(out_u8, orig_dtype, img_min, img_max, scale)
        print(f"fimf: skimage.rank.median (radius={radius}) en {time.perf_counter()-t0:.3f}s")
        return out
    except Exception:
        pass

    # 2) OpenCV medianBlur (fenêtre carrée — approximation)
    try:
        ksize = int(2*radius + 1)
        # OpenCV exige un impair >= 3
        if ksize < 3:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1
        out_u8 = cv2.medianBlur(img_u8, ksize)
        out = _from_uint8(out_u8, orig_dtype, img_min, img_max, scale)
        print(f"fimf: cv2.medianBlur (ksize={ksize}) en {time.perf_counter()-t0:.3f}s")
        return out
    except Exception:
        pass

    # 3) Fallback NumPy circulaire (vectorisé)
    out = circular_median_numpy(gray, radius)
    print(f"fimf: fallback NumPy vectorisé (radius={radius}) en {time.perf_counter()-t0:.3f}s")
    return out

def circular_median_numpy(gray, radius):
    """Médiane isotrope avec masque circulaire, padding miroir, vectorisée."""
    from numpy.lib.stride_tricks import sliding_window_view
    r = int(radius)
    if r == 0:
        return gray.copy()

    # Masque circulaire
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = (x*x + y*y) <= (r*r)
    sel = mask.ravel()

    # Padding + fenêtres glissantes
    pad = np.pad(gray, r, mode='reflect')
    win = sliding_window_view(pad, (2*r+1, 2*r+1))  # shape: (H, W, K, K)
    H, W = win.shape[:2]
    flat = win.reshape(H, W, -1)[..., sel]         # (H, W, nb_points_cercle)

    # Médiane par pixel
    out = np.median(flat, axis=2)
    return out.astype(gray.dtype, copy=False)

# --- Script de test ---
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("image.png introuvable. Place l'image dans le dossier ou mets le bon chemin.")

den_3x3 = fimf(img, radius=1)  # ~3x3 circulaire
den_5x5 = fimf(img, radius=2)  # ~5x5 circulaire

os.makedirs("TP6", exist_ok=True)
cv2.imwrite("TP6/out_fimf_3x3.png", den_3x3)
cv2.imwrite("TP6/out_fimf_5x5.png", den_5x5)
print("OK -> TP6/out_fimf_3x3.png, TP6/out_fimf_5x5.png")
