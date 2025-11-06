# filename: fimf_sobel_pipeline.py
import os
import time
import argparse
import numpy as np
import cv2

# ========= FIMF-like (identique aux versions précédentes) =========
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
        print(f"[FIMF] skimage.rank.median (radius={radius}) en {time.perf_counter()-t0:.3f}s")
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
        print(f"[FIMF] cv2.medianBlur (ksize={ksize}) en {time.perf_counter()-t0:.3f}s")
        return out
    except Exception:
        pass

    # 3) Fallback NumPy circulaire (vectorisé)
    out = circular_median_numpy(gray, radius)
    print(f"[FIMF] fallback NumPy vectorisé (radius={radius}) en {time.perf_counter()-t0:.3f}s")
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

# ========= Pipeline FIMF + SOBEL =========
def sobel_magnitude(gray, ksize=3):
    # gradient X & Y
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    # normalisation 0..255
    maxv = mag.max()
    if maxv > 0:
        mag = (mag * (255.0 / maxv)).astype(np.uint8)
    else:
        mag = np.zeros_like(gray, dtype=np.uint8)
    return mag, gx, gy

def main():
    ap = argparse.ArgumentParser(description="Pipeline FIMF (lissage isotrope) + Sobel (accentuation de contours).")
    ap.add_argument("image", nargs="?", default=None, help="Chemin de l'image d'entrée (laisser vide pour une démo synthétique)")
    ap.add_argument("--radius", type=int, default=1, help="Radius FIMF (1~3x3, 2~5x5, 3~7x7). Defaut=1")
    ap.add_argument("--sobel-ksize", type=int, default=3, help="Taille du noyau Sobel (3,5,7). Defaut=3")
    ap.add_argument("--outdir", default="TP6", help="Dossier de sortie. Defaut=TP6")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Charge l'image si fournie, sinon crée une image de démonstration
    if args.image is not None:
        img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Impossible de lire: {args.image}")
        input_desc = f"file:{args.image}"
    else:
        H, W = 256, 256
        # Démo: gradient + formes pour tester le lissage et les contours
        img = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
        cv2.circle(img, (W//2, H//2), 60, 180, -1)
        cv2.rectangle(img, (30, 30), (100, 100), 80, -1)
        cv2.putText(img, "DEMO", (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 200, 2, cv2.LINE_AA)
        input_desc = "image"
        cv2.imwrite(os.path.join(args.outdir, "image.png"), img)

    # 1) FIMF
    t0 = time.perf_counter()
    smoothed = fimf(img, radius=args.radius)
    t_fimf = time.perf_counter() - t0
    cv2.imwrite(os.path.join(args.outdir, f"fimf_r{args.radius}.png"), smoothed)

    # 2) SOBEL sur l'image lissée
    t1 = time.perf_counter()
    mag, gx, gy = sobel_magnitude(smoothed, ksize=args.sobel_ksize)
    t_sobel = time.perf_counter() - t1

    # Sauvegardes
    cv2.imwrite(os.path.join(args.outdir, f"sobel_mag_r{args.radius}_k{args.sobel_ksize}.png"), mag)

    # Optionnel: garder gx/gy pour analyse (visualisation en uint8)
    gx_abs = np.abs(gx)
    gy_abs = np.abs(gy)
    gx_max = gx_abs.max()
    gy_max = gy_abs.max()
    if gx_max > 0:
        gx_viz = (gx_abs * (255.0 / gx_max)).astype(np.uint8)
    else:
        gx_viz = np.zeros_like(img, dtype=np.uint8)
    if gy_max > 0:
        gy_viz = (gy_abs * (255.0 / gy_max)).astype(np.uint8)
    else:
        gy_viz = np.zeros_like(img, dtype=np.uint8)

    cv2.imwrite(os.path.join(args.outdir, f"sobel_gx_r{args.radius}_k{args.sobel_ksize}.png"), gx_viz)
    cv2.imwrite(os.path.join(args.outdir, f"sobel_gy_r{args.radius}_k{args.sobel_ksize}.png"), gy_viz)

    print(f"[OK] Entrée: {input_desc}")
    print(f"[OK] Sorties -> {args.outdir}/")
    print(f"    FIMF radius={args.radius}: {t_fimf:.3f}s")
    print(f"    Sobel ksize={args.sobel_ksize}: {t_sobel:.3f}s")

if __name__ == "__main__":
    main()
