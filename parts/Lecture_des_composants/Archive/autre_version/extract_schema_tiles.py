"""
extract_schema_tiles.py
=======================
Extraction des références composants (R, C, Q, L, D, IC...) depuis un schéma.

Approche : tuiles chevauchantes
  1. Upscale ×8 (85 DPI → ~680 DPI virtuel)
  2. CLAHE + binarisation adaptative
  3. Division en tuiles 600×600 px (overlap 150 px)
  4. Tesseract PSM 11 (sparse text) sur chaque tuile
  5. Reconstitution des coordonnées dans l'image originale
  6. Déduplication spatiale + filtrage des refs valides

Usage:
  python3 extract_schema_tiles.py schema.png
  python3 extract_schema_tiles.py schema.png --scale 10 --tile 500 --conf 10
"""

import cv2
import numpy as np
import pytesseract
import argparse
import csv
import re
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
#  PATTERN COMPOSANTS — préfixes électroniques valides
# ─────────────────────────────────────────────────────────────────────────────

COMP_PATTERN = re.compile(
    r'^('
    r'IC\d{1,4}[A-Z]?'         # IC401, IC3A
    r'|TR\d{1,4}[A-Z]?'        # TR1, TR12
    r'|BC\d{2,4}[A-Z]?'        # BC178
    r'|[RCLDJ]\d{1,4}[A-Z]?'   # R1, C402, L3, D5
    r'|Q\d{1,3}[A-Z]?'         # Q1, Q12
    r'|VR\d{1,3}'              # VR1
    r'|SW\d{1,3}'              # SW1
    r'|TP\d{1,3}'              # TP12
    r'|ZD\d{1,3}'              # ZD1
    r')$',
    re.IGNORECASE
)

# Corrections OCR classiques sur schémas
OCR_FIX = {
    'O': '0', 'o': '0',  # O → 0 dans la partie numérique
    'I': '1', 'l': '1', '|': '1',
    'G': '6',
    'S': '5', 's': '5',
    'Z': '2',
    'B': '8',
}

# Mots à ignorer (faux positifs fréquents)
STOP = {
    'GND', 'VCC', 'VDD', 'VSS', 'IN', 'OUT', 'COM', 'NC',
    'AMP', 'REF', 'OSC', 'CLK', 'PWR', 'GN', 'VC', 'CH',
    'LEFT', 'RIGHT', 'INPUT', 'OUTPUT', 'CHANNEL',
    'REC', 'PLAY', 'TAPE', 'AUX', 'DIAGRAM', 'SCHEMATIC',
}


# ─────────────────────────────────────────────────────────────────────────────
#  NETTOYAGE OCR
# ─────────────────────────────────────────────────────────────────────────────

def clean_ref(text: str) -> str | None:
    """
    Prend un texte brut OCR, retourne une référence propre ou None.
    Ex : "Rl2" -> "R12",  "C4O2" -> "C402",  "GND" -> None
    """
    t = text.strip()
    if not t or len(t) < 2:
        return None

    t_up = t.upper().replace(' ', '')

    if t_up in STOP:
        return None

    # Cherche un préfixe valide dans le texte
    m = COMP_PATTERN.match(t_up)
    if not m:
        return None

    raw = m.group(1).upper()

    # Trouver la frontière lettre/chiffre
    i = 0
    while i < len(raw) and not raw[i].isdigit():
        i += 1

    if i == len(raw):  # pas de chiffre
        return None

    prefix = raw[:i]
    numpart = raw[i:]

    # Corriger les confusions dans la partie numérique
    numpart_fixed = ''.join(
        OCR_FIX.get(c, c) for c in numpart
    )

    # Garder seulement chiffres + éventuelle lettre suffixe finale
    clean_num = ''
    suffix = ''
    for j, c in enumerate(numpart_fixed):
        if c.isdigit():
            clean_num += c
        elif c.isalpha() and j == len(numpart_fixed) - 1 and clean_num:
            suffix = c  # lettre finale = suffixe valide (ex: IC3A)
        # sinon on ignore

    if not clean_num:
        return None

    # Supprimer les zéros initiaux superflus (Q010 → Q10)
    num_val = int(clean_num)
    if num_val == 0:        # Q0, R0, C0 → probablement un faux positif
        return None
    clean_num = str(num_val)

    ref = prefix + clean_num + suffix

    # Rejeter les refs numériquement impossibles (Q648, R9999...)
    if len(clean_num) > 3 and prefix in ('Q', 'L', 'D', 'TR'):
        return None

    return ref


# ─────────────────────────────────────────────────────────────────────────────
#  PRÉTRAITEMENT DE L'IMAGE ENTIÈRE
# ─────────────────────────────────────────────────────────────────────────────

def filter_tile_blobs(tile_bw: np.ndarray, char_min: int, char_max: int) -> np.ndarray:
    """
    Filtrage par composantes connexes SUR UNE TUILE (rapide car petite image).
    Supprime les blobs trop grands (traces) et trop petits (bruit).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        tile_bw, connectivity=8
    )
    mask = np.zeros_like(tile_bw)
    for lbl in range(1, num_labels):
        x, y, w, h, area = stats[lbl]
        if (char_min <= h <= char_max
                and w <= char_max * 5
                and area <= char_max * char_max * 3):
            mask[labels == lbl] = 255
    k = np.ones((2, 2), np.uint8)
    return cv2.dilate(mask, k, iterations=1)


def preprocess(img_bgr: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Upscale × scale + CLAHE + Otsu.
    Retourne (image couleur upscalée, image binarisée BRUTE traits noirs sur fond blanc).
    Le filtrage par blobs est fait tuile par tuile dans ocr_tiles().
    """
    h, w = img_bgr.shape[:2]
    big = cv2.resize(img_bgr, (w * scale, h * scale),
                     interpolation=cv2.INTER_CUBIC)

    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(big, -1, k)
    gray  = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)

    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Binarisation : traits/texte noirs sur fond blanc
    _, bw = cv2.threshold(enhanced, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return big, bw


# ─────────────────────────────────────────────────────────────────────────────
#  OCR PAR TUILES
# ─────────────────────────────────────────────────────────────────────────────

TESS_CFG = (
    '--oem 3 --psm 11 '
    '-c tessedit_char_whitelist='
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
)

TESS_CFG2 = (
    '--oem 3 --psm 6 '
    '-c tessedit_char_whitelist='
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
)


def ocr_tiles(binary: np.ndarray, scale: int,
              tile_size: int = 600, overlap: int = 150,
              conf_min: int = 15) -> list[dict]:
    """
    Divise l'image binarisée en tuiles chevauchantes et OCR chacune.
    Retourne liste de {ref, cx, cy, conf} dans les coordonnées ORIGINALES.
    """
    H, W = binary.shape
    results = []

    step = tile_size - overlap
    tiles_x = max(1, (W - overlap) // step + (1 if (W - overlap) % step else 0))
    tiles_y = max(1, (H - overlap) // step + (1 if (H - overlap) % step else 0))
    total = tiles_x * tiles_y

    print(f'  Grille de tuiles : {tiles_x} × {tiles_y} = {total} tuiles', flush=True)

    done = 0
    for row in range(tiles_y):
        for col in range(tiles_x):
            x0 = col * step
            y0 = row * step
            x1 = min(x0 + tile_size, W)
            y1 = min(y0 + tile_size, H)
            tile = binary[y0:y1, x0:x1]

            if tile.size == 0:
                continue

            # Filtrage blobs texte sur la tuile (rapide — 600×600)
            char_min = max(8,  scale * 3)
            char_max = max(80, scale * 10)
            tile_clean = filter_tile_blobs(tile, char_min, char_max)
            tile_for_ocr = cv2.bitwise_not(tile_clean)  # texte noir sur blanc

            # Marge blanche autour de la tuile (améliore Tesseract)
            tile_pad = cv2.copyMakeBorder(
                tile_for_ocr, 20, 20, 20, 20,
                cv2.BORDER_CONSTANT, value=255
            )

            try:
                # PSM 11 : sparse text (refs dispersées dans le schéma)
                data = pytesseract.image_to_data(
                    tile_pad, config=TESS_CFG,
                    output_type=pytesseract.Output.DICT
                )
                # PSM 6 : texte en bloc (parfois plus performant par tuile)
                data2 = pytesseract.image_to_data(
                    tile_pad, config=TESS_CFG2,
                    output_type=pytesseract.Output.DICT
                )
                # Fusion des deux passes
                for key in data:
                    data[key] = data[key] + data2[key]
            except Exception:
                done += 1
                continue

            for i, word in enumerate(data['text']):
                word = word.strip()
                if not word:
                    continue
                conf = int(data['conf'][i])
                if conf < conf_min:
                    continue

                ref = clean_ref(word)
                if ref is None:
                    continue

                # Coordonnées dans la tuile paddée
                wx = data['left'][i] + data['width'][i] // 2
                wy = data['top'][i] + data['height'][i] // 2

                # Retrait du padding (20 px)
                wx -= 20
                wy -= 20

                # Coordonnées dans l'image upscalée
                abs_x_up = x0 + wx
                abs_y_up = y0 + wy

                # Coordonnées dans l'image ORIGINALE
                cx_orig = int(abs_x_up / scale)
                cy_orig = int(abs_y_up / scale)

                results.append({
                    'ref':  ref,
                    'raw':  word,
                    'conf': conf,
                    'cx':   cx_orig,
                    'cy':   cy_orig,
                    'x1':   int((x0 + data['left'][i] - 20) / scale),
                    'y1':   int((y0 + data['top'][i] - 20) / scale),
                    'x2':   int((x0 + data['left'][i] + data['width'][i] - 20) / scale),
                    'y2':   int((y0 + data['top'][i] + data['height'][i] - 20) / scale),
                })

            done += 1
            pct = int(100 * done / total)
            bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
            sys.stdout.write(f'\r  [{bar}] {pct}%  ({done}/{total})')
            sys.stdout.flush()

    sys.stdout.write('\n')
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  DÉDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate(results: list[dict], pos_tol: int = 30) -> list[dict]:
    """
    Pour chaque ref valide, garde la détection avec la meilleure conf.
    Fusionne les détections de même ref proches spatialement.
    """
    # Regrouper par ref
    by_ref = defaultdict(list)
    for r in results:
        by_ref[r['ref'].upper()].append(r)

    final = []
    for ref, group in by_ref.items():
        # Trier par conf décroissante
        group.sort(key=lambda x: x['conf'], reverse=True)

        # Clustering spatial simple : garder une détection par zone
        kept = []
        for det in group:
            too_close = False
            for k in kept:
                if (abs(det['cx'] - k['cx']) < pos_tol and
                        abs(det['cy'] - k['cy']) < pos_tol):
                    too_close = True
                    break
            if not too_close:
                kept.append(det)

        final.extend(kept)

    return final


# ─────────────────────────────────────────────────────────────────────────────
#  AFFICHAGE + EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def draw_results(img_orig: np.ndarray, results: list[dict]) -> np.ndarray:
    out = img_orig.copy()
    for d in results:
        x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(out.shape[1], x2), min(out.shape[0], y2)
        color = (0, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        fs = 0.35
        (tw, th), _ = cv2.getTextSize(d['ref'], cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, d['ref'], (x1 + 1, max(th, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1)
    return out


def save_csv(results: list[dict], path: str):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['ref', 'raw', 'conf', 'cx', 'cy', 'x1', 'y1', 'x2', 'y2'])
        w.writeheader()
        for d in results:
            w.writerow(d)


def print_summary(results: list[dict]):
    # Grouper par préfixe
    by_prefix = defaultdict(list)
    for d in results:
        m = re.match(r'^([A-Z]+)', d['ref'])
        prefix = m.group(1) if m else '?'
        by_prefix[prefix].append(d['ref'])

    print(f'\n  {"─"*50}')
    print(f'  RÉSULTATS — {len(results)} références uniques')
    print(f'  {"─"*50}')
    for prefix in sorted(by_prefix.keys()):
        refs = sorted(set(by_prefix[prefix]),
                      key=lambda x: int(re.search(r'\d+', x).group()))
        print(f'  {prefix:4s} ({len(refs):3d}) : {", ".join(refs[:20])}{"..." if len(refs) > 20 else ""}')
    print(f'  {"─"*50}')


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Schema component extractor (tiles)')
    parser.add_argument('img',               help='Image du schéma (PNG/JPG)')
    parser.add_argument('--scale',  type=int, default=8,   help='Facteur upscale (défaut: 8)')
    parser.add_argument('--tile',   type=int, default=600, help='Taille tuile px (défaut: 600)')
    parser.add_argument('--overlap',type=int, default=150, help='Overlap tuile px (défaut: 150)')
    parser.add_argument('--conf',   type=int, default=15,  help='Seuil conf Tesseract 0-100 (défaut: 15)')
    parser.add_argument('--pos-tol',type=int, default=30,  help='Tolérance pos dédup px orig (défaut: 30)')
    args = parser.parse_args()

    img_path = Path(args.img)
    if not img_path.exists():
        print(f'Erreur : image introuvable : {args.img}')
        sys.exit(1)

    print(f'\n  {"─"*54}')
    print(f'  Schema Component Extractor — tuiles')
    print(f'  {"─"*54}')

    # Chargement
    img = cv2.imread(str(img_path))
    if img is None:
        print(f'Erreur : impossible de lire {args.img}')
        sys.exit(1)

    h, w = img.shape[:2]
    dpi_est = int(w / (210 / 25.4))
    print(f'  Image    : {w}×{h} px  (~{dpi_est} DPI)')
    print(f'  Scale    : ×{args.scale}  → {w*args.scale}×{h*args.scale} px')
    print(f'  Tuiles   : {args.tile}×{args.tile} px  overlap {args.overlap} px')

    # Inversion si fond sombre
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray0.mean() < 80:
        img = cv2.bitwise_not(img)
        print('  Fond sombre détecté → image inversée')

    # Prétraitement
    print('\n  Prétraitement (upscale + CLAHE + binarisation)...')
    t0 = time.time()
    img_up, binary = preprocess(img, args.scale)
    print(f'  OK ({time.time()-t0:.1f}s)')

    # OCR par tuiles
    print('\n  OCR par tuiles (Tesseract PSM 11)...')
    t1 = time.time()
    raw_results = ocr_tiles(binary, args.scale,
                            tile_size=args.tile,
                            overlap=args.overlap,
                            conf_min=args.conf)
    print(f'  {len(raw_results)} détections brutes ({time.time()-t1:.1f}s)')

    # Déduplication
    results = deduplicate(raw_results, pos_tol=args.pos_tol)
    print_summary(results)

    # Exports
    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = img_path.stem

    ann  = out_dir / f'{stem}_{ts}_tiles.png'
    csvf = out_dir / f'{stem}_{ts}_tiles.csv'

    cv2.imwrite(str(ann), draw_results(img, results))
    save_csv(results, str(csvf))

    print(f'\n  Image annotée : {ann}')
    print(f'  CSV           : {csvf}')
    print(f'  {"─"*54}\n')


if __name__ == '__main__':
    main()
