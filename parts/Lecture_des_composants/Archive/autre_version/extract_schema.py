"""
extract_schema.py  v2
=====================
Script adapté aux SCHÉMAS ÉLECTRONIQUES (pas PCB).

Approche :
  1. Suppression des lignes du circuit (morphologie) avant OCR
     → les traces horizontales/verticales masquent le texte pour l'OCR
  2. Tesseract PSM 11 (sparse text) — lit le texte dispersé sans layout
     → meilleur que EasyOCR sur les schémas (CRAFT rate les petits textes)
  3. EasyOCR en fallback si Tesseract trouve peu de résultats

⚠ Limitation connue : en dessous de ~150 DPI, les petites refs (R1, C8)
  font <10px — OCR non fiable même à forte résolution virtuelle.
  Pour de meilleurs résultats, utiliser un scan ≥ 300 DPI.

Usage:
  python3 extract_schema.py --img schema.png
  python3 extract_schema.py --img schema.png --scale 8 --conf 10
"""

import cv2
import numpy as np
import pytesseract
import easyocr
import argparse
import json
import csv
import re
import sys
import time
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────
#  TERMINAL — couleurs + helpers
# ──────────────────────────────────────────────────────────

class C:
    RESET  = '\033[0m'
    BOLD   = '\033[1m'
    BLUE   = '\033[34m'
    CYAN   = '\033[36m'
    GREEN  = '\033[32m'
    YELLOW = '\033[33m'
    RED    = '\033[31m'
    GRAY   = '\033[90m'
    WHITE  = '\033[97m'

def _c(text, *codes):
    return ''.join(codes) + str(text) + C.RESET

def step(icon, label, value=''):
    val = f'  {C.GRAY}{value}{C.RESET}' if value else ''
    print(f'  {icon}  {C.BOLD}{label}{C.RESET}{val}')

def ok(label):
    print(f'  {_c("✔", C.GREEN)}  {label}')

def warn(label):
    print(f'  {_c("⚠", C.YELLOW)}  {label}')


# ──────────────────────────────────────────────────────────
#  EASYOCR (chargé une seule fois)
# ──────────────────────────────────────────────────────────

print(f'\n  {_c("⟳", C.CYAN)}  Chargement EasyOCR...', end='', flush=True)
_t0 = time.time()
READER = easyocr.Reader(['en'], gpu=True, verbose=False)
print(f'  {_c("✔", C.GREEN)}  EasyOCR pret  {_c(f"({time.time()-_t0:.1f}s)", C.GRAY)}')

ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'


# ──────────────────────────────────────────────────────────
#  REGEX — RÉFÉRENCES DE SCHÉMA
# ──────────────────────────────────────────────────────────
# R1-R999, C1-C999, L1-L99, D1-D99, Q1-Q99
# T1-T99, TR1-TR99, IC1-IC99, U1-U99
# VR1, SW1, TP12, ZD1, CF1 ...

SCHEMA_PATTERN = re.compile(
    r'^('
    r'[A-Z]{1,3}\d{1,4}[A-Z]?'
    r')$',
    re.IGNORECASE
)

STOP_WORDS = {
    'CHANNEL', 'RIGHT', 'LEFT', 'FILTER', 'INPUT', 'OUTPUT',
    'GND', 'VCC', 'VDD', 'VSS', 'PWR', 'OUT', 'IN', 'NC', 'COM',
    'AMP', 'OSC', 'REF', 'CLK', 'RST', 'INT', 'EXT',
    'REC', 'PLAY', 'STOP', 'AUX', 'TAPE', 'DISC', 'FLAY',
    'DIAGRAM', 'SCHEMATIC',
}

# Corrections OCR spécifiques aux schémas
OCR_FIX = {
    'O': 'Q',  # O lu à la place de Q (transistors)
}

def fix_schema_ocr(text: str) -> str:
    t = text.strip().upper().replace(' ', '')
    # O1, O2... → Q1, Q2... (O seul suivi de chiffres = transistor Q)
    if t and t[0] == 'O' and len(t) > 1 and t[1].isdigit():
        t = 'Q' + t[1:]
    # Correction générique I→1 dans la partie numérique
    i = 0
    while i < len(t) and not t[i].isdigit():
        i += 1
    if i < len(t):
        prefix = t[:i]
        nums = ''.join('1' if c == 'I' else '0' if c == 'O' else c
                       for c in t[i:])
        t = prefix + nums
    return t


def is_valid(text: str) -> bool:
    t = text.strip().upper().replace(' ', '')
    if not t or len(t) < 2:
        return False
    if t in STOP_WORDS:
        return False
    if re.match(r'^\d', t):   # commence par chiffre → valeur
        return False
    return bool(SCHEMA_PATTERN.match(t))


# ──────────────────────────────────────────────────────────
#  PRÉTRAITEMENT — SUPPRESSION DES LIGNES DU CIRCUIT
# ──────────────────────────────────────────────────────────

def remove_circuit_lines(img_bgr, scale: int):
    """
    Upscale + binarise + supprime les lignes H/V longues (traces du circuit).
    Retourne l'image en niveaux de gris, texte noir sur fond blanc.
    """
    h, w = img_bgr.shape[:2]
    big = cv2.resize(img_bgr, (w * scale, h * scale),
                     interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Longueur minimale d'une ligne à supprimer (en pixels scalés)
    # ~1mm sur l'image originale → scale * 4
    line_len = max(40, scale * 6)

    h_kern  = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    v_kern  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kern, iterations=2)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kern, iterations=2)

    no_lines = cv2.subtract(bw, cv2.add(h_lines, v_lines))
    # Légère dilatation pour reconstituer les caractères brisés
    dilated  = cv2.dilate(no_lines, np.ones((2, 2)), iterations=1)

    return cv2.bitwise_not(dilated)   # texte noir sur blanc


# ──────────────────────────────────────────────────────────
#  OCR — TESSERACT PSM11 (sparse text)
# ──────────────────────────────────────────────────────────

TESS_CONFIG = (
    '--psm 11 '
    '-c tessedit_char_whitelist='
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-'
)


def ocr_tesseract(processed_gray, scale: int, conf_min: int = 10):
    data = pytesseract.image_to_data(
        processed_gray, config=TESS_CONFIG,
        output_type=pytesseract.Output.DICT
    )
    results = []
    for i, text in enumerate(data['text']):
        t = text.strip()
        conf = int(data['conf'][i])
        if conf < conf_min or not t or len(t) < 2:
            continue
        cx = int((data['left'][i] + data['width'][i] / 2) / scale)
        cy = int((data['top'][i] + data['height'][i] / 2) / scale)
        x1 = int(data['left'][i] / scale)
        y1 = int(data['top'][i] / scale)
        x2 = int((data['left'][i] + data['width'][i]) / scale)
        y2 = int((data['top'][i] + data['height'][i]) / scale)
        results.append({
            'raw': t.upper(), 'conf': conf / 100.0,
            'cx': cx, 'cy': cy,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        })
    return results


# ──────────────────────────────────────────────────────────
#  OCR — EASYOCR FALLBACK (image entière upscalée)
# ──────────────────────────────────────────────────────────

def ocr_easyocr(img_bgr, scale: int, conf_min: float = 0.20):
    h, w = img_bgr.shape[:2]
    big  = cv2.resize(img_bgr, (w * scale, h * scale),
                      interpolation=cv2.INTER_CUBIC)
    k     = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(big, -1, k)

    try:
        results = READER.readtext(
            sharp, detail=1, paragraph=False,
            allowlist=ALLOWLIST, width_ths=0.9, min_size=5,
        )
    except Exception:
        return []

    out = []
    for (bbox_pts, text, conf) in results:
        if conf < conf_min:
            continue
        t  = text.strip().upper().replace(' ', '')
        if not t:
            continue
        pts = np.array(bbox_pts)
        cx  = int(pts[:, 0].mean() / scale)
        cy  = int(pts[:, 1].mean() / scale)
        x1  = int(pts[:, 0].min()  / scale)
        y1  = int(pts[:, 1].min()  / scale)
        x2  = int(pts[:, 0].max()  / scale)
        y2  = int(pts[:, 1].max()  / scale)
        out.append({
            'raw': t, 'conf': conf,
            'cx': cx, 'cy': cy,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        })
    return out


# ──────────────────────────────────────────────────────────
#  DÉDUPLICATION SPATIALE + PAR TEXTE
# ──────────────────────────────────────────────────────────

def deduplicate(detections, pos_tol=25):
    """Enlève doublons de position (tuiles overlappantes) et de texte."""
    # 1) Dédoublonnage spatial
    kept = []
    for d in detections:
        dup = False
        for k in kept:
            if (abs(d['center']['x'] - k['center']['x']) < pos_tol and
                    abs(d['center']['y'] - k['center']['y']) < pos_tol and
                    d['text'] == k['text']):
                dup = True
                break
        if not dup:
            kept.append(d)

    # 2) Pour les refs valides, garder la meilleure par texte
    seen = {}
    result_invalids = []
    for d in kept:
        if not d['is_valid']:
            result_invalids.append(d)
            continue
        key = d['text'].upper().replace(' ', '')
        if key not in seen or d['ocr_conf'] > seen[key]['ocr_conf']:
            seen[key] = d

    return result_invalids + list(seen.values())


# ──────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────

def extract(img_path, scale=6, tess_conf_min=10):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {img_path}")

    # Gestion RGBA
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())

    # Estimation DPI (hypothèse A4 portrait)
    dpi_est = int(w / (210 / 25.4))
    step('◉', 'Image', f'{w}x{h} px  |  ~{dpi_est} DPI estimé')

    if dpi_est < 150:
        warn(f'Résolution faible ({dpi_est} DPI) — '
             'les petites refs (<10px) seront mal lues. '
             'Idéal : ≥ 300 DPI.')

    # Inversion si fond sombre
    if mean < 80:
        img = cv2.bitwise_not(img)
        ok('Fond sombre → image inversée')

    # ── Tesseract (approche principale) ──────────────────
    step('◈', 'Prétraitement', f'suppression lignes circuit  scale×{scale}')
    processed = remove_circuit_lines(img, scale)

    step('◈', 'Tesseract', 'PSM 11 sparse text')
    t0 = time.time()
    raw_tess = ocr_tesseract(processed, scale, conf_min=tess_conf_min)
    ok(f'Tesseract : {len(raw_tess)} textes en {time.time()-t0:.1f}s')

    # ── EasyOCR fallback si trop peu de résultats ────────
    raw_easy = []
    if len(raw_tess) < 20:
        step('◈', 'EasyOCR', 'fallback (Tesseract a peu trouvé)')
        t1 = time.time()
        raw_easy = ocr_easyocr(img, scale=min(scale, 5))
        ok(f'EasyOCR : {len(raw_easy)} textes en {time.time()-t1:.1f}s')

    # ── Fusion + post-traitement ─────────────────────────
    all_raw = raw_tess + raw_easy
    detections = []
    for r in all_raw:
        fixed = fix_schema_ocr(r['raw'])
        valid = is_valid(fixed)
        detections.append({
            'text':     fixed,
            'raw':      r['raw'],
            'is_valid': valid,
            'ocr_conf': round(r['conf'], 3),
            'source':   'tess' if r in raw_tess else 'easy',
            'bbox':     {'x1': r['x1'], 'y1': r['y1'],
                         'x2': r['x2'], 'y2': r['y2']},
            'center':   {'x': r['cx'], 'y': r['cy']},
        })

    before = sum(1 for d in detections if d['is_valid'])
    detections = deduplicate(detections)
    after  = sum(1 for d in detections if d['is_valid'])
    if before != after:
        ok(f'Déduplication : {before} → {_c(after, C.GREEN, C.BOLD)} refs uniques')

    return img, detections, processed


# ──────────────────────────────────────────────────────────
#  VISUALISATION + EXPORTS
# ──────────────────────────────────────────────────────────

def draw_results(img, detections):
    out = img.copy()
    for d in detections:
        b = d['bbox']
        x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
        color = (0, 200, 0) if d['is_valid'] else (0, 140, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        if d['text']:
            fs = 0.35
            (tw, th), _ = cv2.getTextSize(
                d['text'], cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(out, d['text'], (x1 + 1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1)
    return out


def save_json(detections, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)


def save_csv(detections, path):
    fields = ['text', 'raw', 'is_valid', 'ocr_conf', 'source',
              'x1', 'y1', 'x2', 'y2', 'cx', 'cy']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d in detections:
            writer.writerow({
                'text': d['text'], 'raw': d['raw'],
                'is_valid': d['is_valid'], 'ocr_conf': d['ocr_conf'],
                'source': d['source'],
                'x1': d['bbox']['x1'], 'y1': d['bbox']['y1'],
                'x2': d['bbox']['x2'], 'y2': d['bbox']['y2'],
                'cx': d['center']['x'], 'cy': d['center']['y'],
            })


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Schéma Component Extractor v2")
    parser.add_argument('--img',   required=True,           help='Image schéma')
    parser.add_argument('--scale', type=int,   default=6,   help='Upscale ×N (def: 6)')
    parser.add_argument('--conf',  type=int,   default=10,  help='Seuil conf Tesseract 0-100 (def: 10)')
    parser.add_argument('--save-processed', action='store_true',
                        help='Sauvegarder image après suppression lignes')
    args = parser.parse_args()

    img_path = Path(args.img)
    if not img_path.exists():
        print(f'  {_c("✘", C.RED)}  Image introuvable : {args.img}')
        return

    W = 58
    print(f'\n  {_c("─" * W, C.BLUE)}')
    print(f'  {_c("  Schéma Component Reference Extractor  v2", C.BOLD + C.WHITE)}')
    print(f'  {_c("─" * W, C.BLUE)}\n')

    t_total = time.time()
    img, detections, processed = extract(
        img_path, scale=args.scale, tess_conf_min=args.conf
    )
    elapsed = time.time() - t_total

    valids  = sorted([d for d in detections if d['is_valid']],
                     key=lambda x: x['ocr_conf'], reverse=True)
    others  = [d for d in detections if d['text'] and not d['is_valid']]
    high    = [d for d in valids if d['ocr_conf'] >= 0.6]

    # ── Résumé ──────────────────────────────────────────────
    print(f'\n  {_c("─" * W, C.GRAY)}')
    print(f'  {_c("RÉSULTATS", C.BOLD + C.CYAN)}')
    print(f'  {_c("─" * W, C.GRAY)}')
    print(f'  Textes lus total     {_c(len(detections), C.WHITE, C.BOLD):>6}')
    print(f'  Refs valides         {_c(len(valids),     C.GREEN, C.BOLD):>6}')
    print(f'    dont haute conf    {_c(len(high),       C.GREEN):>6}  '
          f'{_c("(conf >= 0.6)", C.GRAY)}')
    print(f'  Autres textes        {_c(len(others),     C.YELLOW):>6}')
    print(f'  Temps total          {_c(f"{elapsed:.1f}s", C.CYAN):>6}')
    print(f'  {_c("─" * W, C.GRAY)}')

    if valids:
        print(f'\n  {_c("RÉFÉRENCES TROUVÉES", C.BOLD + C.WHITE)} '
              f'{_c(f"({len(valids)})", C.GRAY)}\n')
        print(f'  {_c("Référence", C.BOLD):<18} '
              f'{_c("Conf", C.BOLD):>7} '
              f'{_c("Brut", C.BOLD):<14} '
              f'{_c("Source", C.BOLD):<8} '
              f'{_c("Position", C.BOLD)}')
        print(f'  {_c("─" * W, C.GRAY)}')

        for d in valids:
            conf = d['ocr_conf']
            qual = (_c('●●●', C.GREEN) if conf >= 0.6
                    else _c('●●○', C.YELLOW) if conf >= 0.35
                    else _c('●○○', C.RED))
            ref  = (_c(d['text'], C.WHITE, C.BOLD) if conf >= 0.6
                    else _c(d['text'], C.GRAY) if conf < 0.35
                    else d['text'])
            raw_disp = d['raw'] if d['raw'] != d['text'] else ''
            pos = f"({d['center']['x']},{d['center']['y']})"
            src = _c(d['source'], C.GRAY)
            print(f'  {ref:<18} {conf:>7.2f}  '
                  f'{_c(raw_disp, C.GRAY):<14} {src:<8} '
                  f'{_c(pos, C.GRAY)}  {qual}')

    # ── Exports ─────────────────────────────────────────────
    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = img_path.stem

    ann_path  = out_dir / f'{stem}_{ts}_annotated.png'
    json_path = out_dir / f'{stem}_{ts}_results.json'
    csv_path  = out_dir / f'{stem}_{ts}_results.csv'

    cv2.imwrite(str(ann_path),  draw_results(img, detections))
    save_json(detections, json_path)
    save_csv(detections, csv_path)

    if args.save_processed:
        proc_path = out_dir / f'{stem}_{ts}_processed.png'
        cv2.imwrite(str(proc_path), processed)
        ok(f'Processed  {_c(proc_path, C.GRAY)}')

    print(f'\n  {_c("─" * W, C.GRAY)}')
    print(f'  {_c("FICHIERS GÉNÉRÉS", C.BOLD + C.CYAN)}')
    print(f'  {_c("─" * W, C.GRAY)}')
    ok(f'Image    {_c(ann_path, C.GRAY)}')
    ok(f'JSON     {_c(json_path, C.GRAY)}')
    ok(f'CSV      {_c(csv_path, C.GRAY)}')
    print(f'\n  {_c("➜", C.CYAN)}  xdg-open {ann_path}')
    print(f'  {_c("─" * W, C.GRAY)}\n')


if __name__ == '__main__':
    main()
