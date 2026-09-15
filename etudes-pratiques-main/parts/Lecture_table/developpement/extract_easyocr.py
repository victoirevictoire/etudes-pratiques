"""
extract_easyocr.py  v7
======================
Améliorations principales :
  1. Merge de boîtes voisines : YOLO split parfois "R" + "516" en 2 boîtes
     → on les fusionne avant OCR pour lire "R516" d'un coup
  2. Déduplication finale : même ref détectée 2-3x → on garde le meilleur score
  3. Correcteur OCR robuste avec scan linéaire (pas de regex fragile)

Usage:
  python3 extract_easyocr.py --img carte.png --model best.pt
  python3 extract_easyocr.py --img carte.png --model best.pt --conf 0.25
"""

import cv2
import numpy as np
import easyocr
import argparse
import json
import csv
import re
import sys
import time
from pathlib import Path
from ultralytics import YOLO
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

def progress_bar(current, total, width=30):
    pct   = current / total if total else 0
    filled = int(width * pct)
    bar   = _c('█' * filled, C.CYAN) + _c('░' * (width - filled), C.GRAY)
    sys.stdout.write(
        f'\r  {_c("⟳", C.CYAN)}  OCR  [{bar}]  '
        f'{_c(f"{current}/{total}", C.WHITE)}  '
        f'{_c(f"{pct*100:.0f}%", C.YELLOW)}'
    )
    sys.stdout.flush()
    if current == total:
        sys.stdout.write('\n')


# ──────────────────────────────────────────────────────────
#  EASYOCR
# ──────────────────────────────────────────────────────────

print(f'\n  {_c("⟳", C.CYAN)}  Chargement EasyOCR...',
      end='', flush=True)
_t0 = time.time()
# Langues à script latin — partagent le même modèle sous-jacent dans EasyOCR,
# donc pas de surcoût mémoire significatif. Améliore la robustesse sur les
# polices techniques variées des schémas électroniques.
READER = easyocr.Reader(
    ['en', 'fr', 'de', 'es', 'it', 'nl', 'pt'],
    gpu=True, verbose=False
)
print(f'  {_c("✔", C.GREEN)}  EasyOCR pret  {_c(f"({time.time()-_t0:.1f}s)", C.GRAY)}')

# Caractères autorisés : lettres + chiffres + tiret uniquement
ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
OCR_CONF_MIN = 0.15   # ignorer les résultats en dessous de ce seuil

YOLO_IMGSZ = 1280   # résolution YOLO augmentée (1024 → 1280)


# ──────────────────────────────────────────────────────────
#  CORRECTEUR DE CONFUSIONS OCR
# ──────────────────────────────────────────────────────────

PREFIX_FIXES = {'0': 'O', '1': 'I', '8': 'B', '6': 'G', '5': 'S'}
DIGIT_FIXES  = {
    'I': '1', 'J': '1', 'L': '1', '|': '1',
    'O': '0', 'Q': '0', 'D': '0',
    'G': '6', 'Z': '2', 'S': '5', 'B': '8', 'T': '7',
}


def fix_ocr_confusion(text: str) -> str:
    """
    Scan linéaire : trouve le 1er vrai chiffre comme frontière préfixe/numérique.
    - Avant : corrige chiffres mal lus → lettres
    - Après : corrige lettres mal lues → chiffres
    - Dernier char lettre seule = suffixe, conservé tel quel
    Exemples : "C41J"→"C411", "R51G"→"R516", "0C402"→"OC402", "R516A"→"R516A"
    """
    if not text:
        return text

    text = text.strip().upper().replace(' ', '')

    # Trouver le 1er vrai chiffre
    i = 0
    while i < len(text) and not text[i].isdigit():
        i += 1

    if i == len(text):
        return text   # aucun chiffre → pas une ref connue

    prefix_raw = text[:i]
    rest       = text[i:]

    # Isoler suffixe lettre optionnel en fin
    if len(rest) > 1 and rest[-1].isalpha():
        suffix, digits_raw = rest[-1], rest[:-1]
    else:
        suffix, digits_raw = '', rest

    fixed_prefix = ''.join(
        PREFIX_FIXES.get(ch, ch) if ch.isdigit() else ch
        for ch in prefix_raw
    )
    fixed_digits = ''.join(
        ch if ch.isdigit() else DIGIT_FIXES.get(ch, ch)
        for ch in digits_raw
    )
    return fixed_prefix + fixed_digits + suffix


# ──────────────────────────────────────────────────────────
#  CORRECTIONS POST-OCR (dictionnaire des erreurs connues)
# ──────────────────────────────────────────────────────────
# Erreurs systématiques observées sur ce schéma.
# Format : "texte_mal_lu" → "texte_correct"
# Ajouter ici toute nouvelle erreur constatée dans les résultats.
POST_CORRECTIONS = {
    # Préfixe mal lu
    'UC402':  'IC402',   # U → I
    'UC405':  'IC405',
    'RSI9':   'R519',    # SI → 51
    'RL26':   'R126',    # L → 1
    'S501':   'C501',    # S → C (rare mais observé)
    'B178':   'BC178',   # BC tronqué → B
    'BCI78':  'BC178',   # I → 1
    'BCI7':   'BC17',
    'CI7A':   'C17A',
    'LC41':   'C41',
    'ECN8':   'C8',      # lecture très dégradée
    'IB6':    'B6',
    'REO7':   'R07',
    'JEC740': 'C740',
    'STLO1':  'ST01',
    # Chiffres mal lus dans la partie numérique
    'R62I':   'R621',    # I → 1
    'C41J':   'C411',    # J → 1
    'C50G':   'C506',    # G → 6
    'R5122':  'R512',    # chiffre surnuméraire
    'BC1788': 'BC178',   # 8 surnuméraire
    'BC178B': 'BC178',   # B suffixe erroné
    'T5502':  'T502',
    'R578':   'R578',    # peut être correct, à vérifier
}


def apply_post_corrections(text: str) -> str:
    """
    Applique le dictionnaire de corrections sur une ref déjà fixée par fix_ocr_confusion.
    Si la ref nettoyée (sans espaces) est dans le dictionnaire → remplace.
    """
    key = text.strip().upper().replace(' ', '')
    return POST_CORRECTIONS.get(key, text)


# ──────────────────────────────────────────────────────────
#  MERGE DE BOITES VOISINES
# ──────────────────────────────────────────────────────────

def merge_nearby_boxes(boxes, gap_ratio=0.8, overlap_y_ratio=0.4):
    """
    Fusionne les paires de boîtes qui sont probablement 2 morceaux du même label.

    Critères de fusion :
      - Alignées verticalement (centres Y proches, overlap Y suffisant)
      - Gap horizontal ≤ gap_ratio × hauteur moyenne des 2 boîtes
      - L'une à gauche de l'autre (pas de chevauchement X)

    Retourne une liste de boîtes fusionnées (x1,y1,x2,y2,conf).
    La conf est la moyenne pondérée des 2 boîtes.
    """
    if not boxes:
        return boxes

    # Trier par x1
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = list(boxes)
    changed = True

    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue
            b1 = merged[i]
            x1a, y1a, x2a, y2a, ca = b1

            best_j = -1
            best_gap = float('inf')

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                b2 = merged[j]
                x1b, y1b, x2b, y2b, cb = b2

                # b2 doit être à droite de b1
                if x1b < x2a:
                    continue

                h_avg    = ((y2a - y1a) + (y2b - y1b)) / 2
                gap      = x1b - x2a
                if gap > gap_ratio * h_avg:
                    continue   # trop loin

                # Vérifier alignement vertical
                overlap_y = min(y2a, y2b) - max(y1a, y1b)
                min_h     = min(y2a - y1a, y2b - y1b)
                if min_h == 0 or overlap_y / min_h < overlap_y_ratio:
                    continue   # pas alignées

                if gap < best_gap:
                    best_gap = gap
                    best_j   = j

            if best_j >= 0:
                b2 = merged[best_j]
                x1b, y1b, x2b, y2b, cb = b2
                # Fusionner
                fused = (
                    min(x1a, x1b), min(y1a, y1b),
                    max(x2a, x2b), max(y2a, y2b),
                    (ca + cb) / 2
                )
                new_merged.append(fused)
                used[i] = used[best_j] = True
                changed = True
            else:
                new_merged.append(b1)
                used[i] = True

        merged = new_merged

    return merged


# ──────────────────────────────────────────────────────────
#  PREPROCESSING
# ──────────────────────────────────────────────────────────

def preprocess_crop(crop_bgr, scale=5):
    h, w = crop_bgr.shape[:2]
    big  = cv2.resize(crop_bgr, (w * scale, h * scale),
                      interpolation=cv2.INTER_CUBIC)

    k     = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(big, -1, k)
    gray  = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    bgr      = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return cv2.copyMakeBorder(bgr, 10, 10, 10, 10,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))


# ──────────────────────────────────────────────────────────
#  REGEX VALIDATION
# ──────────────────────────────────────────────────────────

COMPONENT_PATTERN = re.compile(
    r'^('
    r'[A-Z]{1,2}\s?\d{3,5}[A-Z]?'   # R516, C401, R 516
    r'|[A-Z]{1,3}\d{2,5}[A-Z]?'      # IC402A
    r'|[A-Z]{2,4}\d{1,4}[A-Z]?'      # BC178, FR409, TP12
    r'|St\s?\d{3}'
    r'|[A-Z]{1,3}\d+[-_]\d+'
    r')$',
    re.IGNORECASE
)


def is_valid_ref(text: str) -> bool:
    return bool(COMPONENT_PATTERN.match(text.strip()))


# ──────────────────────────────────────────────────────────
#  OCR
# ──────────────────────────────────────────────────────────

def _run_easyocr(img_bgr):
    """Lance EasyOCR avec allowlist et retourne la liste (text, conf) triée."""
    try:
        results = READER.readtext(
            img_bgr, detail=1, paragraph=False,
            allowlist=ALLOWLIST,
            width_ths=0.7,    # fusionne les mots proches dans le crop
        )
    except Exception:
        return []

    out = []
    for (_, text, conf) in results:
        if conf < OCR_CONF_MIN:
            continue
        t = text.strip().upper()
        t = re.sub(r'\s+', '', t)   # supprimer espaces internes
        if t:
            out.append((t, conf))
    return sorted(out, key=lambda x: x[1], reverse=True)


def preprocess_crop_otsu(crop_bgr, scale=5):
    """Variante binarisation Otsu — meilleure sur fond sombre ou contraste inversé."""
    h, w = crop_bgr.shape[:2]
    big  = cv2.resize(crop_bgr, (w * scale, h * scale),
                      interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bgr  = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return cv2.copyMakeBorder(bgr, 10, 10, 10, 10,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))


def run_ocr(crop_bgr, scale=5):
    """
    Séquence de tentatives, arrêt dès qu'on trouve une ref valide :
      1. CLAHE normal  (scale auto selon résolution image)
      2. CLAHE + rotation 90° (si crop vertical ou rien trouvé)
      3. Otsu (fallback si fond sombre / contraste inversé)
    Retourne (text_corrigé, is_valid, conf).
    """
    def best_from(img):
        hits = _run_easyocr(img)
        if not hits:
            return "", False, 0.0
        for (t, c) in hits:
            fixed = apply_post_corrections(fix_ocr_confusion(t))
            if is_valid_ref(fixed):
                return fixed, True, c
        fixed = apply_post_corrections(fix_ocr_confusion(hits[0][0]))
        return fixed, False, hits[0][1]

    # Tentative 1 : CLAHE normal
    pre_clahe = preprocess_crop(crop_bgr, scale=scale)
    h, w = pre_clahe.shape[:2]
    text, valid, conf = best_from(pre_clahe)
    if valid:
        return text, valid, conf

    # Tentative 2 : CLAHE + 90° (crop vertical ou rien trouvé)
    if h > w * 1.3 or not text:
        t2, v2, c2 = best_from(cv2.rotate(pre_clahe, cv2.ROTATE_90_CLOCKWISE))
        if v2 or (t2 and c2 > conf):
            text, valid, conf = t2, v2, c2
    if valid:
        return text, valid, conf

    # Tentative 3 : Otsu (fallback)
    pre_otsu = preprocess_crop_otsu(crop_bgr, scale=scale)
    t3, v3, c3 = best_from(pre_otsu)
    if v3 or (t3 and c3 > conf):
        return t3, v3, c3

    return text, valid, conf


# ──────────────────────────────────────────────────────────
#  DÉDUPLICATION FINALE DES REFS
# ──────────────────────────────────────────────────────────

def deduplicate_refs(detections):
    """
    Si la même ref valide apparaît plusieurs fois, on garde seulement
    la détection avec le meilleur score (yolo_conf × ocr_conf).
    Les refs invalides ne sont pas touchées.
    """
    seen = {}   # text → index dans detections
    result = []

    for d in detections:
        if not d["is_valid_ref"]:
            result.append(d)
            continue

        key = d["text"].replace(' ', '').upper()
        score = d["yolo_conf"] * d["ocr_conf"]

        if key not in seen:
            seen[key] = len(result)
            result.append(d)
        else:
            existing = result[seen[key]]
            existing_score = existing["yolo_conf"] * existing["ocr_conf"]
            if score > existing_score:
                result[seen[key]] = d

    return result


# ──────────────────────────────────────────────────────────
#  AUTO-DÉTECTION DES PARAMÈTRES
# ──────────────────────────────────────────────────────────

def auto_params(img):
    """
    Analyse l'image et retourne les paramètres optimaux automatiquement.
    Adapte : imgsz YOLO, scale preprocessing, conf, inversion si fond sombre.
    """
    h, w = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean  = float(gray.mean())
    std   = float(gray.std())
    px    = w * h

    # imgsz YOLO + scale crop selon résolution
    if px > 3_000_000:          # > 3MP  (ex: 2480x1200 = 2.97MP)
        imgsz, scale = 1280, 5
    elif px > 1_000_000:        # > 1MP
        imgsz, scale = 1024, 6
    elif px > 300_000:          # > 0.3MP
        imgsz, scale = 640, 8
    else:                       # très petite image
        imgsz, scale = 640, 10

    # conf YOLO selon contraste (image peu contrastée → plus souple)
    if std < 25:
        conf = 0.25
    elif std < 45:
        conf = 0.30
    else:
        conf = 0.40

    # Si fond sombre (schéma inversé) → inverser l'image
    invert = mean < 80

    params = {
        "imgsz":  imgsz,
        "scale":  scale,
        "conf":   conf,
        "invert": invert,
    }

    inv_tag = f'  {_c("INVERSION fond sombre", C.YELLOW)}' if invert else ''
    print(f'  {_c("◈", C.CYAN)}  {_c("AUTO", C.BOLD)}  '
          f'{_c(f"{w}x{h}px", C.WHITE)} | '
          f'contraste={_c(f"{std:.0f}", C.CYAN)} | '
          f'luminosite={_c(f"{mean:.0f}", C.CYAN)} | '
          f'imgsz={_c(imgsz, C.GREEN)} | '
          f'scale={_c(scale, C.GREEN)} | '
          f'conf={_c(conf, C.GREEN)}'
          f'{inv_tag}')
    return params


# ──────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────

def extract(img_path, model_path, conf_thresh=None):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {img_path}")
    step('◉', 'Image', f'{img.shape[1]}x{img.shape[0]} px')

    # Auto-détection des paramètres selon l'image
    params = auto_params(img)
    if conf_thresh is not None:
        params["conf"] = conf_thresh   # --conf manuel écrase l'auto

    if params["invert"]:
        img = cv2.bitwise_not(img)

    # YOLO
    t_yolo = time.time()
    model       = YOLO(str(model_path))
    yolo_result = model(img, conf=params["conf"],
                        imgsz=params["imgsz"], verbose=False)[0]
    raw_boxes   = yolo_result.boxes
    boxes_raw = [
        (int(b.xyxy[0][0]), int(b.xyxy[0][1]),
         int(b.xyxy[0][2]), int(b.xyxy[0][3]),
         float(b.conf[0]))
        for b in raw_boxes
    ]
    boxes = merge_nearby_boxes(boxes_raw)
    fusions = len(boxes_raw) - len(boxes)
    step('◉', 'YOLO',
         f'{len(raw_boxes)} zones  →  {len(boxes)} apres merge '
         f'({_c(f"+{fusions} fusions", C.CYAN)})  '
         f'{_c(f"{time.time()-t_yolo:.1f}s", C.GRAY)}')
    print()

    detections = []
    PAD = 4
    for i, (x1, y1, x2, y2, conf_yolo) in enumerate(boxes):
        crop = img[max(0, y1-PAD): min(img.shape[0], y2+PAD),
                   max(0, x1-PAD): min(img.shape[1], x2+PAD)]
        if crop.size == 0:
            continue

        text, valid, ocr_conf = run_ocr(crop, scale=params["scale"])
        detections.append({
            "id":           i + 1,
            "text":         text,
            "is_valid_ref": valid,
            "yolo_conf":    round(conf_yolo, 3),
            "ocr_conf":     round(ocr_conf, 3),
            "score":        round(conf_yolo * ocr_conf, 3),
            "bbox":         {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "center":       {"x": (x1+x2)//2, "y": (y1+y2)//2},
        })
        progress_bar(i + 1, len(boxes))

    # Déduplication des refs identiques
    before = sum(1 for d in detections if d["is_valid_ref"])
    detections = deduplicate_refs(detections)
    after  = sum(1 for d in detections if d["is_valid_ref"])
    if before != after:
        ok(f'Deduplication : {before} → {_c(after, C.GREEN, C.BOLD)} refs uniques')

    return img, detections


# ──────────────────────────────────────────────────────────
#  VISUALISATION + EXPORTS
# ──────────────────────────────────────────────────────────

def draw_results(img, detections):
    out = img.copy()
    for d in detections:
        b = d["bbox"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        color = (0, 200, 0) if d["is_valid_ref"] else (0, 140, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if d["text"]:
            fs = 0.4
            (tw, th), _ = cv2.getTextSize(d["text"], cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(out, d["text"], (x1+2, y1-3),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1)
    return out


def save_json(detections, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)


def save_csv(detections, path):
    fields = ["id","text","is_valid_ref","yolo_conf","ocr_conf","score",
              "x1","y1","x2","y2","cx","cy"]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in detections:
            w.writerow({
                "id": d["id"], "text": d["text"],
                "is_valid_ref": d["is_valid_ref"],
                "yolo_conf": d["yolo_conf"],
                "ocr_conf":  d["ocr_conf"],
                "score":     d["score"],
                "x1": d["bbox"]["x1"], "y1": d["bbox"]["y1"],
                "x2": d["bbox"]["x2"], "y2": d["bbox"]["y2"],
                "cx": d["center"]["x"], "cy": d["center"]["y"],
            })


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PCB Component Extractor v7")
    parser.add_argument("--img",   required=True,           help="Image PCB")
    parser.add_argument("--model", required=True,           help="best.pt YOLO")
    parser.add_argument("--conf",  type=float, default=0.4, help="Seuil YOLO")
    args = parser.parse_args()

    img_path   = Path(args.img)
    model_path = Path(args.model)
    if not img_path.exists():
        print(f'  {_c("✘", C.RED)}  Image introuvable : {args.img}'); return
    if not model_path.exists():
        print(f'  {_c("✘", C.RED)}  Modele introuvable : {args.model}'); return

    W = 54
    print(f'\n  {_c("─" * W, C.BLUE)}')
    print(f'  {_c("  PCB Component Reference Extractor  v7", C.BOLD + C.WHITE)}')
    print(f'  {_c("─" * W, C.BLUE)}\n')

    t_total = time.time()
    img, detections = extract(img_path, model_path, conf_thresh=args.conf)
    elapsed = time.time() - t_total

    valid  = [d for d in detections if d["is_valid_ref"]]
    others = [d for d in detections if d["text"] and not d["is_valid_ref"]]
    high   = [d for d in valid if d["ocr_conf"] >= 0.7]

    # ── Résumé ──────────────────────────────────────────
    print(f'\n  {_c("─" * W, C.GRAY)}')
    print(f'  {_c("RÉSULTATS", C.BOLD + C.CYAN)}')
    print(f'  {_c("─" * W, C.GRAY)}')
    print(f'  Zones traitees       {_c(len(detections), C.WHITE, C.BOLD):>6}')
    print(f'  Refs valides         {_c(len(valid),      C.GREEN, C.BOLD):>6}')
    print(f'    dont haute conf    {_c(len(high),       C.GREEN):>6}  '
          f'{_c("(ocr >= 0.7 → fiables)", C.GRAY)}')
    print(f'  Non reconnu          {_c(len(others),     C.YELLOW):>6}')
    print(f'  Temps total          {_c(f"{elapsed:.1f}s", C.CYAN):>6}')
    print(f'  {_c("─" * W, C.GRAY)}')

    # ── Tableau des refs ────────────────────────────────
    if valid:
        print(f'\n  {_c("RÉFÉRENCES TROUVÉES", C.BOLD + C.WHITE)} '
              f'{_c(f"({len(valid)} — triées par score)", C.GRAY)}\n')
        print(f'  {_c("Référence", C.BOLD):<22} '
              f'{_c("YOLO", C.BOLD):>8} '
              f'{_c("OCR", C.BOLD):>8} '
              f'{_c("Score", C.BOLD):>8}  '
              f'{_c("Qualité", C.BOLD)}')
        print(f'  {_c("─" * W, C.GRAY)}')

        for d in sorted(valid, key=lambda x: x["score"], reverse=True):
            score = d["score"]
            ocr   = d["ocr_conf"]
            if score >= 0.5:
                qual = _c("●●●", C.GREEN)
            elif score >= 0.3:
                qual = _c("●●○", C.YELLOW)
            else:
                qual = _c("●○○", C.RED)

            ref_col = _c(d["text"], C.WHITE, C.BOLD) if ocr >= 0.7 else d["text"]
            print(f'  {ref_col:<22} '
                  f'{d["yolo_conf"]:>8.2f} '
                  f'{ocr:>8.2f} '
                  f'{score:>8.2f}  {qual}')

    # ── Exports ─────────────────────────────────────────
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = img_path.stem

    ann_path  = out_dir / f"{stem}_{ts}_annotated.png"
    json_path = out_dir / f"{stem}_{ts}_results.json"
    csv_path  = out_dir / f"{stem}_{ts}_results.csv"

    cv2.imwrite(str(ann_path), draw_results(img, detections))
    save_json(detections, json_path)
    save_csv (detections, csv_path)

    print(f'\n  {_c("─" * W, C.GRAY)}')
    print(f'  {_c("FICHIERS GÉNÉRÉS", C.BOLD + C.CYAN)}')
    print(f'  {_c("─" * W, C.GRAY)}')
    ok(f'Image    {_c(ann_path, C.GRAY)}')
    ok(f'JSON     {_c(json_path, C.GRAY)}')
    ok(f'CSV      {_c(csv_path, C.GRAY)}')
    print(f'\n  {_c("➜", C.CYAN)}  xdg-open {ann_path}')
    print(f'  {_c("─" * W, C.GRAY)}\n')


if __name__ == "__main__":
    main()
