#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python3 ./test1.py IMAGES/table.png IMAGES/schema.png IMAGES/carte.png

Dépendances OCR : EasyOCR (remplace Tesseract)
  pip install easyocr
"""
import cv2
import re
import sys
import time
import numpy as np
from ultralytics import YOLO
import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

# ──────────────────────────────────────────────────────────
#  EASYOCR  (chargement unique au démarrage du module)
# ──────────────────────────────────────────────────────────
import easyocr

print("  ⟳  Chargement EasyOCR...", end='', flush=True)
_t0 = time.time()
READER = easyocr.Reader(
    ['en', 'fr', 'de', 'es', 'it', 'nl', 'pt'],
    gpu=True, verbose=False
)
print(f"  ✔  EasyOCR prêt  ({time.time() - _t0:.1f}s)")

ALLOWLIST    = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
OCR_CONF_MIN = 0.15

# ──────────────────────────────────────────────────────────
#  MODELE YOLO pour schéma/carte
# ──────────────────────────────────────────────────────────
PATH_TO_YOLO_MODEL = "bestr.pt"

try:
    yolo_model = YOLO(PATH_TO_YOLO_MODEL)
    print("  ✔  Modèle YOLO chargé.")
except Exception as e:
    print(f"  ✘  Erreur chargement YOLO : {e}")
    sys.exit(1)

PREFIXES_VALIDES = ['R', 'C', 'U', 'D', 'Q', 'L', 'J', 'SW', 'F', 'TP', 'RV', 'Y', 'DZ', 'IC']

# ──────────────────────────────────────────────────────────
#  CORRECTEUR OCR  (porté depuis extract_easyocr.py)
# ──────────────────────────────────────────────────────────
PREFIX_FIXES = {'0': 'O', '1': 'I', '8': 'B', '6': 'G', '5': 'S'}
DIGIT_FIXES  = {
    'I': '1', 'J': '1', 'L': '1', '|': '1',
    'O': '0', 'Q': '0', 'D': '0',
    'G': '6', 'Z': '2', 'S': '5', 'B': '8', 'T': '7',
}

POST_CORRECTIONS = {
    'UC402': 'IC402', 'UC405': 'IC405',
    'RSI9':  'R519',  'RL26':  'R126',
    'S501':  'C501',  'B178':  'BC178',
    'BCI78': 'BC178', 'BCI7':  'BC17',
    'CI7A':  'C17A',  'LC41':  'C41',
    'ECN8':  'C8',    'IB6':   'B6',
    'REO7':  'R07',   'JEC740':'C740',
    'STLO1': 'ST01',  'R62I':  'R621',
    'C41J':  'C411',  'C50G':  'C506',
    'R5122': 'R512',  'BC1788':'BC178',
    'BC178B':'BC178', 'T5502': 'T502',
}

COMPONENT_PATTERN = re.compile(
    r'^('
    r'[A-Z]{1,2}\s?\d{3,5}[A-Z]?'
    r'|[A-Z]{1,3}\d{2,5}[A-Z]?'
    r'|[A-Z]{2,4}\d{1,4}[A-Z]?'
    r'|St\s?\d{3}'
    r'|[A-Z]{1,3}\d+[-_]\d+'
    r')$',
    re.IGNORECASE
)


def fix_ocr_confusion(text: str) -> str:
    """Corrige confusions chiffres/lettres selon position préfixe/numérique."""
    if not text:
        return text
    text = text.strip().upper().replace(' ', '')
    i = 0
    while i < len(text) and not text[i].isdigit():
        i += 1
    if i == len(text):
        return text
    prefix_raw = text[:i]
    rest = text[i:]
    if len(rest) > 1 and rest[-1].isalpha():
        suffix, digits_raw = rest[-1], rest[:-1]
    else:
        suffix, digits_raw = '', rest
    fixed_prefix = ''.join(PREFIX_FIXES.get(ch, ch) if ch.isdigit() else ch for ch in prefix_raw)
    fixed_digits = ''.join(ch if ch.isdigit() else DIGIT_FIXES.get(ch, ch) for ch in digits_raw)
    return fixed_prefix + fixed_digits + suffix


def apply_post_corrections(text: str) -> str:
    key = text.strip().upper().replace(' ', '')
    return POST_CORRECTIONS.get(key, text)


def is_valid_ref(text: str) -> bool:
    return bool(COMPONENT_PATTERN.match(text.strip()))


# ──────────────────────────────────────────────────────────
#  PREPROCESSING
# ──────────────────────────────────────────────────────────

def preprocess_crop_clahe(crop_bgr, scale=4):
    """Upscale + netteté + CLAHE pour EasyOCR."""
    h, w = crop_bgr.shape[:2]
    big  = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    k    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(big, -1, k)
    gray  = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return cv2.copyMakeBorder(bgr, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def preprocess_crop_otsu(crop_bgr, scale=4):
    """Variante Otsu — meilleure sur fond sombre."""
    h, w = crop_bgr.shape[:2]
    big  = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return cv2.copyMakeBorder(bgr, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))


# ──────────────────────────────────────────────────────────
#  OCR EasyOCR
# ──────────────────────────────────────────────────────────

def _run_easyocr(img_bgr):
    """Lance EasyOCR et retourne la liste (text, conf) triée par confiance."""
    try:
        results = READER.readtext(
            img_bgr, detail=1, paragraph=False,
            allowlist=ALLOWLIST,
            width_ths=0.7,
        )
    except Exception:
        return []
    out = []
    for (_, text, conf) in results:
        if conf < OCR_CONF_MIN:
            continue
        t = text.strip().upper()
        t = re.sub(r'\s+', '', t)
        if t:
            out.append((t, conf))
    return sorted(out, key=lambda x: x[1], reverse=True)


def run_ocr_on_crop(crop_bgr, scale=4):
    """
    Essaie CLAHE → CLAHE+rotation → Otsu.
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

    pre_clahe = preprocess_crop_clahe(crop_bgr, scale=scale)
    h, w = pre_clahe.shape[:2]
    text, valid, conf = best_from(pre_clahe)
    if valid:
        return text, valid, conf

    # Tentative rotation 90° (labels verticaux fréquents sur les schémas)
    if h > w * 1.3 or not text:
        t2, v2, c2 = best_from(cv2.rotate(pre_clahe, cv2.ROTATE_90_CLOCKWISE))
        if v2 or (t2 and c2 > conf):
            text, valid, conf = t2, v2, c2
    if valid:
        return text, valid, conf

    # Fallback Otsu
    pre_otsu = preprocess_crop_otsu(crop_bgr, scale=scale)
    t3, v3, c3 = best_from(pre_otsu)
    if v3 or (t3 and c3 > conf):
        return t3, v3, c3

    return text, valid, conf


# ──────────────────────────────────────────────────────────
#  MERGE DE BOÎTES VOISINES  (porté depuis extract_easyocr.py)
# ──────────────────────────────────────────────────────────

def merge_nearby_boxes(boxes, gap_ratio=0.8, overlap_y_ratio=0.4):
    """Fusionne les boîtes YOLO proches qui correspondent au même label."""
    if not boxes:
        return boxes
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
            best_j, best_gap = -1, float('inf')
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                x1b, y1b, x2b, y2b, cb = merged[j]
                if x1b < x2a:
                    continue
                h_avg  = ((y2a - y1a) + (y2b - y1b)) / 2
                gap    = x1b - x2a
                if gap > gap_ratio * h_avg:
                    continue
                overlap_y = min(y2a, y2b) - max(y1a, y1b)
                min_h     = min(y2a - y1a, y2b - y1b)
                if min_h == 0 or overlap_y / min_h < overlap_y_ratio:
                    continue
                if gap < best_gap:
                    best_gap, best_j = gap, j
            if best_j >= 0:
                x1b, y1b, x2b, y2b, cb = merged[best_j]
                new_merged.append((min(x1a, x1b), min(y1a, y1b),
                                   max(x2a, x2b), max(y2a, y2b),
                                   (ca + cb) / 2))
                used[i] = used[best_j] = True
                changed = True
            else:
                new_merged.append(b1)
                used[i] = True
        merged = new_merged
    return merged


# ──────────────────────────────────────────────────────────
#  DÉTECTION DES RÉFÉRENCES
# ──────────────────────────────────────────────────────────

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def detect_refs_table(img):
    """
    Version améliorée : Zoom x2 + mémoire du préfixe pour éviter les oublis.
    """
    if img is None: return []
    h, w = img.shape[:2]
    
    # 1. On isole la colonne des refs (environ 22% de la largeur)
    crop = img[:, :int(w * 0.22)]
    
    # 2. AMÉLIORATION IMAGE : On double la taille pour aider l'OCR
    scale = 2
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 3. LECTURE OCR
    # On utilise ton READER global déjà initialisé
    results = READER.readtext(big, detail=1, paragraph=False, allowlist=ALLOWLIST)
    
    # Tri vertical pour garantir qu'on lit de haut en bas
    results.sort(key=lambda x: x[0][0][1])

    found_refs = []
    last_prefix = "R" # Préfixe par défaut (souvent Résistance)

    for (bbox_pts, text, conf) in results:
        if conf < OCR_CONF_MIN: continue
        
        # Nettoyage du texte lu
        raw_text = text.upper().replace(' ', '')
        
        # LOGIQUE DE PRÉFIXE : 
        # On cherche si la ligne commence par des lettres (ex: R, C, Q)
        prefix_match = re.match(r'^([A-Z]+)', raw_text)
        
        if prefix_match:
            current_prefix = prefix_match.group(1)
            # On stocke ce préfixe pour les lignes suivantes qui n'en auraient pas
            last_prefix = current_prefix 
            clean_text = raw_text
        else:
            # Si on ne lit que des chiffres, on rajoute le dernier préfixe connu
            if re.search(r'\d+', raw_text):
                clean_text = f"{last_prefix}{raw_text}"
            else:
                continue

        # Calcul des coordonnées (on divise par 'scale' pour revenir à la taille originale)
        xs = [p[0]/scale for p in bbox_pts]
        ys = [p[1]/scale for p in bbox_pts]
        
        # On remplit ton dictionnaire exactement comme avant pour ne pas casser l'affichage
        found_refs.append({
            "ref": clean_text,
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
            "cx": np.mean(xs),
            "cy": np.mean(ys)
        })
            
    return found_refs


def detect_refs_easyocr(img):
    """
    Analyse du Schéma ou de la Carte :
      1. YOLO localise les zones de texte (où ?)
      2. EasyOCR lit chaque zone avec correcteur OCR (quoi ?)
    Remplace l'ancien detect_refs_hybrid.
    """
    if img is None:
        return []

    results = yolo_model.predict(source=img, conf=0.20, imgsz=1280, verbose=False)
    PAD = 5
    raw_boxes = []
    for result in results:
        for box in result.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            raw_boxes.append((b[0], b[1], b[2], b[3], float(box.conf[0])))

    # Fusion des boîtes fragmentées (ex: "R" + "516" → "R516")
    boxes = merge_nearby_boxes(raw_boxes)

    found_refs = []
    for (x1, y1, x2, y2, conf_yolo) in boxes:
        crop = img[max(0, y1 - PAD):y2 + PAD, max(0, x1 - PAD):x2 + PAD]
        if crop.size == 0:
            continue
        text, valid, ocr_conf = run_ocr_on_crop(crop, scale=4)
        if text:
            found_refs.append({
                "ref":        text,
                "is_valid":   valid,
                "yolo_conf":  round(conf_yolo, 3),
                "ocr_conf":   round(ocr_conf, 3),
                "cx":         float((x1 + x2) / 2),
                "cy":         float((y1 + y2) / 2),
                "bbox":       (float(x1), float(y1), float(x2), float(y2)),
            })

    # Déduplication : garder la détection avec le meilleur score par référence
    seen = {}
    deduped = []
    for d in found_refs:
        key = d["ref"].upper().replace(' ', '')
        score = d["yolo_conf"] * d["ocr_conf"]
        if key not in seen:
            seen[key] = len(deduped)
            deduped.append(d)
        else:
            existing_score = deduped[seen[key]]["yolo_conf"] * deduped[seen[key]]["ocr_conf"]
            if score > existing_score:
                deduped[seen[key]] = d

    return deduped


# ──────────────────────────────────────────────────────────
#  INTERFACE GRAPHIQUE
# ──────────────────────────────────────────────────────────

def afficher(table_path, schema_path, carte_path=None):
    img_t = cv2.imread(table_path)
    img_s = cv2.imread(schema_path)
    img_c = cv2.imread(carte_path) if carte_path else None

    print("  ⟳  Analyse en cours...")
    t0 = time.time()

    # Table : EasyOCR colonne gauche
    dict_table  = {r["ref"]: r for r in detect_refs_table(img_t)}
    # Schéma : YOLO + EasyOCR
    dict_schema = {r["ref"]: r for r in detect_refs_easyocr(img_s)}
    # Carte  : YOLO + EasyOCR (si fournie)
    dict_carte  = {r["ref"]: r for r in detect_refs_easyocr(img_c)} if img_c is not None else {}

    print(f"  ✔  Bilan ({time.time()-t0:.1f}s) : "
          f"Table({len(dict_table)}) | Schéma({len(dict_schema)}) | Carte({len(dict_carte)})")

    # ── Mise en page ────────────────────────────────────────
    fig  = plt.figure(figsize=(18, 9))
    cols = 4 if img_c is not None else 3
    gs   = fig.add_gridspec(1, cols, width_ratios=[1.2, 2, 2, 2][:cols])

    ax_list   = fig.add_subplot(gs[0, 0])
    ax_table  = fig.add_subplot(gs[0, 1])
    ax_schema = fig.add_subplot(gs[0, 2])
    ax_carte  = fig.add_subplot(gs[0, 3]) if img_c is not None else None
    axes_images = [ax for ax in [ax_table, ax_schema, ax_carte] if ax is not None]

    ax_table.imshow(cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB))
    ax_schema.imshow(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB))
    if ax_carte is not None:
        ax_carte.imshow(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB))

    ax_table.set_title("Table (BOM)",   fontsize=11)
    ax_schema.set_title("Schéma",       fontsize=11)
    if ax_carte is not None:
        ax_carte.set_title("Carte PCB", fontsize=11)

    for ax in [ax_list] + axes_images:
        ax.axis("off")

    # ── Liste des refs (panneau gauche) ─────────────────────
    ids     = sorted(dict_table.keys(), key=natural_sort_key)
    colors  = list(mcolors.TABLEAU_COLORS.values())
    text_mapping, y_pos, last_prefix, color_idx = {}, 0.98, None, 0

    for ident in ids:
        m = re.match(r'([A-Z]+)', ident)
        prefix = m.group(1) if m else "?"
        if prefix != last_prefix:
            if last_prefix:
                y_pos -= 0.02
            ax_list.text(0.05, y_pos, f"--- {prefix} ---",
                         fontsize=10, fontweight='bold', alpha=0.7,
                         transform=ax_list.transAxes)
            y_pos -= 0.03
            color_idx = (color_idx + 1) % len(colors)

        txt_obj = ax_list.text(0.15, y_pos, ident,
                               fontsize=9, picker=5, fontweight='bold',
                               color=colors[color_idx],
                               transform=ax_list.transAxes)
        text_mapping[txt_obj] = ident
        last_prefix = prefix
        y_pos -= 0.025
        if y_pos < 0.02:
            break

    # ── Rectangles de focus ─────────────────────────────────
    rects    = {ax: patches.Rectangle((0, 0), 0, 0, edgecolor="red",
                                       facecolor="none", linewidth=2, visible=False)
                for ax in axes_images}
    for ax in axes_images:
        ax.add_patch(rects[ax])
    orig_lims = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in axes_images}

    def do_focus(ident):
        zoom  = 350
        found = False
        dicts = {ax_table: dict_table, ax_schema: dict_schema}
        if ax_carte is not None:
            dicts[ax_carte] = dict_carte

        for ax, dic in dicts.items():
            if ident in dic:
                info = dic[ident]
                ax.set_xlim(info["cx"] - zoom, info["cx"] + zoom)
                ax.set_ylim(info["cy"] + zoom, info["cy"] - zoom)
                b = info["bbox"]
                rects[ax].set_xy((b[0], b[1]))
                rects[ax].set_width(b[2] - b[0])
                rects[ax].set_height(b[3] - b[1])
                rects[ax].set_visible(True)
                found = True
            else:
                rects[ax].set_visible(False)

        if found:
            fig.suptitle(f"Focus : {ident}", color="red", fontsize=16, fontweight="bold")
            plt.draw()

    def on_click(event):
        if event.button == 3:   # clic droit → reset zoom
            for ax in axes_images:
                ax.set_xlim(orig_lims[ax][0])
                ax.set_ylim(orig_lims[ax][1])
                rects[ax].set_visible(False)
            fig.suptitle("")
            plt.draw()
        elif event.inaxes in axes_images and event.button == 1:
            dicts = {ax_table: dict_table, ax_schema: dict_schema}
            if ax_carte is not None:
                dicts[ax_carte] = dict_carte
            curr_dict = dicts.get(event.inaxes)
            if curr_dict:
                pts = np.array([[r["cx"], r["cy"]] for r in curr_dict.values()])
                if len(pts) > 0:
                    dist = np.linalg.norm(pts - [event.xdata, event.ydata], axis=1)
                    idx  = np.argmin(dist)
                    if dist[idx] < 200:
                        do_focus(list(curr_dict.keys())[idx])

    fig.canvas.mpl_connect("pick_event",         lambda e: do_focus(text_mapping[e.artist]))
    fig.canvas.mpl_connect("button_press_event",  on_click)
    fig.canvas.mpl_connect("scroll_event",        lambda e: on_scroll(e, axes_images))

    print("  ✔  Interface prête.")
    plt.tight_layout()
    plt.show()


def on_scroll(event, axes_images):
    if event.inaxes not in axes_images:
        return
    ax    = event.inaxes
    scale = 1 / 1.5 if event.button == 'up' else 1.5
    cur_xlim, cur_ylim = ax.get_xlim(), ax.get_ylim()
    new_w = (cur_xlim[1] - cur_xlim[0]) * scale
    new_h = (cur_ylim[1] - cur_ylim[0]) * scale
    ax.set_xlim([event.xdata - new_w / 2, event.xdata + new_w / 2])
    ax.set_ylim([event.ydata + new_h / 2, event.ydata - new_h / 2])
    plt.draw()


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        afficher(
            sys.argv[1],
            sys.argv[2],
            sys.argv[3] if len(sys.argv) > 3 else None
        )
    else:
        print("Usage: python3 test1.py table.png schema.png [carte.png]")
