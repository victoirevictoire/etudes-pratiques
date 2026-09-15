#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viewer.py
=========
Interface graphique interactive pour la visualisation et la localisation
de références de composants électroniques dans trois sources :
  - Table (BOM) : colonne de références lue par OCR
  - Schéma      : zones détectées par YOLO + OCR
  - Carte PCB   : zones détectées par YOLO + OCR (optionnel)

Fonctionnalités :
  - Clic gauche sur un composant dans la liste → zoom vers sa position
  - Clic gauche sur une image → zoom vers le composant le plus proche
  - Clic droit  → réinitialisation du zoom
  - Molette     → zoom manuel dans chaque vue

Usage standalone:
    python3 viewer.py table.png schema.png [carte.png]

Importation depuis pdf_to_viewer.py :
    from viewer import show_interface, detect_refs_table
"""

import cv2
import re
import sys
import time
import numpy as np
from ultralytics import YOLO
import matplotlib
try:
    matplotlib.use('TkAgg')   # Backend Tkinter — nécessaire pour affichage interactif
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import easyocr


#  CHARGEMENT EASYOCR (unique au démarrage du module)

# Le lecteur est initialisé une seule fois pour éviter les rechargements
# coûteux à chaque appel OCR.

print("Chargement EasyOCR...", end='', flush=True)
_t0 = time.time()
READER = easyocr.Reader(
    ['en', 'fr', 'de', 'es', 'it', 'nl', 'pt'],   # Langues à script latin (même modèle sous-jacent)
    gpu=True, verbose=False
)
print(f"EasyOCR prêt  ({time.time() - _t0:.1f}s)")

# Caractères autorisés pour l'OCR (évite la détection de symboles parasites)
ALLOWLIST    = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'

# Seuil minimal de confiance OCR en dessous duquel le résultat est ignoré
OCR_CONF_MIN = 0.15


#  CHARGEMENT DU MODÈLE YOLO (détection de références sur schéma/carte)

YOLO_REFS_MODEL_PATH = "best_composants.pt"   # Modèle entraîné sur les références de composants

try:
    yolo_refs_model = YOLO(YOLO_REFS_MODEL_PATH)
    print("Modèle YOLO chargé.")
except Exception as e:
    print(f"Erreur chargement YOLO : {e}")
    sys.exit(1)

# Préfixes valides pour la validation d'une référence de composant
VALID_COMPONENT_PREFIXES = ['R', 'C', 'U', 'D', 'Q', 'L', 'J', 'SW', 'F', 'TP', 'RV', 'Y', 'DZ', 'IC']


#  CORRECTEUR OCR — CONFUSIONS CHIFFRES / LETTRES

# Corrections applicables dans la partie préfixe (avant le premier chiffre)
# Ex: "0C402" → "OC402" (0 mal lu à la place de O)
PREFIX_FIXES = {'0': 'O', '1': 'I', '8': 'B', '6': 'G', '5': 'S'}

# Corrections applicables dans la partie numérique (après le premier chiffre)
# Ex: "R51G" → "R516" (G mal lu à la place de 6)
DIGIT_FIXES = {
    'I': '1', 'J': '1', 'L': '1', '|': '1',
    'O': '0', 'Q': '0', 'D': '0',
    'G': '6', 'Z': '2', 'S': '5', 'B': '8', 'T': '7',
}

# Dictionnaire des erreurs systématiques observées sur ce schéma spécifique
# Format : "texte_mal_lu" → "texte_correct"
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

# Expression régulière pour valider qu'un texte est une référence de composant
# Couvre : R516, C401, IC402A, BC178, TP12, ST 301, R12-3, etc.
COMPONENT_PATTERN = re.compile(
    r'^('
    r'[A-Z]{1,2}\s?\d{3,5}[A-Z]?'   # Ex: R516, C401, R 516
    r'|[A-Z]{1,3}\d{2,5}[A-Z]?'      # Ex: IC402A
    r'|[A-Z]{2,4}\d{1,4}[A-Z]?'      # Ex: BC178, FR409, TP12
    r'|St\s?\d{3}'                    # Ex: ST301
    r'|[A-Z]{1,3}\d+[-_]\d+'         # Ex: R12-3
    r')$',
    re.IGNORECASE
)


def fix_ocr_confusion(text: str) -> str:
    """
    Corrige les confusions chiffres/lettres selon leur position dans la référence.

    Logique :
      - Avant le premier chiffre (préfixe) : corrige les chiffres mal lus en lettres
      - Après le premier chiffre (partie numérique) : corrige les lettres mal lues en chiffres
      - Dernier caractère si lettre seule = suffixe optionnel, conservé tel quel

    Exemples : "C41J" → "C411", "R51G" → "R516", "0C402" → "OC402"
    """
    if not text:
        return text
    text = text.strip().upper().replace(' ', '')

    # Trouver l'index du premier chiffre (frontière préfixe/numérique)
    i = 0
    while i < len(text) and not text[i].isdigit():
        i += 1

    if i == len(text):
        return text   # Aucun chiffre → pas une référence connue

    prefix_raw = text[:i]
    rest       = text[i:]

    # Isoler le suffixe lettre optionnel en fin de référence (ex: "A" dans "IC402A")
    if len(rest) > 1 and rest[-1].isalpha():
        suffix, digits_raw = rest[-1], rest[:-1]
    else:
        suffix, digits_raw = '', rest

    fixed_prefix = ''.join(PREFIX_FIXES.get(ch, ch) if ch.isdigit() else ch for ch in prefix_raw)
    fixed_digits = ''.join(ch if ch.isdigit() else DIGIT_FIXES.get(ch, ch) for ch in digits_raw)
    return fixed_prefix + fixed_digits + suffix


def apply_post_corrections(text: str) -> str:
    """
    Applique le dictionnaire de corrections d'erreurs systématiques (POST_CORRECTIONS).
    À appeler après fix_ocr_confusion pour couvrir les cas non corrigibles par règle.
    """
    key = text.strip().upper().replace(' ', '')
    return POST_CORRECTIONS.get(key, text)


def is_valid_component_ref(text: str) -> bool:
    """Retourne True si le texte correspond au format d'une référence de composant."""
    return bool(COMPONENT_PATTERN.match(text.strip()))


#  PREPROCESSING DES CROPS POUR L'OCR

def preprocess_crop_clahe(crop_bgr, scale=4):
    """
    Prétraitement CLAHE : upscale + filtre de netteté + égalisation adaptative.
    Méthode principale, adaptée aux fonds clairs standard.

    Paramètres
    ----------
    crop_bgr : image BGR découpée autour d'une zone de texte
    scale    : facteur d'agrandissement avant OCR (défaut 4×)
    """
    h, w  = crop_bgr.shape[:2]
    big   = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Filtre de netteté (Laplacian sharpening)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp  = cv2.filter2D(big, -1, kernel)

    # Égalisation adaptative (CLAHE) sur canal gris
    gray     = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Reconversion en BGR + bordure blanche pour éviter les artifacts OCR aux bords
    bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return cv2.copyMakeBorder(bgr, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def preprocess_crop_otsu(crop_bgr, scale=4):
    """
    Prétraitement Otsu : binarisation par seuillage automatique.
    Méthode de secours, plus robuste sur les fonds sombres ou contrastés.
    """
    h, w = crop_bgr.shape[:2]
    big  = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bgr  = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return cv2.copyMakeBorder(bgr, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))


#  OCR EASYOCR

def _run_easyocr_on_image(img_bgr):
    """
    Lance EasyOCR sur une image BGR et retourne la liste (text, conf) triée par confiance.
    Les résultats sous OCR_CONF_MIN sont filtrés.
    """
    try:
        results = READER.readtext(
            img_bgr, detail=1, paragraph=False,
            allowlist=ALLOWLIST,
            width_ths=0.7,   # Fusionne les mots proches dans le crop
        )
    except Exception:
        return []

    out = []
    for (_, text, conf) in results:
        if conf < OCR_CONF_MIN:
            continue
        t = text.strip().upper()
        t = re.sub(r'\s+', '', t)   # Suppression des espaces internes
        if t:
            out.append((t, conf))

    return sorted(out, key=lambda x: x[1], reverse=True)


def run_ocr_on_crop(crop_bgr, scale=4):
    """
    Essaie plusieurs stratégies d'OCR, s'arrête dès qu'une référence valide est trouvée.

    Stratégies (dans l'ordre) :
      1. CLAHE normal
      2. CLAHE + rotation 90° (pour les labels verticaux fréquents sur les schémas)
      3. Otsu (fallback pour fonds sombres ou inversés)

    Retourne
    --------
    (text_corrigé, is_valid, conf)
    """
    def best_match_from(img):
        """Tente l'OCR sur img, retourne la meilleure référence corrigée."""
        hits = _run_easyocr_on_image(img)
        if not hits:
            return "", False, 0.0
        # On cherche en priorité une référence valide
        for (t, c) in hits:
            fixed = apply_post_corrections(fix_ocr_confusion(t))
            if is_valid_component_ref(fixed):
                return fixed, True, c
        # Sinon on retourne le meilleur résultat même invalide
        fixed = apply_post_corrections(fix_ocr_confusion(hits[0][0]))
        return fixed, False, hits[0][1]

    # Tentative 1 : CLAHE standard
    pre_clahe = preprocess_crop_clahe(crop_bgr, scale=scale)
    h, w = pre_clahe.shape[:2]
    text, valid, conf = best_match_from(pre_clahe)
    if valid:
        return text, valid, conf

    # Tentative 2 : rotation 90° (labels verticaux ou rien trouvé)
    if h > w * 1.3 or not text:
        t2, v2, c2 = best_match_from(cv2.rotate(pre_clahe, cv2.ROTATE_90_CLOCKWISE))
        if v2 or (t2 and c2 > conf):
            text, valid, conf = t2, v2, c2
    if valid:
        return text, valid, conf

    # Tentative 3 : Otsu (fallback)
    pre_otsu = preprocess_crop_otsu(crop_bgr, scale=scale)
    t3, v3, c3 = best_match_from(pre_otsu)
    if v3 or (t3 and c3 > conf):
        return t3, v3, c3

    return text, valid, conf


#  FUSION DES BOÎTES YOLO FRAGMENTÉES

def merge_nearby_boxes(boxes, gap_ratio=0.8, overlap_y_ratio=0.4):
    """
    Fusionne les paires de boîtes YOLO qui correspondent probablement au même label.
    Utile quand YOLO fragmente une référence en deux (ex: "R" + "516" → "R516").

    Critères de fusion :
      - La boîte B est à droite de A (pas de chevauchement X)
      - Gap horizontal ≤ gap_ratio × hauteur moyenne des deux boîtes
      - Chevauchement vertical ≥ overlap_y_ratio × hauteur de la plus petite boîte

    Paramètres
    ----------
    boxes           : liste de tuples (x1, y1, x2, y2, conf)
    gap_ratio       : distance horizontale max relative à la hauteur (défaut 0.8)
    overlap_y_ratio : chevauchement vertical minimum pour considérer l'alignement (défaut 0.4)

    Retourne une liste de boîtes fusionnées, conf = moyenne des deux boîtes.
    """
    if not boxes:
        return boxes

    boxes  = sorted(boxes, key=lambda b: b[0])   # Tri par x1 (gauche -> droite)
    merged = list(boxes)
    changed = True

    while changed:
        changed   = False
        new_merged = []
        used      = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue
            x1a, y1a, x2a, y2a, ca = merged[i]
            best_j, best_gap = -1, float('inf')

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                x1b, y1b, x2b, y2b, cb = merged[j]

                # B doit être strictement à droite de A
                if x1b < x2a:
                    continue

                # Vérification du gap horizontal
                h_avg = ((y2a - y1a) + (y2b - y1b)) / 2
                gap   = x1b - x2a
                if gap > gap_ratio * h_avg:
                    continue

                # Vérification de l'alignement vertical
                overlap_y = min(y2a, y2b) - max(y1a, y1b)
                min_h     = min(y2a - y1a, y2b - y1b)
                if min_h == 0 or overlap_y / min_h < overlap_y_ratio:
                    continue

                if gap < best_gap:
                    best_gap, best_j = gap, j

            if best_j >= 0:
                x1b, y1b, x2b, y2b, cb = merged[best_j]
                # Fusion : boîte englobante + conf moyenne
                new_merged.append((
                    min(x1a, x1b), min(y1a, y1b),
                    max(x2a, x2b), max(y2a, y2b),
                    (ca + cb) / 2
                ))
                used[i] = used[best_j] = True
                changed = True
            else:
                new_merged.append(merged[i])
                used[i] = True

        merged = new_merged

    return merged


#  UTILITAIRE DE TRI NATUREL

def natural_sort_key(s):
    """
    Clé de tri naturel : trie les chaînes comme un humain.
    Ex: ["R2", "R10", "R1"] → ["R1", "R2", "R10"] (et non ["R1", "R10", "R2"])
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


#  DÉTECTION DES RÉFÉRENCES

def detect_refs_table(img):
    """
    Détecte les références de composants dans la colonne gauche d'une image de table (BOM).

    Stratégie :
      1. Isole les 22% gauches de l'image (colonne des références)
      2. Upscale ×2 pour améliorer la lisibilité OCR
      3. Lit les références avec EasyOCR de haut en bas
      4. Reconstitue les références incomplètes en mémorisant le dernier préfixe lu

    Retourne une liste de dicts {"ref", "bbox", "cx", "cy"}.
    """
    if img is None:
        return []

    h, w = img.shape[:2]

    # Isoler la colonne de gauche contenant les références (environ 22% de la largeur)
    ref_column = img[:, :int(w * 0.22)]

    # Upscale ×2 pour améliorer la précision OCR sur les petits caractères
    scale = 2
    ref_column_big = cv2.resize(ref_column, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Lecture OCR de la colonne de références
    results = READER.readtext(ref_column_big, detail=1, paragraph=False, allowlist=ALLOWLIST)

    # Tri de haut en bas selon la coordonnée Y du coin supérieur gauche
    results.sort(key=lambda x: x[0][0][1])

    found_refs = []
    last_prefix = "R"   # Préfixe par défaut (Résistance, le plus courant)

    for (bbox_pts, text, conf) in results:
        if conf < OCR_CONF_MIN:
            continue

        raw_text = text.upper().replace(' ', '')

        # Détection du préfixe alphabétique en début de texte
        prefix_match = re.match(r'^([A-Z]+)', raw_text)

        if prefix_match:
            # La ligne contient un préfixe → référence complète
            current_prefix = prefix_match.group(1)
            last_prefix    = current_prefix   # Mémorisation pour les lignes suivantes
            clean_text     = raw_text
        else:
            # Pas de préfixe → OCR a lu uniquement les chiffres
            # On reconstitue la référence avec le dernier préfixe connu
            if re.search(r'\d+', raw_text):
                clean_text = f"{last_prefix}{raw_text}"
            else:
                continue   # Ni chiffres ni préfixe → ligne ignorée

        # Recalcul des coordonnées dans l'image originale (avant upscale)
        xs = [p[0] / scale for p in bbox_pts]
        ys = [p[1] / scale for p in bbox_pts]

        found_refs.append({
            "ref":  clean_text,
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
            "cx":   np.mean(xs),
            "cy":   np.mean(ys),
        })

    return found_refs


def detect_refs_on_image(img):
    """
    Détecte les références de composants sur un schéma ou une carte PCB.

    Pipeline :
      1. YOLO localise les zones de texte (où ?)
      2. Fusion des boîtes fragmentées (ex: "R" + "516" → zone unique)
      3. EasyOCR lit chaque zone avec correction des confusions OCR

    Retourne une liste de dicts {"ref", "is_valid", "yolo_conf", "ocr_conf", "cx", "cy", "bbox"}.
    """
    if img is None:
        return []

    # Détection YOLO des zones de texte
    results  = yolo_refs_model.predict(source=img, conf=0.20, imgsz=1280, verbose=False)
    PAD      = 5   # Marge autour de chaque zone pour ne pas rogner les bords
    raw_boxes = []

    for result in results:
        for box in result.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            raw_boxes.append((b[0], b[1], b[2], b[3], float(box.conf[0])))

    # Fusion des boîtes fragmentées avant OCR
    boxes = merge_nearby_boxes(raw_boxes)

    found_refs = []
    for (x1, y1, x2, y2, conf_yolo) in boxes:
        # Découpage de la zone avec marge
        crop = img[max(0, y1 - PAD):y2 + PAD, max(0, x1 - PAD):x2 + PAD]
        if crop.size == 0:
            continue

        text, valid, ocr_conf = run_ocr_on_crop(crop, scale=4)
        if text:
            found_refs.append({
                "ref":       text,
                "is_valid":  valid,
                "yolo_conf": round(conf_yolo, 3),
                "ocr_conf":  round(ocr_conf, 3),
                "cx":        float((x1 + x2) / 2),
                "cy":        float((y1 + y2) / 2),
                "bbox":      (float(x1), float(y1), float(x2), float(y2)),
            })

    # Déduplication : garder la détection avec le meilleur score composite par référence
    seen   = {}   # ref -> index dans deduped
    deduped = []
    for d in found_refs:
        key   = d["ref"].upper().replace(' ', '')
        score = d["yolo_conf"] * d["ocr_conf"]
        if key not in seen:
            seen[key] = len(deduped)
            deduped.append(d)
        else:
            existing_score = deduped[seen[key]]["yolo_conf"] * deduped[seen[key]]["ocr_conf"]
            if score > existing_score:
                deduped[seen[key]] = d

    return deduped


#  INTERFACE GRAPHIQUE

def show_interface(table_path: str, schema_path: str, board_path: str = None):
    """
    Affiche l'interface graphique interactive de visualisation croisée.

    Paramètres
    ----------
    table_path  : chemin vers l'image de la table (BOM)
    schema_path : chemin vers l'image du schéma électronique
    board_path  : chemin vers l'image de la carte PCB (optionnel)
    """
    # Chargement des images
    img_table  = cv2.imread(table_path)
    img_schema = cv2.imread(schema_path)
    img_board  = cv2.imread(board_path) if board_path else None

    print("Analyse OCR en cours...")
    t0 = time.time()

    # Détection des références dans chaque source
    dict_table  = {r["ref"]: r for r in detect_refs_table(img_table)}
    dict_schema = {r["ref"]: r for r in detect_refs_on_image(img_schema)}
    dict_board  = {r["ref"]: r for r in detect_refs_on_image(img_board)} if img_board is not None else {}

    print(f"Analyse terminée ({time.time()-t0:.1f}s) : "
          f"Table({len(dict_table)}) | Schéma({len(dict_schema)}) | Carte({len(dict_board)})")

    # ── Mise en page de la fenêtre
    fig  = plt.figure(figsize=(18, 9))
    cols = 4 if img_board is not None else 3
    gs   = fig.add_gridspec(1, cols, width_ratios=[1.2, 2, 2, 2][:cols])

    ax_list   = fig.add_subplot(gs[0, 0])   # Panneau gauche : liste des références
    ax_table  = fig.add_subplot(gs[0, 1])   # Vue Table (BOM)
    ax_schema = fig.add_subplot(gs[0, 2])   # Vue Schéma
    ax_board  = fig.add_subplot(gs[0, 3]) if img_board is not None else None

    # Axes avec images (utilisés pour le zoom et les clics)
    axes_images = [ax for ax in [ax_table, ax_schema, ax_board] if ax is not None]

    # Affichage des images (conversion BGR → RGB pour matplotlib)
    ax_table.imshow(cv2.cvtColor(img_table, cv2.COLOR_BGR2RGB))
    ax_schema.imshow(cv2.cvtColor(img_schema, cv2.COLOR_BGR2RGB))
    if ax_board is not None:
        ax_board.imshow(cv2.cvtColor(img_board, cv2.COLOR_BGR2RGB))

    ax_table.set_title("Table (BOM)", fontsize=11)
    ax_schema.set_title("Schéma",     fontsize=11)
    if ax_board is not None:
        ax_board.set_title("Carte PCB", fontsize=11)

    for ax in [ax_list] + axes_images:
        ax.axis("off")

    # Liste des références (panneau gauche)
    ids        = sorted(dict_table.keys(), key=natural_sort_key)
    colors     = list(mcolors.TABLEAU_COLORS.values())
    text_mapping                       = {}   # objet texte matplotlib → identifiant
    y_pos, last_prefix, color_idx = 0.98, None, 0

    for ident in ids:
        # Séparation visuelle par préfixe (R, C, U, ...)
        m      = re.match(r'([A-Z]+)', ident)
        prefix = m.group(1) if m else "?"

        if prefix != last_prefix:
            if last_prefix:
                y_pos -= 0.02   # Espacement entre groupes
            ax_list.text(0.05, y_pos, f"--- {prefix} ---",
                         fontsize=10, fontweight='bold', alpha=0.7,
                         transform=ax_list.transAxes)
            y_pos     -= 0.03
            color_idx  = (color_idx + 1) % len(colors)

        # Texte cliquable pour chaque référence
        txt_obj = ax_list.text(0.15, y_pos, ident,
                               fontsize=9, picker=5, fontweight='bold',
                               color=colors[color_idx],
                               transform=ax_list.transAxes)
        text_mapping[txt_obj] = ident
        last_prefix = prefix
        y_pos -= 0.025

        if y_pos < 0.02:
            break   # Plus de place dans le panneau

    #Rectangles de mise en évidence (focus)
    # Un rectangle rouge par vue image, initialement invisible
    focus_rects = {
        ax: patches.Rectangle((0, 0), 0, 0, edgecolor="red",
                               facecolor="none", linewidth=2, visible=False)
        for ax in axes_images
    }
    for ax in axes_images:
        ax.add_patch(focus_rects[ax])

    # Limites d'origine pour le reset de zoom (clic droit)
    orig_lims = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in axes_images}

    def focus_on_component(ident):
        """Zoome sur le composant 'ident' dans toutes les vues où il est trouvé."""
        zoom_radius = 350   # Demi-largeur/hauteur de la fenêtre de zoom en pixels
        found       = False

        view_dicts = {ax_table: dict_table, ax_schema: dict_schema}
        if ax_board is not None:
            view_dicts[ax_board] = dict_board

        for ax, component_dict in view_dicts.items():
            if ident in component_dict:
                info = component_dict[ident]
                # Zoom centré sur le composant
                ax.set_xlim(info["cx"] - zoom_radius, info["cx"] + zoom_radius)
                ax.set_ylim(info["cy"] + zoom_radius, info["cy"] - zoom_radius)
                # Affichage du rectangle de mise en évidence
                b = info["bbox"]
                focus_rects[ax].set_xy((b[0], b[1]))
                focus_rects[ax].set_width(b[2] - b[0])
                focus_rects[ax].set_height(b[3] - b[1])
                focus_rects[ax].set_visible(True)
                found = True
            else:
                focus_rects[ax].set_visible(False)

        if found:
            fig.suptitle(f"Focus : {ident}", color="red", fontsize=16, fontweight="bold")
            plt.draw()

    def on_mouse_click(event):
        """
        Gestionnaire de clic souris :
          - Clic gauche sur une vue → zoom vers le composant le plus proche
          - Clic droit → réinitialisation du zoom sur toutes les vues
        """
        if event.button == 3:
            # Clic droit : reset zoom + masquer rectangles
            for ax in axes_images:
                ax.set_xlim(orig_lims[ax][0])
                ax.set_ylim(orig_lims[ax][1])
                focus_rects[ax].set_visible(False)
            fig.suptitle("")
            plt.draw()

        elif event.inaxes in axes_images and event.button == 1:
            # Clic gauche dans une vue image : trouver le composant le plus proche
            view_dicts = {ax_table: dict_table, ax_schema: dict_schema}
            if ax_board is not None:
                view_dicts[ax_board] = dict_board

            curr_dict = view_dicts.get(event.inaxes)
            if curr_dict:
                pts = np.array([[r["cx"], r["cy"]] for r in curr_dict.values()])
                if len(pts) > 0:
                    dist = np.linalg.norm(pts - [event.xdata, event.ydata], axis=1)
                    idx  = np.argmin(dist)
                    if dist[idx] < 200:   # Seuil de proximité en pixels
                        focus_on_component(list(curr_dict.keys())[idx])

    # Connexion des événements interactifs
    fig.canvas.mpl_connect("pick_event",        lambda e: focus_on_component(text_mapping[e.artist]))
    fig.canvas.mpl_connect("button_press_event", on_mouse_click)
    fig.canvas.mpl_connect("scroll_event",       lambda e: on_scroll(e, axes_images))

    print("  Interface prête.")
    plt.tight_layout()
    plt.show()


def on_scroll(event, axes_images):
    """
    Gestionnaire de scroll souris : zoom in/out centré sur la position du curseur.
    Molette vers le haut -> zoom in (×1.5), vers le bas → zoom out (÷1.5).
    """
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


#  POINT D'ENTRÉE (exécution standalone)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        show_interface(
            table_path  = sys.argv[1],
            schema_path = sys.argv[2],
            board_path  = sys.argv[3] if len(sys.argv) > 3 else None
        )
    else:
        print("Usage: python3 viewer.py table.png schema.png [carte.png]")
