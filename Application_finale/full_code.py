#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pcb_pipeline_complet.py
=======================
Pipeline complet d'analyse de documents électroniques (PDF ou image) :
extraction de zones structurées (Table BOM, Schéma, Carte PCB) via YOLO,
reconnaissance optique des références de composants (EasyOCR) et interface
graphique interactive de visualisation croisée.

Ce module est la consolidation de quatre scripts indépendants :
  - pdf_zone_extractor.py  : extraction brute des zones YOLO depuis un PDF
  - pdf_to_viewer.py       : pipeline PDF → YOLO → viewer
  - ocr_ref_extractor.py   : extraction OCR des références sur image PCB seule
  - viewer.py              : interface graphique interactive

Modes d'utilisation (trois points d'entrée) :
  1. Extraction seule (zones du PDF vers fichiers PNG) :
       python3 pcb_pipeline_complet.py extract <document.pdf> <sortie/>

  2. Pipeline complet PDF → interface interactive :
       python3 pcb_pipeline_complet.py view <document.pdf> <modele_zones.pt>

  3. Extraction OCR sur image PCB seule :
       python3 pcb_pipeline_complet.py ocr --img carte.png --model best.pt [--conf 0.4]

Dépendances :
  - opencv-python (cv2)      : manipulation d'images, preprocessing
  - numpy                    : calculs matriciels sur les images
  - pdf2image                : conversion PDF → images PIL via Poppler
  - ultralytics (YOLO)       : détection d'objets (zones et références)
  - easyocr                  : OCR basé sur réseau CRNN + CRAFT/DBNET
  - matplotlib               : interface graphique interactive
"""

import os
import sys
import cv2
import re
import csv
import json
import time
import argparse
import numpy as np

from pathlib import Path
from datetime import datetime
from pdf2image import convert_from_path
from ultralytics import YOLO

import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

import easyocr


# ==============================================================================
# SECTION 1 — INITIALISATION EASYOCR (singleton de module)
# ==============================================================================
#
# EasyOCR utilise deux modèles de deep learning :
#   - CRAFT (Character Region Awareness For Text detection) : localise les
#     zones de texte dans l'image en générant une carte de chaleur de probabilité
#     de présence de caractères.
#   - CRNN (Convolutional Recurrent Neural Network) : lit le contenu textuel
#     de chaque zone détectée par CRAFT via une combinaison CNN + BiLSTM + CTC.
#
# Le chargement de ces modèles est coûteux (~2-5 s selon le matériel, GPU ou CPU).
# On l'effectue une seule fois au niveau module pour éviter toute réinitialisation
# lors de traitements multi-images.
#
# Langues choisies ('en', 'fr', 'de', 'es', 'it', 'nl', 'pt') : toutes à script
# latin et partageant le même modèle CRNN sous-jacent dans EasyOCR. Ajouter ces
# langues n'augmente donc pas significativement la consommation mémoire GPU mais
# améliore la robustesse sur les polices techniques variées des schémas électroniques
# (silkscreen, annotation CAO, police Gerber).
#
# gpu=True : utilise CUDA si disponible (nVidia), sinon bascule automatiquement
# sur le CPU sans erreur. verbose=False : supprime les logs Torch/CRNN.

print("Chargement EasyOCR...", end='', flush=True)
_t0_easyocr = time.time()
READER = easyocr.Reader(
    ['en', 'fr', 'de', 'es', 'it', 'nl', 'pt'],
    gpu=True,
    verbose=False,
)
print(f"  OK  ({time.time() - _t0_easyocr:.1f}s)")

# Allowlist OCR : restreint l'espace de caractères reconnaissables.
# En électronique, les références de composants n'utilisent que des lettres
# majuscules, des chiffres, des tirets et des underscores. Exclure les symboles,
# la ponctuation et les lettres accentuées élimine une grande partie des fausses
# détections sur les annotations graphiques du schéma (flèches, cadres, etc.).
ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'

# Seuil de confiance OCR minimal (score CTC normalisé entre 0 et 1).
# En dessous de ce seuil, le résultat EasyOCR est considéré comme non fiable
# et ignoré. Une valeur de 0.15 est volontairement basse pour ne pas manquer
# de vraies références sur des crops de faible qualité (petite taille, flou).
OCR_CONF_MIN = 0.15


# ==============================================================================
# SECTION 2 — AFFICHAGE TERMINAL (couleurs ANSI)
# ==============================================================================
#
# Classe utilitaire pour l'injection de codes de contrôle ANSI dans les chaînes
# de sortie terminal. Ces codes sont interprétés par les émulateurs de terminal
# compatibles VT100 (bash, zsh, xterm, gnome-terminal, Windows Terminal >= 1.4).
# Ils n'affectent pas les fichiers de log si la sortie est redirigée (le code
# RESET '\033[0m' clôture chaque séquence colorée).

class C:
    """Codes ANSI de mise en forme terminal."""
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
    """
    Entoure 'text' avec les codes ANSI fournis et termine par RESET.
    Permet de combiner plusieurs attributs : ex. _c("OK", C.GREEN, C.BOLD).

    Paramètres
    ----------
    text  : valeur à afficher (sera convertie en str)
    codes : un ou plusieurs codes ANSI (attributs de la classe C)
    """
    return ''.join(codes) + str(text) + C.RESET


def step(icon, label, value=''):
    """Affiche une ligne de progression formatée : icône + label en gras + valeur grisée."""
    val = f'  {C.GRAY}{value}{C.RESET}' if value else ''
    print(f'  {icon}  {C.BOLD}{label}{C.RESET}{val}')


def ok(label):
    """Affiche une ligne de succès verte."""
    print(f'  {_c("ok", C.GREEN)}  {label}')


def progress_bar(current, total, width=30):
    """
    Affiche une barre de progression ASCII sur une seule ligne (écrasée à chaque appel).
    Utilise '\r' (retour chariot sans saut de ligne) combiné à sys.stdout.write pour
    réécrire la même ligne à chaque appel. Le saut de ligne final n'est émis que
    lorsque current == total.

    Paramètres
    ----------
    current : index traité (1-based)
    total   : nombre total d'éléments
    width   : largeur de la barre en caractères (défaut 30)
    """
    pct    = current / total if total else 0
    filled = int(width * pct)
    bar    = _c('█' * filled, C.CYAN) + _c('░' * (width - filled), C.GRAY)
    sys.stdout.write(
        f'\r  {_c("up", C.CYAN)}  OCR  [{bar}]  '
        f'{_c(f"{current}/{total}", C.WHITE)}  '
        f'{_c(f"{pct*100:.0f}%", C.YELLOW)}'
    )
    sys.stdout.flush()
    if current == total:
        sys.stdout.write('\n')


# ==============================================================================
# SECTION 3 — CORRECTION DES CONFUSIONS OCR CHIFFRES/LETTRES
# ==============================================================================
#
# Les OCR entraînés sur du texte générique confondent fréquemment certains
# caractères visuellement proches dans les polices techniques : "O" / "0",
# "I" / "1", "B" / "8", etc.
#
# La stratégie adoptée exploite la structure prévisible des références de
# composants électroniques : PREFIX ALPHANUMÉRIQUE + PARTIE NUMÉRIQUE + SUFFIXE OPTIONNEL.
# Exemples : R516, C401, IC402A, BC178, TP12.
#
# Ce schéma permet de déduire la nature attendue de chaque caractère selon sa
# position et d'appliquer la correction inverse à la confusion probable :
#   - Dans le préfixe (avant le 1er chiffre) : un '0' est probablement un 'O',
#     un '1' est probablement un 'I', etc.
#   - Dans la partie numérique (après le 1er chiffre) : un 'I' est probablement
#     un '1', un 'O' est probablement un '0', un 'G' est probablement un '6', etc.
#   - Le dernier caractère si lettre seule = suffixe optionnel (ex: "A" dans
#     IC402A), conservé sans modification.

PREFIX_FIXES = {
    '0': 'O',
    '1': 'I',
    '8': 'B',
    '6': 'G',
    '5': 'S',
}

DIGIT_FIXES = {
    'I': '1', 'J': '1', 'L': '1', '|': '1',
    'O': '0', 'Q': '0', 'D': '0',
    'G': '6', 'Z': '2', 'S': '5', 'B': '8', 'T': '7',
}

# Dictionnaire de corrections d'erreurs systématiques observées sur le
# schéma de référence (Accuphase E202). Ces erreurs persistent après
# l'application des règles génériques ci-dessus car elles résultent de
# confusions multi-caractères (ex: "UC" → "IC", "RL" → "R1") ou d'artefacts
# spécifiques à la qualité du scan original.
# Format : "texte_mal_lu" (clé) → "texte_correct" (valeur).
POST_CORRECTIONS = {
    'UC402':  'IC402',
    'UC405':  'IC405',
    'RSI9':   'R519',
    'RL26':   'R126',
    'S501':   'C501',
    'B178':   'BC178',
    'BCI78':  'BC178',
    'BCI7':   'BC17',
    'CI7A':   'C17A',
    'LC41':   'C41',
    'ECN8':   'C8',
    'IB6':    'B6',
    'REO7':   'R07',
    'JEC740': 'C740',
    'STLO1':  'ST01',
    'R62I':   'R621',
    'C41J':   'C411',
    'C50G':   'C506',
    'R5122':  'R512',
    'BC1788': 'BC178',
    'BC178B': 'BC178',
    'T5502':  'T502',
}


def fix_ocr_confusion(text: str) -> str:
    """
    Corrige les confusions chiffres/lettres par analyse de la structure de la référence.

    Algorithme de scan linéaire :
      1. Normalise le texte (majuscules, suppression des espaces).
      2. Recherche l'index i du premier vrai chiffre ASCII (0-9), qui délimite
         la frontière entre le préfixe alphabétique et la partie numérique.
         Si aucun chiffre n'est trouvé, le texte est retourné sans modification
         car il ne ressemble pas à une référence de composant.
      3. Détecte un suffixe optionnel : si le dernier caractère de la partie
         numérique est une lettre et qu'il y a au moins un autre caractère,
         ce dernier est isolé comme suffixe (ex: "A" dans "IC402A").
      4. Applique PREFIX_FIXES caractère par caractère sur le préfixe pour
         corriger les chiffres mal lus en lettres.
      5. Applique DIGIT_FIXES caractère par caractère sur la partie numérique
         pour corriger les lettres mal lues en chiffres.
      6. Reconstruit : prefix_corrigé + digits_corrigés + suffixe.

    Exemples de transformation :
      "C41J"   → "C411"   (J → 1 dans la partie numérique)
      "R51G"   → "R516"   (G → 6 dans la partie numérique)
      "0C402"  → "OC402"  (0 → O dans le préfixe)
      "R516A"  → "R516A"  (A = suffixe, conservé)
      "8C178"  → "BC178"  (8 → B dans le préfixe)

    Paramètres
    ----------
    text : chaîne brute issue d'EasyOCR

    Retourne
    --------
    Chaîne corrigée (str)
    """
    if not text:
        return text

    text = text.strip().upper().replace(' ', '')

    i = 0
    while i < len(text) and not text[i].isdigit():
        i += 1

    if i == len(text):
        return text

    prefix_raw = text[:i]
    rest       = text[i:]

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


def apply_post_corrections(text: str) -> str:
    """
    Applique le dictionnaire POST_CORRECTIONS sur une référence déjà traitée
    par fix_ocr_confusion. Couvre les cas non corrigibles par règle générique
    (ex: confusions multi-caractères, artefacts scan).

    La clé de recherche est normalisée (majuscules, sans espaces) pour garantir
    la correspondance indépendamment de la casse résiduelle.

    Paramètres
    ----------
    text : référence partiellement corrigée

    Retourne
    --------
    Référence corrigée ou texte original si non trouvé dans le dictionnaire.
    """
    key = text.strip().upper().replace(' ', '')
    return POST_CORRECTIONS.get(key, text)


# ==============================================================================
# SECTION 4 — VALIDATION PAR EXPRESSION RÉGULIÈRE
# ==============================================================================
#
# Expression régulière qui couvre les formats de références rencontrés sur les
# schémas et cartes PCB de l'électronique grand public et professionnelle.
#
# Décomposition des alternatives :
#
#   [A-Z]{1,2}\s?\d{3,5}[A-Z]?
#     Résistances, condensateurs, bobines : R516, C401, L12, R 516 (espace OCR)
#     Préfixe 1-2 lettres + 3-5 chiffres + suffixe lettre optionnel.
#
#   [A-Z]{1,3}\d{2,5}[A-Z]?
#     Circuits intégrés, diodes numérotées : IC402A, U201, D14
#     Préfixe 1-3 lettres + 2-5 chiffres + suffixe optionnel.
#
#   [A-Z]{2,4}\d{1,4}[A-Z]?
#     Composants à préfixe long : BC178 (transistor), FR409 (diode redresseur),
#     TP12 (test point), DZ15 (diode Zener).
#
#   St\s?\d{3}
#     Cas particulier : connecteurs de type ST-BUS notés "St" + 3 chiffres.
#
#   [A-Z]{1,3}\d+[-_]\d+
#     Références sous-numérotées : R12-3, C5_1 (composants en réseau).
#
# Le flag re.IGNORECASE permet de traiter indifféremment "IC402" et "ic402",
# ce qui est utile car fix_ocr_confusion normalise en majuscules mais la
# validation est appliquée aussi sur des textes non encore normalisés.

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

# Préfixes reconnus comme valides pour un composant électronique.
# Utilisé dans certains filtres rapides avant validation regex complète.
VALID_COMPONENT_PREFIXES = [
    'R', 'C', 'U', 'D', 'Q', 'L', 'J', 'SW', 'F',
    'TP', 'RV', 'Y', 'DZ', 'IC',
]


def is_valid_ref(text: str) -> bool:
    """
    Retourne True si 'text' correspond au format d'une référence de composant
    électronique, selon COMPONENT_PATTERN.

    Paramètres
    ----------
    text : chaîne corrigée à valider

    Retourne
    --------
    bool
    """
    return bool(COMPONENT_PATTERN.match(text.strip()))


# ==============================================================================
# SECTION 5 — PRÉTRAITEMENT DES CROPS POUR L'OCR
# ==============================================================================
#
# Les zones de texte découpées par YOLO sont souvent de petite taille (20-80 px
# de hauteur) et nécessitent un prétraitement avant lecture OCR pour améliorer
# la précision du réseau CRNN d'EasyOCR.
#
# Deux méthodes sont disponibles et utilisées selon une stratégie de cascade :
#
# 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
#    Méthode principale, adaptée aux fonds clairs standard (schémas imprimés,
#    scans haute résolution). Opère en niveaux de gris avec une grille de 8×8
#    tuiles et un clipLimit de 2.0. Ce paramètre limite l'amplification des
#    zones très homogènes pour éviter d'amplifier le bruit dans les régions sans
#    texte. Le filtre Laplacien de netteté (kernel 3×3 avec valeur centrale 5)
#    est appliqué avant CLAHE pour accentuer les contours des caractères.
#
# 2. Otsu (binarisation par seuillage automatique)
#    Méthode de secours pour les fonds sombres ou les images à fort contraste
#    inversé. La méthode d'Otsu détermine automatiquement le seuil de binarisation
#    optimal en minimisant la variance intra-classe entre les pixels de fond et
#    de texte (critère de Nobuyuki Otsu, 1979).
#
# Dans les deux cas :
#   - L'image est agrandie d'un facteur 'scale' (défaut 4-5×) par interpolation
#     bicubique avant prétraitement. Cette étape est critique : EasyOCR/CRNN
#     est entraîné sur des images où les caractères ont au moins 20-30px de
#     hauteur. Sur des crops de 15px, l'upscale garantit cette condition.
#   - Une bordure blanche de 10px est ajoutée sur les 4 côtés pour éviter les
#     artefacts de détection aux bords (CRAFT génère des fausses régions sur les
#     pixels de bord des images sans marge).


def preprocess_crop_clahe(crop_bgr, scale=4):
    """
    Prétraitement principal : upscale + filtre de netteté Laplacien + CLAHE.
    Retourne une image BGR prête pour EasyOCR.

    Paramètres
    ----------
    crop_bgr : image BGR (numpy array H×W×3) découpée autour d'une zone de texte
    scale    : facteur d'agrandissement (défaut 4×, ajusté par auto_params selon
               la résolution de l'image source)

    Pipeline interne :
      1. cv2.resize (INTER_CUBIC) : interpolation bicubique, meilleure qualité
         que INTER_LINEAR pour l'upscale de petits textes.
      2. cv2.filter2D avec kernel Laplacien : accentue les transitions de luminosité
         (contours des caractères) sans modifier la teinte.
      3. cv2.cvtColor BGR→GRAY : conversion en niveaux de gris.
      4. cv2.createCLAHE + apply : égalisation adaptative limitée.
      5. cv2.cvtColor GRAY→BGR : reconversion pour compatibilité EasyOCR.
      6. cv2.copyMakeBorder : ajout de la bordure blanche périphérique.
    """
    h, w  = crop_bgr.shape[:2]
    big   = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp  = cv2.filter2D(big, -1, kernel)

    gray     = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return cv2.copyMakeBorder(bgr, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def preprocess_crop_otsu(crop_bgr, scale=4):
    """
    Prétraitement de secours : upscale + binarisation Otsu.
    Plus robuste sur les fonds sombres et les images à contraste inversé.

    La binarisation Otsu (cv2.THRESH_BINARY + cv2.THRESH_OTSU) cherche le seuil T
    qui minimise la variance intra-classe entre les deux groupes de pixels
    (fond / texte). Elle est entièrement automatique et n'a aucun paramètre à
    régler, ce qui en fait un fallback idéal.

    Paramètres
    ----------
    crop_bgr : image BGR source
    scale    : facteur d'agrandissement avant binarisation
    """
    h, w = crop_bgr.shape[:2]
    big  = cv2.resize(crop_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bgr  = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return cv2.copyMakeBorder(bgr, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))


# ==============================================================================
# SECTION 6 — LECTURE OCR SUR CROP
# ==============================================================================
#
# L'OCR est structuré en deux niveaux :
#
# Niveau bas (_run_easyocr / _run_easyocr_on_image) :
#   Appel direct à READER.readtext(). Paramètres clés :
#   - detail=1 : retourne (bbox, text, conf) au lieu de juste text.
#   - paragraph=False : chaque région de texte est retournée séparément
#     (pas de fusion paragraphe), ce qui est correct pour des crops d'une
#     seule référence.
#   - allowlist=ALLOWLIST : restreint le vocabulaire reconnaissable.
#   - width_ths=0.7 : fusionne les détections proches horizontalement dont
#     le ratio de chevauchement est > 0.7. Utile pour les références dont
#     les caractères sont légèrement espacés sur le schéma.
#
# Niveau haut (run_ocr / run_ocr_on_crop) :
#   Stratégie de cascade à 3 tentatives, arrêt dès qu'une référence valide
#   est trouvée (is_valid_ref True) :
#     1. CLAHE normal : cas standard.
#     2. CLAHE + rotation 90° sens horaire : pour les labels verticaux fréquents
#        sur les schémas électroniques (repères de résistances placées à la verticale).
#        Déclenchée si le crop est plus haut que large (h > w × 1.3) OU si
#        aucun texte n'a été trouvé à l'étape 1.
#     3. Otsu : fallback pour les cas résiduels (fond sombre, contraste inversé).
#   À chaque tentative, le meilleur résultat est sélectionné parmi tous les
#   textes détectés dans le crop, en priorité les références valides, sinon
#   le texte avec la confiance EasyOCR la plus élevée.


def _run_easyocr(img_bgr):
    """
    Appelle READER.readtext sur img_bgr et retourne une liste de (text, conf)
    triée par confiance décroissante. Les résultats sous OCR_CONF_MIN sont exclus.
    Utilisé par ocr_ref_extractor (pipeline standalone PCB).
    """
    try:
        results = READER.readtext(
            img_bgr,
            detail=1,
            paragraph=False,
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


def _run_easyocr_on_image(img_bgr):
    """
    Version identique à _run_easyocr, nommée séparément pour la lisibilité du code
    du viewer (contexte multi-sources). Appelle READER.readtext et filtre par seuil.
    """
    try:
        results = READER.readtext(
            img_bgr,
            detail=1,
            paragraph=False,
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


def run_ocr(crop_bgr, scale=5):
    """
    Pipeline OCR à 3 tentatives pour le mode extraction standalone (ocr_ref_extractor).
    Utilisé quand l'image source provient directement d'une carte PCB photographiée
    ou scannée (non issue d'un PDF). Le scale par défaut est 5 (vs 4 pour viewer)
    car les crops PCB sont généralement plus petits.

    Retourne
    --------
    (text_corrigé : str, is_valid : bool, conf : float)
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
    h, w      = pre_clahe.shape[:2]
    text, valid, conf = best_from(pre_clahe)
    if valid:
        return text, valid, conf

    if h > w * 1.3 or not text:
        t2, v2, c2 = best_from(cv2.rotate(pre_clahe, cv2.ROTATE_90_CLOCKWISE))
        if v2 or (t2 and c2 > conf):
            text, valid, conf = t2, v2, c2
    if valid:
        return text, valid, conf

    pre_otsu    = preprocess_crop_otsu(crop_bgr, scale=scale)
    t3, v3, c3  = best_from(pre_otsu)
    if v3 or (t3 and c3 > conf):
        return t3, v3, c3

    return text, valid, conf


def run_ocr_on_crop(crop_bgr, scale=4):
    """
    Pipeline OCR à 3 tentatives pour le viewer (schéma + carte PCB dans l'interface).
    Identique à run_ocr mais avec scale par défaut de 4, adapté aux crops issus
    d'images haute résolution générées par conversion PDF à 300 DPI.

    Retourne
    --------
    (text_corrigé : str, is_valid : bool, conf : float)
    """
    def best_match_from(img):
        hits = _run_easyocr_on_image(img)
        if not hits:
            return "", False, 0.0
        for (t, c) in hits:
            fixed = apply_post_corrections(fix_ocr_confusion(t))
            if is_valid_ref(fixed):
                return fixed, True, c
        fixed = apply_post_corrections(fix_ocr_confusion(hits[0][0]))
        return fixed, False, hits[0][1]

    pre_clahe = preprocess_crop_clahe(crop_bgr, scale=scale)
    h, w      = pre_clahe.shape[:2]
    text, valid, conf = best_match_from(pre_clahe)
    if valid:
        return text, valid, conf

    if h > w * 1.3 or not text:
        t2, v2, c2 = best_match_from(cv2.rotate(pre_clahe, cv2.ROTATE_90_CLOCKWISE))
        if v2 or (t2 and c2 > conf):
            text, valid, conf = t2, v2, c2
    if valid:
        return text, valid, conf

    pre_otsu   = preprocess_crop_otsu(crop_bgr, scale=scale)
    t3, v3, c3 = best_match_from(pre_otsu)
    if v3 or (t3 and c3 > conf):
        return t3, v3, c3

    return text, valid, conf


# ==============================================================================
# SECTION 7 — FUSION DES BOÎTES YOLO FRAGMENTÉES
# ==============================================================================
#
# Problème : YOLO peut détecter une référence comme deux boîtes séparées.
# Exemple typique : "R" → boîte 1, "516" → boîte 2, alors que le label souhaité
# est "R516" en une seule zone. Ce phénomène survient quand le préfixe lettre
# est visuellement plus éloigné des chiffres (espacement inter-caractère élevé
# sur certains silkscreens PCB, ou caractère partiellement masqué).
#
# Solution : algorithme de fusion glouton multi-passes (while changed).
# À chaque passe, toutes les paires (i, j) sont examinées :
#   - j doit être à droite de i (x1b ≥ x2a) : évite les fusions sur des
#     boîtes qui se chevauchent (déjà gérées par NMS de YOLO).
#   - Gap horizontal ≤ gap_ratio × hauteur moyenne : garantit que seules les
#     boîtes vraiment proches sont fusionnées. Un gap_ratio de 0.8 signifie
#     que le blanc entre les deux boîtes ne dépasse pas 80% de leur hauteur
#     moyenne, ce qui correspond à un espace inter-caractère plausible pour
#     une écriture électronique standard.
#   - Chevauchement vertical ≥ overlap_y_ratio × hauteur minimale : garantit
#     l'alignement horizontal des deux boîtes. Un overlap_y de 40% évite de
#     fusionner des labels sur deux lignes différentes du schéma.
# Pour chaque boîte i, on sélectionne le meilleur partenaire j (gap minimal).
# La fusion produit la boîte englobante min/max et la confiance moyenne.
# Les passes se répètent jusqu'à stabilité (plus aucune fusion possible).
# Complexité : O(n²) par passe, acceptable car n (détections YOLO) est
# typiquement < 500 par image.


def merge_nearby_boxes(boxes, gap_ratio=0.8, overlap_y_ratio=0.4):
    """
    Fusionne itérativement les paires de boîtes YOLO fragmentées.

    Paramètres
    ----------
    boxes           : liste de tuples (x1, y1, x2, y2, conf) en pixels entiers
    gap_ratio       : ratio gap/hauteur_moyenne max pour la fusion (défaut 0.8)
    overlap_y_ratio : ratio chevauchement_Y/hauteur_min min pour l'alignement (défaut 0.4)

    Retourne
    --------
    Liste de tuples (x1, y1, x2, y2, conf) fusionnés. La conf est la moyenne
    pondérée des deux boîtes d'origine.
    """
    if not boxes:
        return boxes

    boxes   = sorted(boxes, key=lambda b: b[0])
    merged  = list(boxes)
    changed = True

    while changed:
        changed    = False
        new_merged = []
        used       = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue
            x1a, y1a, x2a, y2a, ca = merged[i]
            best_j, best_gap        = -1, float('inf')

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                x1b, y1b, x2b, y2b, cb = merged[j]

                if x1b < x2a:
                    continue

                h_avg = ((y2a - y1a) + (y2b - y1b)) / 2
                gap   = x1b - x2a
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
                new_merged.append((
                    min(x1a, x1b), min(y1a, y1b),
                    max(x2a, x2b), max(y2a, y2b),
                    (ca + cb) / 2,
                ))
                used[i] = used[best_j] = True
                changed = True
            else:
                new_merged.append(merged[i])
                used[i] = True

        merged = new_merged

    return merged


# ==============================================================================
# SECTION 8 — DÉDUPLICATION DES RÉFÉRENCES
# ==============================================================================
#
# Phénomène : une même référence physique peut être détectée plusieurs fois sur
# la même image si elle est visible depuis deux angles légèrement différents, si
# le label est partiellement à cheval sur deux zones YOLO, ou si le modèle génère
# des boîtes redondantes non supprimées par NMS (NMS = Non-Maximum Suppression,
# l'étape post-YOLO qui fusionne les détections chevauchantes).
#
# Solution : après OCR de toutes les zones, on construit un dictionnaire
# { ref_normalisée → index_dans_liste } et on ne conserve par référence que la
# détection ayant le score composite le plus élevé (yolo_conf × ocr_conf).
# Le score composite reflète la double fiabilité : confiance géométrique YOLO
# (qualité de la localisation) × confiance CTC EasyOCR (qualité de la lecture).
# Les détections invalides (is_valid_ref False) sont conservées sans déduplication
# car elles sont utiles pour le diagnostic.


def deduplicate_refs(detections):
    """
    Déduplique les références valides en gardant la détection avec le meilleur
    score composite (yolo_conf × ocr_conf). Utilisée dans le mode ocr standalone.

    Paramètres
    ----------
    detections : liste de dicts avec les clés 'text', 'is_valid_ref',
                 'yolo_conf', 'ocr_conf'

    Retourne
    --------
    Liste dédupliquée (les détections invalides sont conservées intégralement).
    """
    seen   = {}
    result = []

    for d in detections:
        if not d["is_valid_ref"]:
            result.append(d)
            continue

        key   = d["text"].replace(' ', '').upper()
        score = d["yolo_conf"] * d["ocr_conf"]

        if key not in seen:
            seen[key] = len(result)
            result.append(d)
        else:
            existing       = result[seen[key]]
            existing_score = existing["yolo_conf"] * existing["ocr_conf"]
            if score > existing_score:
                result[seen[key]] = d

    return result


def _deduplicate_refs_viewer(found_refs):
    """
    Variante de déduplication pour le viewer (format dict différent : clé 'ref'
    au lieu de 'text', clé 'is_valid' au lieu de 'is_valid_ref').
    Même algorithme, garde le meilleur score composite par référence.

    Paramètres
    ----------
    found_refs : liste de dicts avec 'ref', 'yolo_conf', 'ocr_conf'

    Retourne
    --------
    Liste dédupliquée.
    """
    seen   = {}
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


# ==============================================================================
# SECTION 9 — AUTO-DÉTECTION DES PARAMÈTRES (mode OCR standalone)
# ==============================================================================
#
# Problème : les paramètres optimaux de détection YOLO (imgsz, conf) et de
# prétraitement OCR (scale d'upscale, inversion fond sombre) varient selon
# les caractéristiques de l'image source. Une image 4K à fort contraste
# nécessite des paramètres différents d'une image 640px peu contrastée.
#
# Solution : analyse de l'histogramme de luminosité de l'image complète pour
# déduire automatiquement les paramètres :
#
# imgsz YOLO (taille d'entrée du réseau) :
#   YOLO redimensionne l'image à imgsz avant inférence. Une image grande avec
#   un petit imgsz perd de l'information ; une petite image avec un grand imgsz
#   est du sur-traitement. On adapte imgsz à la résolution source.
#
# scale OCR (facteur d'upscale des crops) :
#   Les crops YOLO sont proportionnels à l'image source. Sur une image basse
#   résolution, les crops sont très petits et nécessitent un scale élevé pour
#   atteindre la taille minimale exploitable par EasyOCR (≈ 30px de hauteur
#   de caractère). Sur une image haute résolution, les crops sont déjà grands
#   et un scale modéré suffit.
#
# conf YOLO (seuil de détection) :
#   Sur une image peu contrastée (std < 25), abaisser le seuil de confiance
#   permet de détecter les références mal imprimées. Sur une image très
#   contrastée, un seuil plus élevé réduit les faux positifs.
#
# invert (inversion couleurs) :
#   Certains schémas ont un fond sombre (négatif). EasyOCR et YOLO sont
#   entraînés sur des images à fond clair. Si la luminosité moyenne < 80/255,
#   on inverse l'image avec cv2.bitwise_not avant traitement.


def auto_params(img):
    """
    Analyse l'image et déduit les paramètres optimaux pour YOLO et OCR.

    Métriques calculées :
      - mean : luminosité moyenne du canal gris (0-255)
      - std  : écart-type de luminosité (proxy du contraste global)
      - px   : résolution totale en mégapixels (w × h)

    Paramètres retournés dans un dict :
      - imgsz  : taille d'entrée YOLO (640 à 1280)
      - scale  : facteur d'upscale pour le prétraitement des crops OCR (5 à 10)
      - conf   : seuil de confiance YOLO (0.25 à 0.40)
      - invert : bool, True si fond sombre détecté (mean < 80)

    Paramètres
    ----------
    img : image BGR (numpy array) chargée par cv2.imread

    Retourne
    --------
    dict avec les clés 'imgsz', 'scale', 'conf', 'invert'
    """
    h, w  = img.shape[:2]
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean  = float(gray.mean())
    std   = float(gray.std())
    px    = w * h

    if px > 3_000_000:
        imgsz, scale = 1280, 5
    elif px > 1_000_000:
        imgsz, scale = 1024, 6
    elif px > 300_000:
        imgsz, scale = 640, 8
    else:
        imgsz, scale = 640, 10

    if std < 25:
        conf = 0.25
    elif std < 45:
        conf = 0.30
    else:
        conf = 0.40

    invert = mean < 80

    params = {"imgsz": imgsz, "scale": scale, "conf": conf, "invert": invert}

    inv_tag = f'  {_c("INVERSION fond sombre", C.YELLOW)}' if invert else ''
    print(
        f'  {_c("◈", C.CYAN)}  {_c("AUTO", C.BOLD)}  '
        f'{_c(f"{w}x{h}px", C.WHITE)} | '
        f'contraste={_c(f"{std:.0f}", C.CYAN)} | '
        f'luminosite={_c(f"{mean:.0f}", C.CYAN)} | '
        f'imgsz={_c(imgsz, C.GREEN)} | '
        f'scale={_c(scale, C.GREEN)} | '
        f'conf={_c(conf, C.GREEN)}'
        f'{inv_tag}'
    )
    return params


# ==============================================================================
# SECTION 10 — EXTRACTION OCR STANDALONE (mode 'ocr')
# ==============================================================================
#
# Pipeline complet d'extraction des références sur une image PCB unique :
#   1. Chargement de l'image via cv2.imread.
#   2. Auto-détection des paramètres (auto_params).
#   3. Inversion optionnelle si fond sombre.
#   4. Inférence YOLO pour localiser les zones de texte (références).
#   5. Fusion des boîtes fragmentées (merge_nearby_boxes).
#   6. Pour chaque boîte : découpage avec marge (PAD=4px), appel run_ocr.
#   7. Construction d'un dict de détection avec métadonnées complètes.
#   8. Déduplication finale.
#
# Le score composite (yolo_conf × ocr_conf) reflète la double fiabilité de
# chaque détection. Il est utilisé pour trier les résultats dans le tableau
# terminal et pour sélectionner la meilleure détection lors de la déduplication.


def extract(img_path, model_path, conf_thresh=None):
    """
    Extrait les références de composants d'une image PCB via YOLO + EasyOCR.

    Paramètres
    ----------
    img_path    : chemin vers l'image source (str ou Path)
    model_path  : chemin vers le modèle YOLO des références (.pt)
    conf_thresh : seuil de confiance YOLO manuel (float, None = auto)

    Retourne
    --------
    (img : numpy array BGR, detections : liste de dicts)

    Chaque dict de détection contient :
      id          : numéro séquentiel (1-based)
      text        : référence corrigée
      is_valid_ref: bool, True si valide selon COMPONENT_PATTERN
      yolo_conf   : confiance YOLO (0.0-1.0)
      ocr_conf    : confiance EasyOCR (0.0-1.0)
      score       : yolo_conf × ocr_conf
      bbox        : dict {x1, y1, x2, y2} en pixels
      center      : dict {x, y} coordonnées du centre de la bbox
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {img_path}")
    step('◉', 'Image', f'{img.shape[1]}x{img.shape[0]} px')

    params = auto_params(img)
    if conf_thresh is not None:
        params["conf"] = conf_thresh

    if params["invert"]:
        img = cv2.bitwise_not(img)

    t_yolo      = time.time()
    model       = YOLO(str(model_path))
    yolo_result = model(img, conf=params["conf"], imgsz=params["imgsz"], verbose=False)[0]
    raw_boxes   = yolo_result.boxes
    boxes_raw   = [
        (int(b.xyxy[0][0]), int(b.xyxy[0][1]),
         int(b.xyxy[0][2]), int(b.xyxy[0][3]),
         float(b.conf[0]))
        for b in raw_boxes
    ]
    boxes   = merge_nearby_boxes(boxes_raw)
    fusions = len(boxes_raw) - len(boxes)

    step('◉', 'YOLO',
         f'{len(raw_boxes)} zones  →  {len(boxes)} apres merge '
         f'({_c(f"+{fusions} fusions", C.CYAN)})  '
         f'{_c(f"{time.time()-t_yolo:.1f}s", C.GRAY)}')
    print()

    detections = []
    PAD        = 4

    for i, (x1, y1, x2, y2, conf_yolo) in enumerate(boxes):
        crop = img[max(0, y1 - PAD): min(img.shape[0], y2 + PAD),
                   max(0, x1 - PAD): min(img.shape[1], x2 + PAD)]
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
            "center":       {"x": (x1 + x2) // 2, "y": (y1 + y2) // 2},
        })
        progress_bar(i + 1, len(boxes))

    before     = sum(1 for d in detections if d["is_valid_ref"])
    detections = deduplicate_refs(detections)
    after      = sum(1 for d in detections if d["is_valid_ref"])
    if before != after:
        ok(f'Deduplication : {before} → {_c(after, C.GREEN, C.BOLD)} refs uniques')

    return img, detections


# ==============================================================================
# SECTION 11 — VISUALISATION ET EXPORTS (mode OCR standalone)
# ==============================================================================
#
# draw_results : génère une image annotée avec un rectangle coloré autour de
# chaque zone détectée et le texte OCR en superposition. Vert (#00C800) pour
# les références valides, orange (#008CFF) pour les détections invalides.
# La taille de police EasyOCR est fixée à 0.4 (échelle relative cv2) pour
# rester lisible même sur des images haute résolution (DPI 300).
#
# save_json : sérialisation complète des détections en JSON UTF-8 indenté.
# Format structuré adapté à l'import dans des outils de gestion de BOM.
#
# save_csv : export tabulaire avec un en-tête fixe. Les coordonnées de bbox
# et du centre sont aplaties (x1, y1, x2, y2, cx, cy) pour faciliter
# l'import dans Excel, LibreOffice Calc ou tout outil d'analyse tabulaire.


def draw_results(img, detections):
    """
    Génère une copie annotée de l'image avec les zones de détection colorées.

    Paramètres
    ----------
    img        : image BGR source (non modifiée)
    detections : liste de dicts de détection (format extract())

    Retourne
    --------
    Image BGR annotée (numpy array)
    """
    out = img.copy()
    for d in detections:
        b              = d["bbox"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        color          = (0, 200, 0) if d["is_valid_ref"] else (0, 140, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if d["text"]:
            fs          = 0.4
            (tw, th), _ = cv2.getTextSize(d["text"], cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, d["text"], (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1)
    return out


def save_json(detections, path):
    """
    Sérialise la liste de détections en JSON UTF-8 indenté (2 espaces).

    Paramètres
    ----------
    detections : liste de dicts (format extract())
    path       : chemin de sortie (str ou Path)
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)


def save_csv(detections, path):
    """
    Exporte les détections en CSV avec en-tête.
    Les coordonnées bbox (x1, y1, x2, y2) et centre (cx, cy) sont aplaties
    en colonnes séparées pour faciliter l'import tabulaire.

    Paramètres
    ----------
    detections : liste de dicts (format extract())
    path       : chemin de sortie (str ou Path)
    """
    fields = ["id", "text", "is_valid_ref", "yolo_conf", "ocr_conf", "score",
              "x1", "y1", "x2", "y2", "cx", "cy"]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in detections:
            w.writerow({
                "id":          d["id"],
                "text":        d["text"],
                "is_valid_ref": d["is_valid_ref"],
                "yolo_conf":   d["yolo_conf"],
                "ocr_conf":    d["ocr_conf"],
                "score":       d["score"],
                "x1": d["bbox"]["x1"],   "y1": d["bbox"]["y1"],
                "x2": d["bbox"]["x2"],   "y2": d["bbox"]["y2"],
                "cx": d["center"]["x"],  "cy": d["center"]["y"],
            })


# ==============================================================================
# SECTION 12 — EXTRACTION DE ZONES PDF (pdf_zone_extractor)
# ==============================================================================
#
# Ce module opère en amont du pipeline complet et produit des fichiers PNG
# correspondant aux zones d'intérêt (Table, Diagram, Board) détectées dans un PDF.
#
# pdf2image.convert_from_path() utilise Poppler (bibliothèque C++) pour rastériser
# chaque page du PDF en image PIL (Pillow). Le paramètre dpi=300 produit des images
# à 300 points par pouce, ce qui garantit une résolution suffisante pour l'OCR
# (règle empirique : au moins 200 DPI pour un OCR fiable, 300 recommandé).
# À 300 DPI, une page A4 (210×297 mm) génère une image de 2480×3508 pixels.
#
# Le modèle YOLO de zones est distinct du modèle de références :
#   - Modèle de zones (pdf_zone_extractor / pdf_to_viewer) : détecte les grandes
#     régions structurelles du document (Table BOM, Schéma électronique, Carte PCB).
#     Utilisé avec YOLO_CONF_THRESHOLD = 0.212 (seuil bas pour ne pas rater une zone
#     unique par document).
#   - Modèle de références (ocr_ref_extractor / viewer) : détecte les petites zones
#     de texte des références individuelles sur le schéma ou la carte.
#
# CROP_MARGIN_PX : marge de sécurité de 10px autour de chaque boîte YOLO pour
# éviter que l'interpolation bicubique du redimensionnement ne perde des pixels
# de bord. Les coordonnées sont clampées à 0 et aux dimensions de l'image.

PDF_DPI              = 300
YOLO_CONF_ZONES      = 0.212
CROP_MARGIN_PX       = 10

# Rotations à appliquer selon la classe de zone détectée.
# Les schémas électroniques et cartes PCB sont fréquemment placés en orientation
# paysage dans les PDF techniques pour maximiser l'espace. On les pivote vers
# l'orientation portrait pour que l'interface de visualisation les affiche droits.
# cv2.ROTATE_90_COUNTERCLOCKWISE = rotation de 90° dans le sens antihoraire
# cv2.ROTATE_90_CLOCKWISE        = rotation de 90° dans le sens horaire
ZONE_ROTATIONS = {
    "Table":   None,
    "Diagram": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "Board":   cv2.ROTATE_90_CLOCKWISE,
}


def extract_zones_from_pdf(pdf_path: str, output_dir: str):
    """
    Détecte et découpe les zones structurelles (Table, Diagram, Board) d'un PDF.
    Sauvegarde chaque zone en PNG dans output_dir avec le nom :
    <nom_pdf>_<index_page>_<classe>.png

    Cette fonction opère en mode standalone : elle ne sélectionne pas la "meilleure"
    zone par classe mais sauvegarde toutes les détections de toutes les pages.

    Paramètres
    ----------
    pdf_path   : chemin vers le fichier PDF source
    output_dir : répertoire de sortie pour les PNG (créé si absent)
    """
    model = YOLO(YOLO_MODEL_PATH_ZONES if 'YOLO_MODEL_PATH_ZONES' in dir() else "best.pt")
    pages = convert_from_path(pdf_path, dpi=PDF_DPI)

    for page_index, page in enumerate(pages):
        page_array = np.array(page)
        result     = model.predict(source=page, conf=YOLO_CONF_ZONES)

        for box in result[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            y1_m  = max(0, y1 - CROP_MARGIN_PX)
            y2_m  = min(page_array.shape[0], y2 + CROP_MARGIN_PX)
            x1_m  = max(0, x1 - CROP_MARGIN_PX)
            x2_m  = min(page_array.shape[1], x2 + CROP_MARGIN_PX)
            crop  = page_array[y1_m:y2_m, x1_m:x2_m]

            if crop is not None and crop.size > 0:
                class_name = result[0].names[int(box.cls[0])]
                filename   = f"{Path(pdf_path).stem}_{page_index}_{class_name}.png"
                cv2.imwrite(
                    os.path.join(output_dir, filename),
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                )

    print(f"Traitement terminé. Images sauvegardées dans : {output_dir}")


# ==============================================================================
# SECTION 13 — PIPELINE PDF → VIEWER (pdf_to_viewer)
# ==============================================================================
#
# Ce pipeline orchestre les étapes intermédiaires entre le PDF brut et l'interface
# graphique. Contrairement à extract_zones_from_pdf qui sauvegarde tout, ce pipeline
# sélectionne pour chaque classe la zone de plus grande surface (proxy de qualité :
# une grande zone correctement détectée est préférable à un fragment partiel).
#
# Stratégie de sélection par surface maximale :
#   On parcourt toutes les pages et toutes les détections. Pour chaque classe,
#   on ne conserve que le crop dont la surface (h × w en pixels) est maximale.
#   Cela filtre efficacement les petites détections parasites qui peuvent survenir
#   sur des pages de garde ou des tableaux récapitulatifs partiels.
#
# Rotation avant sauvegarde (ZONE_ROTATIONS) :
#   La rotation est appliquée avec cv2.rotate sur l'image BGR finale, après
#   conversion depuis le format RGB (PIL). L'image résultante est sauvegardée
#   dans un fichier temporaire (temp_table.png, temp_diagram.png, temp_board.png)
#   dans le répertoire courant. Ces fichiers sont consommés par show_interface
#   puis peuvent être supprimés.
#
# Table et Diagram sont obligatoires pour lancer l'interface (elle nécessite
# au minimum une BOM et un schéma). Board est optionnel (None passé à show_interface
# si non détecté).


def run_pipeline(pdf_path: str, model_path: str):
    """
    Pipeline complet : PDF → détection de zones YOLO → fichiers temporaires → viewer.

    Paramètres
    ----------
    pdf_path   : chemin vers le fichier PDF à analyser
    model_path : chemin vers le modèle YOLO de détection de zones structurelles (.pt)
    """
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Erreur chargement modèle YOLO zones : {e}")
        return

    print("--- Conversion du PDF en images (DPI 300) ---")
    pages = convert_from_path(pdf_path, dpi=PDF_DPI)

    best_zones = {cls: (None, 0) for cls in ZONE_ROTATIONS}

    print("--- Détection des zones via YOLO ---")
    for page_index, page in enumerate(pages):
        results    = model.predict(source=page, conf=YOLO_CONF_ZONES, verbose=False)
        page_array = np.array(page)

        for box in results[0].boxes:
            class_name = results[0].names[int(box.cls[0])]
            class_key  = class_name.capitalize()
            if class_key not in best_zones:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            y1_m = max(0, y1 - CROP_MARGIN_PX)
            y2_m = min(page_array.shape[0], y2 + CROP_MARGIN_PX)
            x1_m = max(0, x1 - CROP_MARGIN_PX)
            x2_m = min(page_array.shape[1], x2 + CROP_MARGIN_PX)
            crop = page_array[y1_m:y2_m, x1_m:x2_m]

            if crop is None or crop.size == 0:
                continue

            surface = crop.shape[0] * crop.shape[1]
            if surface > best_zones[class_key][1]:
                best_zones[class_key] = (crop, surface)

    temp_paths = {}

    for class_key, (img_data, _) in best_zones.items():
        if img_data is None:
            continue

        img_bgr  = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        rotation = ZONE_ROTATIONS[class_key]

        if rotation is not None:
            print(f"Rotation appliquée sur : {class_key}")
            img_bgr = cv2.rotate(img_bgr, rotation)

        temp_path = f"temp_{class_key.lower()}.png"
        cv2.imwrite(temp_path, img_bgr)
        temp_paths[class_key] = temp_path

    if "Table" in temp_paths and "Diagram" in temp_paths:
        print("--- Lancement de l'interface de visualisation ---")
        show_interface(
            table_path  = temp_paths["Table"],
            schema_path = temp_paths["Diagram"],
            board_path  = temp_paths.get("Board"),
        )
    else:
        found = [k for k, (img, _) in best_zones.items() if img is not None]
        print(f"\nERREUR : 'Table' ou 'Diagram' non détecté dans le PDF.")
        print(f"Zones trouvées : {found if found else 'aucune'}")


# ==============================================================================
# SECTION 14 — DÉTECTION DES RÉFÉRENCES DANS UNE TABLE BOM (viewer)
# ==============================================================================
#
# Une BOM (Bill Of Materials) en format image présente les références de composants
# dans une colonne de gauche. La stratégie ici est géométrique plutôt que YOLO :
#
#   1. Isolation des 22% gauches de l'image. Ce ratio empirique correspond à la
#      largeur typique de la colonne "Référence" dans les BOM électroniques japonaises
#      et européennes de l'époque (années 70-80, format papier). Il peut nécessiter
#      un ajustement pour d'autres formats de documents.
#
#   2. Upscale ×2 par interpolation bicubique avant OCR. La colonne isolée est
#      souvent étroite (quelques centaines de pixels) ; l'upscale garantit que
#      les caractères atteignent la taille minimale exploitable par EasyOCR.
#
#   3. READER.readtext sur la colonne entière (pas de découpage YOLO par référence).
#      EasyOCR détecte automatiquement les lignes de texte via CRAFT. Les résultats
#      sont triés par ordonnée Y croissante (haut de page → bas) pour reconstituer
#      l'ordre naturel de la BOM.
#
#   4. Reconstruction des références incomplètes : l'OCR lit parfois uniquement
#      les chiffres d'une référence (ex: "516" au lieu de "R516") car le préfixe
#      "R" est trop petit, effacé ou mal imprimé. On mémorise le dernier préfixe
#      alphabétique rencontré (last_prefix) pour reconstituer la référence manquante.
#      Le préfixe initial par défaut est "R" (résistances = composants les plus
#      fréquents dans une BOM type).
#
#   5. Recalcul des coordonnées : les boîtes EasyOCR sont exprimées dans le repère
#      de la colonne upscalée. On les divise par le facteur d'upscale pour les
#      remettre dans le repère de l'image d'origine (nécessaire pour l'affichage
#      du rectangle de focus dans l'interface).


def detect_refs_table(img):
    """
    Extrait les références de composants depuis la colonne gauche d'une image de BOM.

    Paramètres
    ----------
    img : image BGR de la table (BOM), chargée par cv2.imread

    Retourne
    --------
    Liste de dicts {"ref" : str, "bbox" : tuple(x1,y1,x2,y2), "cx" : float, "cy" : float}
    Les coordonnées sont en pixels dans le repère de l'image originale (non upscalée).
    """
    if img is None:
        return []

    h, w        = img.shape[:2]
    ref_column  = img[:, :int(w * 0.22)]
    scale       = 2
    ref_column_big = cv2.resize(ref_column, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    results = READER.readtext(ref_column_big, detail=1, paragraph=False, allowlist=ALLOWLIST)
    results.sort(key=lambda x: x[0][0][1])

    found_refs  = []
    last_prefix = "R"

    for (bbox_pts, text, conf) in results:
        if conf < OCR_CONF_MIN:
            continue

        raw_text     = text.upper().replace(' ', '')
        prefix_match = re.match(r'^([A-Z]+)', raw_text)

        if prefix_match:
            current_prefix = prefix_match.group(1)
            last_prefix    = current_prefix
            clean_text     = raw_text
        else:
            if re.search(r'\d+', raw_text):
                clean_text = f"{last_prefix}{raw_text}"
            else:
                continue

        xs = [p[0] / scale for p in bbox_pts]
        ys = [p[1] / scale for p in bbox_pts]

        found_refs.append({
            "ref":  clean_text,
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
            "cx":   np.mean(xs),
            "cy":   np.mean(ys),
        })

    return found_refs


# ==============================================================================
# SECTION 15 — DÉTECTION DES RÉFÉRENCES SUR SCHÉMA / CARTE PCB (viewer)
# ==============================================================================
#
# Ce pipeline est similaire au pipeline OCR standalone (section 10) mais adapté
# au contexte du viewer : il utilise le modèle de références chargé au niveau
# du viewer (yolo_refs_model), opère avec conf=0.20 (seuil légèrement plus bas
# que le mode standalone pour maximiser le rappel dans l'interface interactive)
# et retourne un format de dict différent ("ref" au lieu de "text").
#
# Le modèle YOLO_REFS_MODEL_PATH = "best_composants.pt" est distinct du modèle
# de zones ("best.pt"). Il est entraîné spécifiquement pour détecter les petits
# labels de références sur les schémas et cartes PCB.
#
# La déduplication (_deduplicate_refs_viewer) est identique en logique à
# deduplicate_refs mais opère sur le format dict du viewer.

YOLO_REFS_MODEL_PATH = "best_composants.pt"

try:
    yolo_refs_model = YOLO(YOLO_REFS_MODEL_PATH)
    print("Modèle YOLO références chargé.")
except Exception as e:
    print(f"Avertissement : modèle YOLO références non chargé ({e}). "
          f"La détection sur schéma/carte sera indisponible.")
    yolo_refs_model = None


def detect_refs_on_image(img):
    """
    Détecte les références de composants sur un schéma électronique ou une carte PCB.
    Utilise le modèle yolo_refs_model (best_composants.pt) pour localiser les zones,
    puis run_ocr_on_crop pour lire chaque zone.

    Paramètres
    ----------
    img : image BGR (numpy array), chargée par cv2.imread

    Retourne
    --------
    Liste de dicts {
        "ref"       : str   — référence corrigée
        "is_valid"  : bool  — valide selon COMPONENT_PATTERN
        "yolo_conf" : float — confiance YOLO
        "ocr_conf"  : float — confiance EasyOCR
        "cx"        : float — coordonnée X du centre (pixels)
        "cy"        : float — coordonnée Y du centre (pixels)
        "bbox"      : tuple (x1, y1, x2, y2) — en float, pixels
    }
    """
    if img is None or yolo_refs_model is None:
        return []

    results   = yolo_refs_model.predict(source=img, conf=0.20, imgsz=1280, verbose=False)
    PAD       = 5
    raw_boxes = []

    for result in results:
        for box in result.boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            raw_boxes.append((b[0], b[1], b[2], b[3], float(box.conf[0])))

    boxes      = merge_nearby_boxes(raw_boxes)
    found_refs = []

    for (x1, y1, x2, y2, conf_yolo) in boxes:
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

    return _deduplicate_refs_viewer(found_refs)


# ==============================================================================
# SECTION 16 — UTILITAIRE DE TRI NATUREL
# ==============================================================================
#
# Le tri alphabétique standard ("lexicographique") produit un ordre incorrect pour
# les chaînes contenant des nombres : ["R2", "R10", "R1"] → ["R1", "R10", "R2"]
# car "1" < "2" dans la table ASCII mais "10" > "2" en valeur numérique.
#
# Le tri naturel (Natural Sort, aussi appelé "human sort") résout ce problème en
# scindant chaque chaîne en segments alternativement textuels et numériques
# (via re.split(r'(\d+)', s)), puis en comparant chaque segment selon son type :
# les segments numériques sont convertis en int (comparaison numérique), les
# segments textuels sont comparés en minuscules (comparaison lexicographique
# insensible à la casse).
#
# Exemple :
#   natural_sort_key("R10") → ["r", 10, ""]
#   natural_sort_key("R2")  → ["r", 2, ""]
#   Comparaison : "r"=="r" → 2 < 10 → R2 < R10 ✓


def natural_sort_key(s):
    """
    Génère une clé de tri naturel pour une chaîne contenant des nombres.
    Permet de trier ["R1", "R2", "R10"] correctement (au lieu de ["R1", "R10", "R2"]).

    Paramètres
    ----------
    s : chaîne à trier

    Retourne
    --------
    Liste alternant str (en minuscules) et int, utilisable comme clé de sorted().
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


# ==============================================================================
# SECTION 17 — INTERFACE GRAPHIQUE INTERACTIVE (viewer)
# ==============================================================================
#
# L'interface est construite avec matplotlib en mode interactif (backend TkAgg).
# Elle affiche simultanément 3 ou 4 panneaux :
#   - Panneau 0 (1/5 de la largeur) : liste des références avec groupes par préfixe
#   - Panneau 1 (2/5) : image de la Table BOM avec annotations
#   - Panneau 2 (2/5) : image du Schéma avec annotations
#   - Panneau 3 (2/5, optionnel) : image de la Carte PCB
#
# GridSpec avec width_ratios permet des panneaux de largeur différente. La liste
# des références occupe moins d'espace (ratio 1.2) que les images (ratio 2).
#
# Événements interactifs (mpl_connect) :
#
#   pick_event : déclenché quand l'utilisateur clique sur un objet matplotlib
#   avec picker=5 (distance de détection de 5 pixels). Chaque texte de référence
#   dans la liste a picker=5. L'objet texte est mappé à son identifiant via
#   text_mapping. On appelle focus_on_component(ident).
#
#   button_press_event : déclenché à chaque clic souris.
#     - Bouton 1 (gauche) sur une vue image : calcule la distance euclidienne entre
#       le point cliqué et tous les centres de détection via np.linalg.norm avec
#       broadcasting (vecteur pts de shape N×2, point cliqué de shape 2 → distances
#       de shape N). Si la distance minimale est < 200px, on fait le focus.
#     - Bouton 3 (droit) : réinitialise les limites de tous les axes à leurs valeurs
#       d'origine (orig_lims sauvegardées avant toute interaction).
#
#   scroll_event : zoom centré sur la position du curseur.
#     Le zoom est implémenté en recalculant les limites x et y de l'axe :
#     on multiplie la plage actuelle par scale (1/1.5 pour zoom in, 1.5 pour zoom out)
#     et on centre autour de event.xdata / event.ydata (coordonnées dans le repère
#     image, pas dans le repère figure).
#
# focus_on_component(ident) :
#   Pour chaque vue, si la référence est dans le dict correspondant :
#     - Recentre l'axe avec set_xlim/set_ylim sur une fenêtre de ±350px autour
#       du centre de détection. L'axe Y est intentionnellement inversé
#       (ylim = [cy+350, cy-350]) car matplotlib affiche les images avec Y croissant
#       vers le bas (convention image) mais set_ylim attendrait Y croissant vers
#       le haut (convention mathématique) → l'inversion corrige ce comportement.
#     - Rend visible le rectangle patches.Rectangle de mise en évidence rouge.
#   Si la référence est absente d'une vue, le rectangle de cette vue est masqué.
#
# Groupes visuels par préfixe dans la liste :
#   Les références sont triées par natural_sort_key, ce qui regroupe naturellement
#   les R ensemble, les C ensemble, etc. Un en-tête "--- R ---" est inséré à chaque
#   changement de préfixe. Une couleur différente (palette TABLEAU_COLORS de
#   matplotlib, 10 couleurs distinctes) est attribuée à chaque groupe préfixe,
#   facilitant la discrimination visuelle dans la liste.


def show_interface(table_path: str, schema_path: str, board_path: str = None):
    """
    Construit et affiche l'interface graphique interactive de visualisation croisée.
    Bloque jusqu'à fermeture de la fenêtre (plt.show() est bloquant avec TkAgg).

    Paramètres
    ----------
    table_path  : chemin vers l'image PNG de la Table (BOM)
    schema_path : chemin vers l'image PNG du Schéma électronique
    board_path  : chemin vers l'image PNG de la Carte PCB (None si absent)
    """
    img_table  = cv2.imread(table_path)
    img_schema = cv2.imread(schema_path)
    img_board  = cv2.imread(board_path) if board_path else None

    print("Analyse OCR en cours...")
    t0 = time.time()

    dict_table  = {r["ref"]: r for r in detect_refs_table(img_table)}
    dict_schema = {r["ref"]: r for r in detect_refs_on_image(img_schema)}
    dict_board  = {r["ref"]: r for r in detect_refs_on_image(img_board)} if img_board is not None else {}

    print(f"Analyse terminée ({time.time()-t0:.1f}s) : "
          f"Table({len(dict_table)}) | Schéma({len(dict_schema)}) | Carte({len(dict_board)})")

    fig  = plt.figure(figsize=(18, 9))
    cols = 4 if img_board is not None else 3
    gs   = fig.add_gridspec(1, cols, width_ratios=[1.2, 2, 2, 2][:cols])

    ax_list   = fig.add_subplot(gs[0, 0])
    ax_table  = fig.add_subplot(gs[0, 1])
    ax_schema = fig.add_subplot(gs[0, 2])
    ax_board  = fig.add_subplot(gs[0, 3]) if img_board is not None else None

    axes_images = [ax for ax in [ax_table, ax_schema, ax_board] if ax is not None]

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

    ids        = sorted(dict_table.keys(), key=natural_sort_key)
    colors     = list(mcolors.TABLEAU_COLORS.values())
    text_mapping                       = {}
    y_pos, last_prefix, color_idx      = 0.98, None, 0

    for ident in ids:
        m      = re.match(r'([A-Z]+)', ident)
        prefix = m.group(1) if m else "?"

        if prefix != last_prefix:
            if last_prefix:
                y_pos -= 0.02
            ax_list.text(0.05, y_pos, f"--- {prefix} ---",
                         fontsize=10, fontweight='bold', alpha=0.7,
                         transform=ax_list.transAxes)
            y_pos     -= 0.03
            color_idx  = (color_idx + 1) % len(colors)

        txt_obj = ax_list.text(0.15, y_pos, ident,
                               fontsize=9, picker=5, fontweight='bold',
                               color=colors[color_idx],
                               transform=ax_list.transAxes)
        text_mapping[txt_obj] = ident
        last_prefix = prefix
        y_pos -= 0.025

        if y_pos < 0.02:
            break

    focus_rects = {
        ax: patches.Rectangle((0, 0), 0, 0, edgecolor="red",
                               facecolor="none", linewidth=2, visible=False)
        for ax in axes_images
    }
    for ax in axes_images:
        ax.add_patch(focus_rects[ax])

    orig_lims = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in axes_images}

    def focus_on_component(ident):
        """
        Centre chaque vue sur le composant 'ident' et affiche son rectangle rouge.
        Rayon de zoom fixe : ±350px autour du centre de détection.
        Si le composant est absent d'une vue, le rectangle de cette vue est masqué.
        """
        zoom_radius = 350
        found       = False
        view_dicts  = {ax_table: dict_table, ax_schema: dict_schema}
        if ax_board is not None:
            view_dicts[ax_board] = dict_board

        for ax, component_dict in view_dicts.items():
            if ident in component_dict:
                info = component_dict[ident]
                ax.set_xlim(info["cx"] - zoom_radius, info["cx"] + zoom_radius)
                ax.set_ylim(info["cy"] + zoom_radius, info["cy"] - zoom_radius)
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
        Gestionnaire de clic souris.
        Clic gauche sur image → zoom vers le composant le plus proche du pointeur.
        Clic droit → réinitialisation du zoom sur toutes les vues.
        """
        if event.button == 3:
            for ax in axes_images:
                ax.set_xlim(orig_lims[ax][0])
                ax.set_ylim(orig_lims[ax][1])
                focus_rects[ax].set_visible(False)
            fig.suptitle("")
            plt.draw()

        elif event.inaxes in axes_images and event.button == 1:
            view_dicts = {ax_table: dict_table, ax_schema: dict_schema}
            if ax_board is not None:
                view_dicts[ax_board] = dict_board

            curr_dict = view_dicts.get(event.inaxes)
            if curr_dict:
                pts  = np.array([[r["cx"], r["cy"]] for r in curr_dict.values()])
                if len(pts) > 0:
                    dist = np.linalg.norm(pts - [event.xdata, event.ydata], axis=1)
                    idx  = np.argmin(dist)
                    if dist[idx] < 200:
                        focus_on_component(list(curr_dict.keys())[idx])

    fig.canvas.mpl_connect("pick_event",         lambda e: focus_on_component(text_mapping[e.artist]))
    fig.canvas.mpl_connect("button_press_event",  on_mouse_click)
    fig.canvas.mpl_connect("scroll_event",        lambda e: on_scroll(e, axes_images))

    print("Interface prête.")
    plt.tight_layout()
    plt.show()


def on_scroll(event, axes_images):
    """
    Gestionnaire de scroll molette : zoom in/out centré sur la position du curseur.

    Principe de recalcul des limites :
      La plage visible actuelle [xmin, xmax] est multipliée par 'scale'.
      Le nouveau centre est event.xdata (position du curseur en pixels image).
      Les nouvelles limites sont : [cursor - new_range/2, cursor + new_range/2].
      Idem pour Y, avec inversion (cf. note dans SECTION 17 sur l'axe Y inversé).

    Paramètres
    ----------
    event       : événement matplotlib scroll (event.button == 'up' ou 'down')
    axes_images : liste des axes images actifs dans la figure
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


# ==============================================================================
# SECTION 18 — POINT D'ENTRÉE ET DISPATCH DES MODES
# ==============================================================================
#
# Trois modes d'exécution accessibles via le premier argument positionnel :
#
#   extract <pdf> <sortie/>
#     Appelle extract_zones_from_pdf(). Utile pour inspecter les zones détectées
#     avant de lancer le viewer, ou pour générer des datasets d'entraînement.
#
#   view <pdf> <model_zones.pt>
#     Appelle run_pipeline(). Pipeline complet PDF → interface interactive.
#     Nécessite Table ET Diagram dans le PDF.
#
#   ocr --img <image.png> --model <best.pt> [--conf <float>]
#     Appelle extract() sur une image PCB seule. Génère image annotée, JSON et CSV
#     dans un sous-répertoire ./results/ horodaté.
#
# Le mode 'ocr' utilise argparse pour gérer ses arguments nommés (--img, --model,
# --conf), tandis que les modes 'extract' et 'view' utilisent des arguments
# positionnels simples pour rester cohérents avec les scripts originaux.


def main_ocr(args_list):
    """
    Point d'entrée du mode extraction OCR standalone (ocr_ref_extractor).
    Parse les arguments, lance extract(), génère les sorties et affiche le résumé.
    """
    parser = argparse.ArgumentParser(description="PCB Component Reference Extractor")
    parser.add_argument("--img",   required=True,           help="Image PCB source")
    parser.add_argument("--model", required=True,           help="Modèle YOLO (.pt)")
    parser.add_argument("--conf",  type=float, default=None, help="Seuil confiance YOLO (défaut: auto)")
    args = parser.parse_args(args_list)

    img_path   = Path(args.img)
    model_path = Path(args.model)

    if not img_path.exists():
        print(f"Image introuvable : {args.img}")
        return
    if not model_path.exists():
        print(f"Modèle introuvable : {args.model}")
        return

    W = 54
    print(f'\n  {_c("─" * W, C.BLUE)}')
    print(f'  {_c("  PCB Component Reference Extractor", C.BOLD + C.WHITE)}')
    print(f'  {_c("─" * W, C.BLUE)}\n')

    t_total    = time.time()
    img, detections = extract(img_path, model_path, conf_thresh=args.conf)
    elapsed    = time.time() - t_total

    valid  = [d for d in detections if d["is_valid_ref"]]
    others = [d for d in detections if d["text"] and not d["is_valid_ref"]]
    high   = [d for d in valid if d["ocr_conf"] >= 0.7]

    print(f'\n  {_c("─" * W, C.GRAY)}')
    print(f'  {_c("RÉSULTATS", C.BOLD + C.CYAN)}')
    print(f'  {_c("─" * W, C.GRAY)}')
    print(f'  Zones traitees       {_c(len(detections), C.WHITE, C.BOLD):>6}')
    print(f'  Refs valides         {_c(len(valid),      C.GREEN, C.BOLD):>6}')
    print(f'    dont haute conf    {_c(len(high),       C.GREEN):>6}  '
          f'{_c("(ocr >= 0.7 — fiables)", C.GRAY)}')
    print(f'  Non reconnu          {_c(len(others),     C.YELLOW):>6}')
    print(f'  Temps total          {_c(f"{elapsed:.1f}s", C.CYAN):>6}')
    print(f'  {_c("─" * W, C.GRAY)}')

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

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = img_path.stem

    ann_path  = out_dir / f"{stem}_{ts}_annotated.png"
    json_path = out_dir / f"{stem}_{ts}_results.json"
    csv_path  = out_dir / f"{stem}_{ts}_results.csv"

    cv2.imwrite(str(ann_path), draw_results(img, detections))
    save_json(detections, json_path)
    save_csv(detections, csv_path)

    print(f'\n  {_c("─" * W, C.GRAY)}')
    print(f'  {_c("FICHIERS GÉNÉRÉS", C.BOLD + C.CYAN)}')
    print(f'  {_c("─" * W, C.GRAY)}')
    ok(f'Image    {_c(ann_path, C.GRAY)}')
    ok(f'JSON     {_c(json_path, C.GRAY)}')
    ok(f'CSV      {_c(csv_path, C.GRAY)}')
    print(f'\n  {_c("➜", C.CYAN)}  xdg-open {ann_path}')
    print(f'  {_c("─" * W, C.GRAY)}\n')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 pcb_pipeline_complet.py <document.pdf>")
        sys.exit(1)

    run_pipeline(sys.argv[1], "best.pt")
