"""
pdf_zone_extractor.py
=====================
Script standalone : détecte et découpe les zones d'intérêt (Table, Diagram, Board)
dans un PDF en utilisant un modèle YOLO entraîné.

Chaque zone détectée est sauvegardée en tant qu'image PNG dans le répertoire cible.

Usage:
    python3 pdf_zone_extractor.py <document.pdf> <repertoire_sortie/>

Exemple:
    python3 pdf_zone_extractor.py Accuphase-E202amp.pdf ./output/
"""

import os
import sys
from pathlib import Path
from pdf2image import convert_from_path
import numpy as np
import cv2
from ultralytics import YOLO


#  CONFIGURATION

# Chemin vers le modèle YOLO de détection de zones (Table, Diagram, Board)
YOLO_MODEL_PATH = "best.pt"

# Seuil de confiance YOLO : détections en dessous sont ignorées
YOLO_CONF_THRESHOLD = 0.212

# Résolution de conversion PDF → image (plus élevé = meilleure qualité, plus lent)
PDF_DPI = 300

# Marge en pixels ajoutée autour de chaque zone découpée
CROP_MARGIN_PX = 10


#  CHARGEMENT DU MODÈLE

# Vérifie les arguments de la ligne de commande
if len(sys.argv) >= 3:
    pdf_path   = sys.argv[1]   # Chemin du PDF à analyser
    output_dir = sys.argv[2]   # Répertoire où sauvegarder les images découpées
else:
    raise IOError(
        "Usage : python3 pdf_zone_extractor.py <document.pdf> <repertoire_sortie/>"
    )

# Chargement du modèle YOLO de détection de zones
model = YOLO(YOLO_MODEL_PATH)


#  TRAITEMENT PAGE PAR PAGE

# Conversion du PDF en liste d'images PIL (une par page)
pages = convert_from_path(pdf_path, dpi=PDF_DPI)

for page_index, page in enumerate(pages):
    # Conversion PIL → numpy array (format RGB)
    page_array = np.array(page)

    # Prédiction YOLO sur la page courante
    # result[0] contient les détections de la première (et unique) image passée
    result = model.predict(source=page, conf=YOLO_CONF_THRESHOLD)

    for box in result[0].boxes:
        # Récupération des coordonnées de la boîte englobante (x1, y1, x2, y2)
        # ATTENTION : l'ordre correct est x1, y1, x2, y2 (pas x, y, y1, x1)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Ajout d'une marge autour de la zone pour ne pas rogner les bords
        y1_m = max(0, y1 - CROP_MARGIN_PX)
        y2_m = min(page_array.shape[0], y2 + CROP_MARGIN_PX)
        x1_m = max(0, x1 - CROP_MARGIN_PX)
        x2_m = min(page_array.shape[1], x2 + CROP_MARGIN_PX)

        # Découpage de la zone dans l'image de la page
        crop = page_array[y1_m:y2_m, x1_m:x2_m]

        # Vérification que le crop n'est pas vide
        if crop is not None and crop.size > 0:
            # Récupération du nom de la classe détectée (ex: "Table", "Diagram", "Board")
            class_name = result[0].names[int(box.cls[0])]

            # Construction du nom de fichier : <nom_pdf>_<page>_<classe>.png
            filename = f"{Path(pdf_path).stem}_{page_index}_{class_name}.png"

            # Sauvegarde en BGR (OpenCV attend BGR, l'image PIL est en RGB)
            cv2.imwrite(
                os.path.join(output_dir, filename),
                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            )

print(f"Traitement terminé. Images sauvegardées dans : {output_dir}")
