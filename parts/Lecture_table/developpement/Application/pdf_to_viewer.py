"""
pdf_to_viewer.py
================
Pipeline complet : PDF → détection de zones YOLO → interface de visualisation.

Ce script orchestre les étapes suivantes :
  1. Conversion du PDF en images haute résolution (DPI 300)
  2. Détection des zones (Table, Diagram, Board) via YOLO
  3. Sélection de la meilleure zone par classe (surface maximale)
  4. Application des rotations nécessaires selon la classe
  5. Lancement de l'interface interactive (viewer.py)

Usage:
    python3 pdf_to_viewer.py <document.pdf> <modele_detection.pt>

Exemple:
    python3 pdf_to_viewer.py Accuphase-E202amp.pdf ../../../models/extraction/best.pt
"""

import os
import sys
import cv2
import numpy as np
from pdf2image import convert_from_path
from ultralytics import YOLO

# Permet d'importer viewer.py depuis le même répertoire,
# même si le script est lancé depuis un autre dossier
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from viewer import show_interface, detect_refs_table
except ImportError as e:
    print(f"Erreur : Impossible de charger viewer.py. {e}")
    sys.exit(1)


#  CONFIGURATION

# Résolution de conversion PDF → image
PDF_DPI = 300

# Seuil de confiance YOLO pour la détection de zones
YOLO_CONF_THRESHOLD = 0.212

# Marge en pixels ajoutée autour de chaque zone découpée
CROP_MARGIN_PX = 10

# Classes gérées et leurs éventuelles rotations
# None = pas de rotation, cv2.ROTATE_* = rotation appliquée avant sauvegarde
ZONE_ROTATIONS = {
    "Table":   None,
    "Diagram": cv2.ROTATE_90_COUNTERCLOCKWISE,   # Schéma souvent en paysage → portrait
    "Board":   cv2.ROTATE_90_CLOCKWISE,           # Carte PCB souvent en paysage → portrait
}


#  PIPELINE PRINCIPAL

def run_pipeline(pdf_path: str, model_path: str):
    """
    Traite un PDF complet et lance l'interface de visualisation.

    Paramètres
    ----------
    pdf_path   : chemin vers le fichier PDF à analyser
    model_path : chemin vers le modèle YOLO de détection de zones (.pt)
    """

    # --- Chargement du modèle YOLO ---
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Erreur chargement modèle YOLO : {e}")
        return

    # --- Conversion PDF → images ---
    print("--- Conversion du PDF en images (DPI 300) ---")
    pages = convert_from_path(pdf_path, dpi=PDF_DPI)

    # Dictionnaire pour stocker la meilleure image (surface max) par classe
    # Format : { "Table": (image_array, surface), ... }
    best_zones = {cls: (None, 0) for cls in ZONE_ROTATIONS}

    # --- Détection des zones sur chaque page ---
    print("--- Détection des zones via YOLO ---")
    for page_index, page in enumerate(pages):

        # Prédiction YOLO sur la page (format PIL accepté directement)
        results = model.predict(source=page, conf=YOLO_CONF_THRESHOLD, verbose=False)

        # Conversion PIL → numpy pour le découpage
        page_array = np.array(page)

        for box in results[0].boxes:
            class_name = results[0].names[int(box.cls[0])]

            # Normalisation du nom de classe pour correspondre aux clés du dictionnaire
            class_key = class_name.capitalize()
            if class_key not in best_zones:
                continue   # Classe inconnue, ignorée

            # Coordonnées de la boîte englobante
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Découpage avec marge de sécurité
            y1_m = max(0, y1 - CROP_MARGIN_PX)
            y2_m = min(page_array.shape[0], y2 + CROP_MARGIN_PX)
            x1_m = max(0, x1 - CROP_MARGIN_PX)
            x2_m = min(page_array.shape[1], x2 + CROP_MARGIN_PX)
            crop = page_array[y1_m:y2_m, x1_m:x2_m]

            if crop is None or crop.size == 0:
                continue

            # On conserve uniquement la zone de plus grande surface par classe
            # (filtre les petites détections parasites)
            surface = crop.shape[0] * crop.shape[1]
            if surface > best_zones[class_key][1]:
                best_zones[class_key] = (crop, surface)

    # --- Sauvegarde des zones temporaires avec rotation ---
    temp_paths = {}

    for class_key, (img_data, _) in best_zones.items():
        if img_data is None:
            continue

        # Conversion RGB (PIL) → BGR (OpenCV)
        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

        # Application de la rotation si définie pour cette classe
        rotation = ZONE_ROTATIONS[class_key]
        if rotation is not None:
            print(f"Rotation appliquée sur : {class_key}")
            img_bgr = cv2.rotate(img_bgr, rotation)

        # Sauvegarde dans un fichier temporaire
        temp_path = f"temp_{class_key.lower()}.png"
        cv2.imwrite(temp_path, img_bgr)
        temp_paths[class_key] = temp_path

    # --- Lancement de l'interface ---
    # Table et Diagram sont obligatoires, Board est optionnel
    if "Table" in temp_paths and "Diagram" in temp_paths:
        print("--- Lancement de l'interface de visualisation ---")
        show_interface(
            table_path   = temp_paths["Table"],
            schema_path  = temp_paths["Diagram"],
            board_path   = temp_paths.get("Board")   # None si absent
        )
    else:
        # Diagnostic en cas de zones manquantes
        found = [k for k, (img, _) in best_zones.items() if img is not None]
        print(f"\nERREUR : 'Table' ou 'Diagram' non détecté dans le PDF.")
        print(f"Zones trouvées : {found if found else 'aucune'}")


#  POINT D'ENTRÉE

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 pdf_to_viewer.py <document.pdf> <modele_detection.pt>")
        sys.exit(1)

    run_pipeline(
        pdf_path   = sys.argv[1],
        model_path = sys.argv[2]
    )
