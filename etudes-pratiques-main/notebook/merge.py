import os
import sys
import cv2
import numpy as np
from pdf2image import convert_from_path
from ultralytics import YOLO

"""
python3 merge.py Accuphase-E202amp.pdf ../../../models/extraction/best.pt
"""

# --- AJOUT POUR L'IMPORT LOCAL ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from test1 import afficher, detect_refs_table
except ImportError as e:
    print(f"Erreur : Impossible de charger test1.py depuis /dev. {e}")
    sys.exit(1)

# ... (le reste de ton code merge.py reste identique)

def process_and_visualize(pdf_path, model_path):
    # 1. Chargement du modèle de zones (prediction.py)
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Erreur chargement modèle : {e}")
        return

    # 2. Conversion PDF en images (DPI 300)
    print("--- Conversion du PDF en images ---")
    pages = convert_from_path(pdf_path, dpi=300)
    
    best_images = {"Table": (None, 0), "Diagram": (None, 0), "Board": (None, 0)}

    print("--- Extraction des zones via YOLO ---")
    for p_id, page in enumerate(pages):
        # Utilisation directe du format PIL pour la détection comme dans prediction.py
        results = model.predict(source=page, conf=0.212, verbose=False)
        page_array = np.array(page)

        for box in results[0].boxes:
            cls_name = results[0].names[int(box.cls[0])]
            
            # Mapping vers nos clés
            key = None
            if cls_name.lower() == "table": key = "Table"
            elif cls_name.lower() == "diagram": key = "Diagram"
            elif cls_name.lower() == "board": key = "Board"

            if key:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Découpage avec marge de 10px
                margin = 10
                y1_m, y2_m = max(0, y1 - margin), min(page_array.shape[0], y2 + margin)
                x1_m, x2_m = max(0, x1 - margin), min(page_array.shape[1], x2 + margin)
                
                crop = page_array[y1_m:y2_m, x1_m:x2_m]
                
                if crop is not None and crop.size > 0:
                    surface = crop.shape[0] * crop.shape[1]
                    if surface > best_images[key][1]:
                        best_images[key] = (crop, surface)

    # 3. Traitement, Rotations et Sauvegarde
    temp_paths = {}
    for cls in ["Table", "Diagram", "Board"]:
        img_data, surf = best_images[cls]
        if img_data is not None:
            # Conversion RGB vers BGR pour OpenCV
            img_to_save = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

            # --- APPLICATION DES ROTATIONS ---
            if cls == "Diagram":
                # Rotation 90° vers la GAUCHE (Counter-Clockwise)
                print("Rotation du Schéma (90° gauche)...")
                img_to_save = cv2.rotate(img_to_save, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            elif cls == "Board":
                # Rotation 90° vers la DROITE (Clockwise)
                print("Rotation de la Carte (90° droite)...")
                img_to_save = cv2.rotate(img_to_save, cv2.ROTATE_90_CLOCKWISE)

            path = f"temp_{cls.lower()}.png"
            cv2.imwrite(path, img_to_save)
            temp_paths[cls] = path

    # 4. Lancement de l'interface test1.py
    if "Table" in temp_paths and "Diagram" in temp_paths:
        print(f"--- Affichage ---")
        afficher(
            temp_paths["Table"], 
            temp_paths["Diagram"], 
            temp_paths.get("Board")
        )
    else:
        print("\nERREUR : 'Table' ou 'Diagram' manquant.")
        print(f"Trouvés : {[k for k, v in best_images.items() if v[0] is not None]}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 merge.py <document.pdf> <modele_extraction.pt>")
    else:
        process_and_visualize(sys.argv[1], sys.argv[2])
