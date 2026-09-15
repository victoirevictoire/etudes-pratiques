import os
import sys
from pathlib import Path
from pdf2image import convert_from_path
import numpy as np
import cv2

from ultralytics import YOLO

#Chargement du modèle entraîné
model = YOLO("../../models/extraction/best.pt")

if len(sys.argv) >= 3:
    pdf_path = sys.argv[1]
    dir_path = sys.argv[2]
else:
    raise IOError("Utilisation : il faut mettre le document a analyser et le répertoire destinataire en paramètre")

# Commande pour convertir le pdf en images
pages = convert_from_path(pdf_path, dpi=300)

for p_id, page in enumerate(pages) :
    page_array = np.array(page)

    # On prédit les classes présentes dans le document
    result = model.predict(source=page, conf=0.212)  # A voir s'il faut ajuster les paramètres

    # On prend le premier car on fait page par page
    for box in result[0].boxes:
        # Récupère les coordonées de la zone
        x, y, y1, x1 = map(int, box.xyxy[0])


        # On ajoute l'image pour qu'elle ne montre que la zone détéctée + une marge au cas où

        margin = 10
        y1_m = max(0, y - margin)
        y2_m = min(page_array.shape[0], y1 + margin)
        x1_m = max(0, x - margin)
        x2_m = min(page_array.shape[1], x1 + margin)

        crop = page_array[y1_m:y2_m, x1_m:x2_m]

        if crop is not None and crop.size > 0:
            # Récupère la classe
            cls = result[0].names[int(box.cls[0])]

            # Sauvegarde de l'image découpée
            filename = f"{Path(pdf_path).stem}_{p_id}_{cls}.png"
            cv2.imwrite(os.path.join(dir_path, filename), crop)

print(f"Le traitement du fichier est terminé")

