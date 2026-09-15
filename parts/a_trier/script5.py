import cv2
import easyocr
import numpy as np
import re
import os
import sys
from PIL import Image

# ================= CONFIGURATION =================
IMAGE_PATH = sys.argv[1]
OUTPUT_TXT = "composants_detectes.txt"
TILE_SIZE = 1000          # Taille de la fenêtre de découpe (plus grand = plus de contexte)
TILE_OVERLAP = 0.25       # 25% de chevauchement pour ne pas couper de texte
UPSCALE_FACTOR = 2.0      # Zoom sur chaque tuile pour aider l'IA
MIN_CONFIDENCE = 0.25     # Seuil de confiance (assez bas, on filtre par Regex après)

# Composants à chercher (Regex stricte)
# R=Resistance, C=Condensateur, T/Q=Transistor, L=Inductance, D=Diode, IC=Puce
COMPONENT_REGEX = r"^(R|C|L|D|T|Q|IC|BC|BF)\s*[0-9\s]{1,6}[A-Z]?$"

# ================= 1. MOTEUR IA (PYTORCH) =================
print("Chargement du modèle Deep Learning (EasyOCR/PyTorch)...")
# gpu=True est recommandé si vous avez une carte NVIDIA, sinon False
reader = easyocr.Reader(['en'], gpu=True) 

# ================= 2. FONCTIONS DE TRAITEMENT =================

def preprocess_tile(tile):
    """
    Prépare une petite portion de l'image pour l'OCR.
    Améliore le contraste et agrandit l'image.
    """
    # Conversion gris
    if len(tile.shape) == 3:
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    else:
        gray = tile

    # Agrandissement (Super-Resolution "Pauvre")
    # L'interpolation CUBIC est meilleure pour le texte
    tile_upscaled = cv2.resize(gray, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)

    # CLAHE : Augmentation locale du contraste (détache le texte noir du fond gris)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(tile_upscaled)

    return enhanced

def clean_text_logic(text):
    """
    Nettoie et corrige les erreurs OCR typiques sur les schémas.
    """
    text = text.upper()
    # Supprimer tout ce qui n'est pas alphanumérique
    text = re.sub(r"[^A-Z0-9]", "", text)
    
    # Corrections courantes (I -> 1, O -> 0, etc.)
    corrections = {"I": "1", "L": "1", "O": "0", "Z": "2", "S": "5", "B": "8"}
    # On applique les corrections seulement si ça aide à matcher le pattern
    # Ici on fait une correction brute sur les chiffres potentiels
    
    # Si le texte commence par R, C, etc, on s'assure que la suite est des chiffres
    match = re.match(r"^([A-Z]+)(.*)$", text)
    if match:
        prefix, rest = match.groups()
        if prefix in ["R", "C", "L", "D", "T", "Q"]:
            # Forcer les conversions de lettres en chiffres dans la partie numérique
            for k, v in corrections.items():
                rest = rest.replace(k, v)
            text = prefix + rest
            
    return text

def non_max_suppression_fast(boxes, overlapThresh=0.3):
    """
    Fusionne les rectangles qui se chevauchent (doublons dus au tuilage).
    """
    if len(boxes) == 0: return []
    
    # Format boxes: [[x1, y1, x2, y2, conf, text], ...]
    if boxes.dtype.kind == "i": boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    score = boxes[:,4]
    
    # Trier par score de confiance
    idxs = np.argsort(score)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / ((x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1))

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick]

# ================= 3. LOGIQUE DE TUILAGE (TILING) =================

def scan_image_tiled(img_path):
    img_cv = cv2.imread(img_path)

    # Prétraitement de l'image
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    preprocessed_path = "preprocessed_" + img_path
    cv2.imwrite(preprocessed_path, thresh)
    img_pre = Image.open(preprocessed_path)
    img_pre = np.array(Image.open(preprocessed_path))

    if img_pre is None: raise ValueError("Image introuvable")
    
    h_img, w_img = img_pre.shape[:2]
    
    # Calcul du pas (stride) avec chevauchement
    stride = int(TILE_SIZE * (1 - TILE_OVERLAP))
    
    detected_objects = [] # Liste brute
    
    print(f"Démarrage du scan sur image {w_img}x{h_img}...")
    print(f"Taille tuile: {TILE_SIZE}px | Stride: {stride}px")

    total_tiles = ((h_img // stride) + 1) * ((w_img // stride) + 1)
    processed_count = 0

    for y in range(0, h_img, stride):
        for x in range(0, w_img, stride):
            processed_count += 1
            print(f"Traitement tuile {processed_count}/{total_tiles}...", end="\r")
            
            # Découpe
            y_end = min(y + TILE_SIZE, h_img)
            x_end = min(x + TILE_SIZE, w_img)
            tile = img_pre[y:y_end, x:x_end]
            
            if tile.size == 0: continue

            # --- ANALYSE DE LA TUILE (0° et 90°) ---
            # On prépare la tuile
            tile_prep = preprocess_tile(tile)
            
            # Liste des rotations à tester pour cette tuile
            rotations = [0, 90] 
            
            for angle in rotations:
                if angle == 90:
                    img_to_read = cv2.rotate(tile_prep, cv2.ROTATE_90_CLOCKWISE)
                else:
                    img_to_read = tile_prep

                # OCR
                results = reader.readtext(img_to_read, mag_ratio=1, batch_size=4, width_ths=0.7)

                for (bbox, text, conf) in results:
                    if conf < MIN_CONFIDENCE: continue
                    
                    clean = clean_text_logic(text)
                    
                    # Filtre Regex
                    if re.match(COMPONENT_REGEX, clean):
                        
                        # --- RECALCUL DES COORDONNÉES GLOBALES ---
                        # 1. Coordonnées dans la tuile (upscaled)
                        (tl, tr, br, bl) = bbox
                        
                        if angle == 90:
                            # Inversion de la rotation 90°
                            # x' = y, y' = w - x
                            h_rot, w_rot = img_to_read.shape
                            rx_min = min(tl[0], br[0])
                            ry_min = min(tl[1], br[1])
                            rx_max = max(tl[0], br[0])
                            ry_max = max(tl[1], br[1])
                            
                            # Conversion inverse
                            lx_min = ry_min
                            ly_min = w_rot - rx_max
                            lx_max = ry_max
                            ly_max = w_rot - rx_min
                        else:
                            lx_min = min(tl[0], br[0])
                            ly_min = min(tl[1], br[1])
                            lx_max = max(tl[0], br[0])
                            ly_max = max(tl[1], br[1])

                        # 2. Coordonnées réelles dans la tuile (downscale)
                        lx_min /= UPSCALE_FACTOR
                        ly_min /= UPSCALE_FACTOR
                        lx_max /= UPSCALE_FACTOR
                        ly_max /= UPSCALE_FACTOR

                        # 3. Coordonnées globales (ajout offset x, y)
                        gx_min = int(lx_min + x)
                        gy_min = int(ly_min + y)
                        gx_max = int(lx_max + x)
                        gy_max = int(ly_max + y)

                        detected_objects.append([gx_min, gy_min, gx_max, gy_max, conf, clean])

    print("\nScan terminé. Fusion des résultats...")
    return img_pre, detected_objects

# ================= 4. MAIN =================

if __name__ == "__main__":
    try:
        image, raw_objects = scan_image_tiled(IMAGE_PATH)
        
        # Conversion en numpy array pour le NMS
        # On doit séparer le texte (str) des chiffres (coords) pour numpy
        if len(raw_objects) > 0:
            coords_conf = np.array([o[:5] for o in raw_objects])
            texts = [o[5] for o in raw_objects]
            
            # Application de la suppression des non-maxima (NMS)
            # Pour éviter d'avoir 2 cadres sur le même objet à cause du chevauchement
            final_boxes = non_max_suppression_fast(coords_conf)
            
            # Récupérer les textes associés (méthode naïve post-NMS)
            # On recrée une liste propre
            cleaned_results = []
            for box in final_boxes:
                # Retrouver le texte correspondant à cette boite exacte
                # (Comparaison float peut être tricky, on utilise une petite tolérance)
                for i, raw in enumerate(raw_objects):
                    if np.allclose(box, raw[:5], atol=1.0):
                        cleaned_results.append({
                            "name": raw[5],
                            "conf": raw[4],
                            "bbox": [int(b) for b in box[:4]]
                        })
                        break
        else:
            cleaned_results = []

        print(f"Total composants trouvés : {len(cleaned_results)}")

        # --- EXPORT ET VISUALISATION ---
        
        # 1. Sauvegarde TXT
        if len(cleaned_results) > 50:
            print(f"Génération du fichier {OUTPUT_TXT}...")
            with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
                f.write("NOM_COMPOSANT, X_MIN, Y_MIN, X_MAX, Y_MAX, CONFIANCE\n")
                # Trier par nom pour être propre (C1, C2, R1...)
                cleaned_results.sort(key=lambda x: x["name"])
                
                for res in cleaned_results:
                    line = f"{res['name']}, {res['bbox'][0]}, {res['bbox'][1]}, {res['bbox'][2]}, {res['bbox'][3]}, {res['conf']:.2f}\n"
                    f.write(line)
        
        # 2. Dessin sur l'image
        for res in cleaned_results:
            x1, y1, x2, y2 = res["bbox"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Fond noir pour le texte
            label = res['name']
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), (0, 0, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Affichage final (réduit pour l'écran)
        plt_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 20))
        plt.imshow(plt_img)
        plt.axis('off')
        plt.title(f"Detection Finale: {len(cleaned_results)} composants")
        plt.show()
        
        # Sauvegarder l'image résultat
        cv2.imwrite("resultat_final.jpg", image)
        print("Image sauvegardée sous 'resultat_final.jpg'")

    except Exception as e:
        print(f"Erreur fatale : {e}")