import cv2
from PIL import Image
import pytesseract
import re
import csv
import os

# -----------------------------
# 1️⃣ Charger l'image
# -----------------------------
# Chemin relatif, car schema.png est dans le même dossier que le script
img_path = "schema.png"

if not os.path.exists(img_path):
    raise FileNotFoundError(f"L'image {img_path} est introuvable dans ce dossier")

img_cv = cv2.imread(img_path)
if img_cv is None:
    raise FileNotFoundError("Impossible de charger schema.png : vérifie le chemin !")

print("Image chargée avec succès !")

# -----------------------------
# 2️⃣ Convertir en niveaux de gris
# -----------------------------
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 3️⃣ Prétraitement pour OCR
# -----------------------------
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
thresh = cv2.bitwise_not(thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
thresh = cv2.dilate(thresh, kernel, iterations=1)

# Sauvegarde pour vérification
preprocessed_path = "schema_preprocessed.png"
cv2.imwrite(preprocessed_path, thresh)
print(f"Image prétraitée sauvegardée sous : {preprocessed_path}")

# -----------------------------
# 4️⃣ OCR avec Tesseract
# -----------------------------
img_pre = Image.open(preprocessed_path)
data = pytesseract.image_to_data(img_pre, output_type=pytesseract.Output.DICT)

# -----------------------------
# 5️⃣ Filtrer les composants (ex: R25, C14, L3…)
# -----------------------------
pattern = re.compile(r'^[RCULDIC]+\d+$', re.IGNORECASE)
components = []

for i in range(len(data['text'])):
    txt = data['text'][i].strip()
    conf = int(data['conf'][i])
    if txt != "" and conf > 0 and pattern.match(txt):
        components.append({
            "name": txt,
            "left": data['left'][i],
            "top": data['top'][i],
            "width": data['width'][i],
            "height": data['height'][i]
        })

# -----------------------------
# 6️⃣ Écrire le CSV
# -----------------------------
csv_path = "components_detected.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name","left","top","width","height"])
    writer.writeheader()
    writer.writerows(components)

print(f"➡️ CSV généré : {csv_path}")
print(f"➡️ Nombre de composants détectés : {len(components)}")
