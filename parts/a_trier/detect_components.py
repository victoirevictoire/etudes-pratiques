import cv2
from PIL import Image
import pytesseract
import re
import csv

# -----------------------------
# 1️⃣ Charger l'image
# -----------------------------
img_path = "/home/ilyas/detection_schema/schema.png"
img_cv = cv2.imread(img_path)
if img_cv is None:
    raise FileNotFoundError(f"L'image {img_path} est introuvable")

# -----------------------------
# 2️⃣ Convertir en niveaux de gris
# -----------------------------
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 3️⃣ Prétraitement pour OCR
# -----------------------------
# Seuillage
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# Inversion (texte blanc sur fond noir)
thresh = cv2.bitwise_not(thresh)
# Dilatation pour renforcer le texte
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
thresh = cv2.dilate(thresh, kernel, iterations=1)
# Sauvegarde pour vérification
cv2.imwrite("schema_preprocessed.png", thresh)

# -----------------------------
# 4️⃣ OCR avec Tesseract
# -----------------------------
img_pre = Image.open("schema_preprocessed.png")
data = pytesseract.image_to_data(img_pre, output_type=pytesseract.Output.DICT)

# -----------------------------
# 5️⃣ Afficher tout le texte détecté pour vérifier
# -----------------------------
for i in range(len(data['text'])):
    txt = data['text'][i].strip()
    if txt != "":
        print(f"{txt} | conf={data['conf'][i]} | x={data['left'][i]} y={data['top'][i]} w={data['width'][i]} h={data['height'][i]}")

# -----------------------------
# 6️⃣ (Optionnel) Filtrer composants selon regex
# -----------------------------
pattern = re.compile(r'^[RCULDIC]+\d+$', re.IGNORECASE)
components = []
for i in range(len(data['text'])):
    txt = data['text'][i].strip()
    conf = int(data['conf'][i])
    if pattern.match(txt) and conf > 0:
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        components.append({'name': txt, 'left': x, 'top': y, 'width': w, 'height': h})

# -----------------------------
# 7️⃣ Écrire le CSV complet (ou juste les composants filtrés)
# -----------------------------
with open("all_text.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name","left","top","width","height"])
    writer.writeheader()
    # Pour test : écrire tout le texte détecté
    for i in range(len(data['text'])):
        txt = data['text'][i].strip()
        if txt != "":
            writer.writerow({
                "name": txt,
                "left": data['left'][i],
                "top": data['top'][i],
                "width": data['width'][i],
                "height": data['height'][i]
            })
