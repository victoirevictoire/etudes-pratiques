from PIL import Image
import pytesseract
import cv2

# Charge l'image
img = cv2.imread("/home/ilyas/detection_schema/schema.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Seuillage pour améliorer le contraste
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Sauvegarde l'image prétraitée pour vérification
cv2.imwrite("schema_preprocessed.png", thresh)

# OCR
from PIL import Image
img_pre = Image.open("schema_preprocessed.png")
text = pytesseract.image_to_string(img_pre)
print("Texte détecté par Tesseract :")
print(text)

