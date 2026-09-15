import cv2 # OpenCV
from PIL import Image
import pytesseract
import re
import csv
import os
import sys

# 1. Image loading
img_path = sys.argv[1]
if not os.path.exists(img_path):
    raise FileNotFoundError(f"{img_path} is missingfrom the current folder")
img_cv = cv2.imread(img_path)
if img_cv is None:
    raise FileNotFoundError(f"Impossible to load {img_path}")
print(f"Successful loading of {img_path}")
# 2. Convertion to grayscale
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
# 3. Preprocessing for OCR
# Adaptative threshold + inversion
thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)
# Expension to join the letters
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
thresh = cv2.dilate(thresh, kernel, iterations=1)
# Backup for checking
preprocessed_path = "preprocessed_"+img_path+".png"
cv2.imwrite(preprocessed_path, thresh)
print(f"Image prétraitée sauvegardée sous : {preprocessed_path}")
# 4. OCR with Tesseract
img_pre = Image.open(preprocessed_path)
data = pytesseract.image_to_data(img_pre, output_type=pytesseract.Output.DICT)
print(f"data : {data}")
# 5. Debug : print all the detected text
print("\n=== Detected text by Tesseract ===")
for i in range(len(data['text'])):
    txt = data['text'][i].strip()
    conf = int(data['conf'][i])
    if txt != "":
        print(repr(txt), "conf=", conf)
# 6. Filtering components and cleaning the text
components = list()
for i in range(len(data['text'])):
    txt = data['text'][i].strip()
    conf = int(data['conf'][i])
    if txt == "":
        continue
    # Cleaning the text to keep only letters and digits
    txt_clean = re.sub(r'[^RCULDIC0-9-]', '', txt.upper())
    # Checking if it is a components
    if re.match(r'[RCULDIC]+\d+', txt_clean):
        components.append({
            "name": txt_clean,
            "left": data['left'][i],
            "top": data['top'][i],
            "width": data['width'][i],
            "height": data['height'][i],
            "conf": conf})
# 7. Writting of the CSV file
csv_path = "detected_components.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name","left","top","width","height","conf"])
    writer.writeheader()
    writer.writerows(components)
print(f"\n️ Generated CSV : {csv_path}")
print(f" Number of detected components : {len(components)}")
