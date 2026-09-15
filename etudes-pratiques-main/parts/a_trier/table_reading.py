#!/usr/bin/env python3

import cv2
from PIL import Image
import pytesseract
import re
import csv
import os
import sys

# -------------------------
# 1. Image loading
# -------------------------
if len(sys.argv) < 2:
    print("Usage: python3 table_reading.py <image.png>")
    sys.exit(1)

img_path = sys.argv[1]

if not os.path.exists(img_path):
    raise FileNotFoundError(f"{img_path} not found")

img_cv_orig = cv2.imread(img_path)
if img_cv_orig is None:
    raise FileNotFoundError(f"Unable to load {img_path}")

print(f"Successful loading of {img_path}")

# -------------------------
# 2. Crop image
# -------------------------
h, w, _ = img_cv_orig.shape

crop_width = int(w * 0.14)
crop_height = int(h * 0.08)

img_cv = img_cv_orig[crop_height:, :crop_width]

cv2.imwrite("cropped_debug.png", img_cv)
print("Cropped image saved: cropped_debug.png")

# -------------------------
# 6. OCR
# -------------------------
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,~'
data = pytesseract.image_to_data(
    img_cv,
    config=custom_config,
    output_type=pytesseract.Output.DICT
)

# -------------------------
# 7. Group words by line
# -------------------------
lines = {}

for i in range(len(data['text'])):
    txt = data['text'][i].strip()
    conf = int(data['conf'][i])

    if txt == "":
        continue

    top = data['top'][i]
    left = data['left'][i]

    # Group by approximate Y position
    line_key = round(top / 12) * 12

    if line_key not in lines:
        lines[line_key] = []

    # Store also index to retrieve box later
    lines[line_key].append((left, txt, conf, i))

# -------------------------
# 8. Reference detection
# -------------------------
references = []

ref_pattern = re.compile(
    r'\b([RCULDQFJ]*)\s*(\d{1,4})(?:\s*[~,–-]\s*(\d{1,4}))?(?:\s*,\s*(\d{1,4}))?\b'
)

for line_key in sorted(lines.keys()):
    words = sorted(lines[line_key], key=lambda x: x[0])
    full_line = " ".join(w[1] for w in words)

    for match in re.finditer(ref_pattern, full_line.upper()):
        letter = match.group(1)
        start = match.group(2)
        end = match.group(3)
        extra = match.group(4)

        if end:
            ref = f"{letter}{start}~{end}"
        elif extra:
            ref = f"{letter}{start},{extra}"
        else:
            ref = f"{letter}{start}"

        # Take first word in line as position anchor
        word_index = words[0][3]

        x_crop = data['left'][word_index]
        y_crop = data['top'][word_index]

        # Convert to original image coordinates
        x_orig = x_crop
        y_orig = y_crop + crop_height

        references.append({
            "ref": ref,
            "x": x_orig,
            "y": y_orig
        })

# -------------------------
# 9. CSV export
# -------------------------
csv_path = "detected_refs.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["ref", "x", "y"])
    writer.writeheader()
    writer.writerows(references)

print(f"\nGenerated CSV: {csv_path}")
print(f"Number of references detected: {len(references)}")
