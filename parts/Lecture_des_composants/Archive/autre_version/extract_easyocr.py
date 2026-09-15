"""
extract_easyocr.py  v5
======================
EasyOCR + correcteur de confusions de caractères PCB :
  I→1, J→1, O→0, G→6, Z→2, S→5, B→8 dans les positions numériques

Usage:
  python3 extract_easyocr.py --img carte.png --model best.pt
"""

import cv2
import numpy as np
import easyocr
import argparse
import json
import csv
import re
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


# ──────────────────────────────────────────────────────────
#  1. EASYOCR
# ──────────────────────────────────────────────────────────

print("⏳ Chargement EasyOCR...")
READER = easyocr.Reader(['en'], gpu=True, verbose=False)
print("✅ EasyOCR prêt")


# ──────────────────────────────────────────────────────────
#  2. CORRECTEUR DE CONFUSIONS
# ──────────────────────────────────────────────────────────

# Confusions classiques OCR sur refs composants PCB
# Règle : dans la partie NUMÉRIQUE d'une ref, certains caractères
# sont toujours des chiffres (jamais des lettres)
LETTER_FIXES = {
    # Dans la partie lettre (préfixe) : chiffres → lettres
    '0': 'O', '1': 'I',
}
DIGIT_FIXES = {
    # Dans la partie chiffre (suffixe) : lettres → chiffres
    'I': '1', 'J': '1', 'L': '1',
    'O': '0', 'Q': '0',
    'G': '6',
    'Z': '2',
    'S': '5',
    'B': '8',
}

def fix_ocr_confusion(text):
    """
    Corrige les confusions OCR en analysant la structure de la ref.
    Ex: "BC17I" → "BC178" n'est pas possible sans contexte,
        mais "C41J" → "C411" (J en position numérique → 1)
        et  "R51G"  → "R516" (G en position numérique → 6)

    Stratégie :
      - Trouver la frontière lettres/chiffres
      - Corriger les caractères mal reconnus dans chaque zone
    """
    if not text:
        return text

    text = text.strip().upper()

    # Trouver où finissent les lettres et commencent les chiffres
    # Format typique : [LETTRES][CHIFFRES][LETTRE_OPTIONNELLE]
    # Ex: R516, BC178, IC402A, C 407
    match = re.match(r'^([A-Z]+)\s?(\d[\dA-Z]*?)([A-Z]?)$', text)
    if not match:
        return text

    prefix  = match.group(1)   # "BC", "R", "IC"
    numbers = match.group(2)   # "17I", "516", "402"
    suffix  = match.group(3)   # "A" ou ""

    # Corriger la partie numérique : lettres → chiffres
    fixed_numbers = ''
    for ch in numbers:
        if ch.isdigit():
            fixed_numbers += ch
        else:
            fixed_numbers += DIGIT_FIXES.get(ch, ch)

    result = prefix + fixed_numbers + suffix

    # Réintégrer l'espace si présent dans l'original
    if ' ' in text:
        result = prefix + ' ' + fixed_numbers + suffix

    return result


# ──────────────────────────────────────────────────────────
#  3. PREPROCESSING
# ──────────────────────────────────────────────────────────

def preprocess_crop(crop_bgr, scale=4):
    h, w = crop_bgr.shape[:2]
    resized  = cv2.resize(crop_bgr, (w * scale, h * scale),
                          interpolation=cv2.INTER_CUBIC)
    kernel   = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp    = cv2.filter2D(resized, -1, kernel)
    gray     = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    bgr      = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    padded   = cv2.copyMakeBorder(bgr, 10, 10, 10, 10,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return padded


# ──────────────────────────────────────────────────────────
#  4. REGEX VALIDATION
# ──────────────────────────────────────────────────────────

COMPONENT_PATTERN = re.compile(
    r'^('
    r'[A-Z]{1,2}\s\d{3,5}[A-Z]?'
    r'|[A-Z]{1,3}\d{2,5}[A-Z]?'
    r'|St\s?\d{3}'
    r'|[A-Z]{1,3}\d+[-_]\d+'
    r')$',
    re.IGNORECASE
)


# ──────────────────────────────────────────────────────────
#  5. OCR
# ──────────────────────────────────────────────────────────

def run_ocr(crop_bgr):
    preprocessed = preprocess_crop(crop_bgr)
    try:
        results = READER.readtext(preprocessed, detail=1)
    except Exception:
        return None, False, 0.0

    if not results:
        return None, False, 0.0

    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)

    candidates = []
    for (_, text, conf) in results_sorted:
        text_clean = re.sub(r'[^\w\-_\.\s]', '', text).strip().upper()
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        if text_clean:
            # Appliquer le correcteur de confusions
            text_fixed = fix_ocr_confusion(text_clean)
            candidates.append((text_fixed, conf))

    if not candidates:
        return None, False, 0.0

    # Priorité aux refs valides après correction
    for (text, conf) in candidates:
        if COMPONENT_PATTERN.match(text):
            return text, True, conf

    return candidates[0][0], False, candidates[0][1]


# ──────────────────────────────────────────────────────────
#  6. PIPELINE
# ──────────────────────────────────────────────────────────

def extract(img_path, model_path, conf=0.3):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {img_path}")
    print(f"  📐 Image : {img.shape[1]}×{img.shape[0]} px")

    model = YOLO(str(model_path))
    yolo_results = model(img, conf=conf, imgsz=1024, verbose=False)[0]
    boxes = yolo_results.boxes
    print(f"  📦 {len(boxes)} zone(s) détectée(s)  (conf ≥ {conf})")
    print(f"  ⏳ OCR + correction en cours...")

    detections = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf_yolo = float(box.conf[0])
        pad = 4
        crop = img[max(0,y1-pad):min(img.shape[0],y2+pad),
                   max(0,x1-pad):min(img.shape[1],x2+pad)]
        if crop.size == 0:
            continue

        text, is_valid, ocr_conf = run_ocr(crop)
        detections.append({
            "id":           i + 1,
            "text":         text or "",
            "is_valid_ref": is_valid,
            "yolo_conf":    round(conf_yolo, 3),
            "ocr_conf":     round(ocr_conf, 3),
            "bbox":         {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "center":       {"x": (x1+x2)//2, "y": (y1+y2)//2},
        })
        if (i+1) % 50 == 0:
            print(f"     {i+1}/{len(boxes)} crops traités...")

    return img, detections


# ──────────────────────────────────────────────────────────
#  7. VISUALISATION + EXPORTS
# ──────────────────────────────────────────────────────────

def draw_results(img, detections):
    output = img.copy()
    for det in detections:
        b = det["bbox"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        color = (0, 200, 0) if det["is_valid_ref"] else (0, 140, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        if det["text"]:
            fs = 0.4
            (tw, th), _ = cv2.getTextSize(det["text"], cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.rectangle(output, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(output, det["text"], (x1+2, y1-3),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 1)
    return output

def save_json(detections, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)
    print(f"  💾 JSON → {path}")

def save_csv(detections, path):
    fields = ["id","text","is_valid_ref","yolo_conf","ocr_conf",
              "x1","y1","x2","y2","cx","cy"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in detections:
            w.writerow({
                "id": d["id"], "text": d["text"],
                "is_valid_ref": d["is_valid_ref"],
                "yolo_conf": d["yolo_conf"], "ocr_conf": d["ocr_conf"],
                "x1": d["bbox"]["x1"], "y1": d["bbox"]["y1"],
                "x2": d["bbox"]["x2"], "y2": d["bbox"]["y2"],
                "cx": d["center"]["x"], "cy": d["center"]["y"],
            })
    print(f"  💾 CSV  → {path}")


# ──────────────────────────────────────────────────────────
#  8. MAIN
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",   required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--conf",  type=float, default=0.3)
    args = parser.parse_args()

    img_path   = Path(args.img)
    model_path = Path(args.model)
    if not img_path.exists():
        print(f"❌ Image introuvable : {args.img}"); return
    if not model_path.exists():
        print(f"❌ Modèle introuvable : {args.model}"); return

    print(f"\n{'═'*50}")
    print(f"  PCB Extractor v5 — EasyOCR + correction")
    print(f"{'═'*50}")

    img, detections = extract(img_path, model_path, conf=args.conf)

    valid  = [d for d in detections if d["is_valid_ref"]]
    others = [d for d in detections if d["text"] and not d["is_valid_ref"]]

    print(f"\n{'─'*50}")
    print(f"  Zones détectées   : {len(detections)}")
    print(f"  Références valides: {len(valid)}")
    print(f"  Texte non reconnu : {len(others)}")
    print(f"{'─'*50}")

    if valid:
        print(f"\n🔧 Références trouvées :")
        for ref in sorted(set(d["text"] for d in valid)):
            print(f"   {ref}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = img_path.stem

    annotated = draw_results(img, detections)
    cv2.imwrite(str(out_dir / f"{stem}_{ts}_annotated.png"), annotated)
    print(f"\n  🖼️  Image → results/{stem}_{ts}_annotated.png")
    save_json(detections, out_dir / f"{stem}_{ts}_results.json")
    save_csv (detections, out_dir / f"{stem}_{ts}_results.csv")
    print(f"\n✅ Terminé ! Ouvre l'image avec :")
    print(f"   xdg-open results/{stem}_{ts}_annotated.png\n")


if __name__ == "__main__":
    main()
