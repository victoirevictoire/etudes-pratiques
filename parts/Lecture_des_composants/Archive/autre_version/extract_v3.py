"""
extract.py  v3
==============
Fixes v3 :
  - Suppression de l'upscale ×2 avant YOLO (causait des détections aberrantes)
  - Upscale ×6 des crops uniquement
  - Sharpen avant binarisation pour mieux détacher les caractères fins
  - Espace dans la whitelist Tesseract (refs "R 516", "C 407")
  - Regex accepte "R 516", "BC178", "St 401"

Usage:
  python3 extract.py --img carte.png --model best.pt
  python3 extract.py --img carte.png --model best.pt --conf 0.3 --show
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import argparse
import json
import csv
import re
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


# ──────────────────────────────────────────────────────────
#  1. PREPROCESSING — Prépare chaque crop pour Tesseract
# ──────────────────────────────────────────────────────────

def preprocess_crop(crop_bgr, scale=6):
    """
    Pipeline de preprocessing pour les petits textes PCB :
      1. Upscale ×6  (refs ~15px → ~90px, lisible par Tesseract)
      2. Sharpen     (accentue les contours des caractères)
      3. CLAHE       (contraste local)
      4. Débruitage
      5. Binarisation adaptative
      6. Padding blanc
    """
    h, w = crop_bgr.shape[:2]

    # 1. Upscale
    resized = cv2.resize(crop_bgr, (w * scale, h * scale),
                         interpolation=cv2.INTER_CUBIC)

    # 2. Sharpen — accentue les bords des caractères fins
    kernel_sharpen = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    sharpened = cv2.filter2D(resized, -1, kernel_sharpen)

    # 3. Niveaux de gris
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

    # 4. CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 5. Débruitage
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    # 6. Binarisation adaptative
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15, C=4
    )

    # Inverser si fond sombre
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # 7. Padding blanc
    padded = cv2.copyMakeBorder(binary, 15, 15, 15, 15,
                                 cv2.BORDER_CONSTANT, value=255)
    return padded


# ──────────────────────────────────────────────────────────
#  2. OCR
# ──────────────────────────────────────────────────────────

TESSERACT_CONFIGS = [
    # PSM 7 = ligne unique | whitelist avec espace pour "R 516"
    r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz-_. ',
    r'--oem 3 --psm 8',
    r'--oem 3 --psm 6',
]

# Refs avec espace : "R 516", "C 407", "IC 402"
# Refs sans espace : "R516", "BC178", "S402", "St401"
COMPONENT_PATTERN = re.compile(
    r'^('
    r'[A-Z]{1,2}\s\d{3,5}[A-Z]?'    # R 516 / C 407
    r'|[A-Z]{1,3}\d{2,5}[A-Z]?'     # R516 / BC178 / S402
    r'|St\s?\d{3}'                   # St 401
    r'|[A-Z]{1,3}\d+[-_]\d+'         # U1-3
    r')$',
    re.IGNORECASE
)


def run_ocr(crop_bgr):
    preprocessed = preprocess_crop(crop_bgr)
    pil_img = Image.fromarray(preprocessed)

    candidates = []
    for config in TESSERACT_CONFIGS:
        try:
            raw = pytesseract.image_to_string(pil_img, config=config)
            text = re.sub(r'[^\w\-_\.\s]', '', raw).strip().upper()
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                candidates.append(text)
        except Exception:
            continue

    if not candidates:
        return None, False

    for c in candidates:
        if COMPONENT_PATTERN.match(c):
            return c, True

    return max(candidates, key=len), False


def run_ocr_with_rotations(crop_bgr):
    best_text, best_valid, best_angle = None, False, 0

    for angle in [0, 90, 180, 270]:
        rotated = np.rot90(crop_bgr, k=angle // 90)
        text, is_valid = run_ocr(rotated)

        if not text:
            continue
        if is_valid and not best_valid:
            best_text, best_valid, best_angle = text, True, angle
            break
        if len(text) > len(best_text or ''):
            best_text, best_angle = text, angle

    return best_text, best_valid, best_angle


# ──────────────────────────────────────────────────────────
#  3. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────

def extract(img_path: Path, model_path: Path, conf: float = 0.3):
    """
    Étape 1 : YOLO sur l'image originale (pas d'upscale → détections correctes).
    Étape 2 : Crop de chaque zone + upscale ×6 + Tesseract.
    """

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {img_path}")
    print(f"  📐 Image : {img.shape[1]}×{img.shape[0]} px")

    print(f"  🤖 Modèle : {model_path.name}")
    model = YOLO(str(model_path))

    # YOLO sur l'image originale
    yolo_results = model(img, conf=conf, imgsz=1024, verbose=False)[0]
    boxes = yolo_results.boxes
    print(f"  📦 {len(boxes)} zone(s) détectée(s)  (conf ≥ {conf})")
    print(f"  ⏳ OCR en cours...")

    detections = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf_yolo = float(box.conf[0])

        pad = 4
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(img.shape[1], x2 + pad)
        cy2 = min(img.shape[0], y2 + pad)

        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        text, is_valid, angle = run_ocr_with_rotations(crop)

        detections.append({
            "id":           i + 1,
            "text":         text or "",
            "is_valid_ref": is_valid,
            "yolo_conf":    round(conf_yolo, 3),
            "rotation":     angle,
            "bbox":         {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "center":       {"x": (x1 + x2) // 2, "y": (y1 + y2) // 2},
        })

    return img, detections


# ──────────────────────────────────────────────────────────
#  4. VISUALISATION
# ──────────────────────────────────────────────────────────

def draw_results(img, detections):
    output = img.copy()
    for det in detections:
        b = det["bbox"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        text     = det["text"]
        is_valid = det["is_valid_ref"]
        color    = (0, 200, 0) if is_valid else (0, 140, 255)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs   = 0.4
            (tw, th), _ = cv2.getTextSize(text, font, fs, 1)
            cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(output, text, (x1 + 2, y1 - 3), font, fs, (255, 255, 255), 1)

    return output


# ──────────────────────────────────────────────────────────
#  5. EXPORTS
# ──────────────────────────────────────────────────────────

def save_json(detections, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)
    print(f"  💾 JSON → {path}")


def save_csv(detections, path):
    fields = ["id", "text", "is_valid_ref", "yolo_conf",
              "rotation", "x1", "y1", "x2", "y2", "cx", "cy"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in detections:
            w.writerow({
                "id":           d["id"],
                "text":         d["text"],
                "is_valid_ref": d["is_valid_ref"],
                "yolo_conf":    d["yolo_conf"],
                "rotation":     d["rotation"],
                "x1": d["bbox"]["x1"],  "y1": d["bbox"]["y1"],
                "x2": d["bbox"]["x2"],  "y2": d["bbox"]["y2"],
                "cx": d["center"]["x"], "cy": d["center"]["y"],
            })
    print(f"  💾 CSV  → {path}")


# ──────────────────────────────────────────────────────────
#  6. POINT D'ENTRÉE
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extrait les références composants d'une image PCB")
    parser.add_argument("--img",   required=True, help="Image PCB (.png / .jpg)")
    parser.add_argument("--model", required=True, help="Modèle YOLO (best.pt)")
    parser.add_argument("--conf",  type=float, default=0.3,
                        help="Seuil de confiance YOLO (défaut : 0.3)")
    parser.add_argument("--show",  action="store_true",
                        help="Afficher l'image annotée")
    args = parser.parse_args()

    img_path   = Path(args.img)
    model_path = Path(args.model)

    if not img_path.exists():
        print(f"❌ Image introuvable : {args.img}"); return
    if not model_path.exists():
        print(f"❌ Modèle introuvable : {args.model}"); return

    print(f"\n{'═'*50}")
    print(f"  PCB Component Reference Extractor  v3")
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
    img_out   = out_dir / f"{stem}_{ts}_annotated.png"
    cv2.imwrite(str(img_out), annotated)
    print(f"\n  🖼️  Image annotée → {img_out}")

    save_json(detections, out_dir / f"{stem}_{ts}_results.json")
    save_csv (detections, out_dir / f"{stem}_{ts}_results.csv")

    if args.show:
        cv2.imshow("PCB Extractor v3", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"\n✅ Terminé ! Résultats dans : {out_dir}/\n")


if __name__ == "__main__":
    main()
