"""
extract.py  v4
==============
Remplace Tesseract par EasyOCR :
  - Bien meilleur sur les polices techniques (PCB, schémas)
  - Gère nativement les rotations
  - Plus robuste sur les petits textes dégradés

Installation :
  pip install easyocr ultralytics opencv-python numpy

Usage:
  python3 extract.py --img carte.png --model best.pt
  python3 extract.py --img carte.png --model best.pt --conf 0.3 --show
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
#  1. INITIALISATION EASYOCR (une seule fois au démarrage)
# ──────────────────────────────────────────────────────────

print("⏳ Chargement EasyOCR (première fois = téléchargement du modèle)...")
READER = easyocr.Reader(
    ['en'],          # Anglais suffit pour les refs composants
    gpu=True,        # Utilise le GPU si dispo, sinon CPU auto
    verbose=False,
)
print("✅ EasyOCR prêt")


# ──────────────────────────────────────────────────────────
#  2. PREPROCESSING — Prépare chaque crop
# ──────────────────────────────────────────────────────────

def preprocess_crop(crop_bgr, scale=4):
    """
    Upscale + sharpen + CLAHE.
    EasyOCR est déjà bien meilleur que Tesseract sur les petits textes,
    donc on garde un preprocessing léger (scale=4 suffit).
    """
    h, w = crop_bgr.shape[:2]

    # Upscale
    resized = cv2.resize(crop_bgr, (w * scale, h * scale),
                         interpolation=cv2.INTER_CUBIC)

    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, kernel)

    # CLAHE sur niveaux de gris puis reconvertir en BGR
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Repasser en BGR pour EasyOCR (attend du BGR ou RGB)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # Padding
    padded = cv2.copyMakeBorder(enhanced_bgr, 10, 10, 10, 10,
                                 cv2.BORDER_CONSTANT,
                                 value=(255, 255, 255))
    return padded


# ──────────────────────────────────────────────────────────
#  3. OCR avec EasyOCR
# ──────────────────────────────────────────────────────────

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
    """
    Lance EasyOCR sur un crop préprocessé.
    Retourne (texte, est_une_référence_valide, confiance).
    """
    preprocessed = preprocess_crop(crop_bgr)

    try:
        # detail=1 → retourne aussi les scores de confiance
        results = READER.readtext(preprocessed, detail=1)
    except Exception as e:
        return None, False, 0.0

    if not results:
        return None, False, 0.0

    # Trier par confiance décroissante
    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)

    candidates = []
    for (bbox, text, conf) in results_sorted:
        # Nettoyage
        text_clean = re.sub(r'[^\w\-_\.\s]', '', text).strip().upper()
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()
        if text_clean:
            candidates.append((text_clean, conf))

    if not candidates:
        return None, False, 0.0

    # Priorité aux textes qui matchent le pattern référence
    for (text, conf) in candidates:
        if COMPONENT_PATTERN.match(text):
            return text, True, conf

    # Sinon → le candidat avec la meilleure confiance
    best_text, best_conf = candidates[0]
    return best_text, False, best_conf


# ──────────────────────────────────────────────────────────
#  4. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────

def extract(img_path: Path, model_path: Path, conf: float = 0.3):
    """
    Étape 1 : YOLO détecte les zones de texte.
    Étape 2 : EasyOCR lit le texte dans chaque crop.
    """

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image introuvable : {img_path}")
    print(f"  📐 Image : {img.shape[1]}×{img.shape[0]} px")

    print(f"  🤖 Modèle YOLO : {model_path.name}")
    model = YOLO(str(model_path))

    yolo_results = model(img, conf=conf, imgsz=1024, verbose=False)[0]
    boxes = yolo_results.boxes
    print(f"  📦 {len(boxes)} zone(s) détectée(s)  (conf ≥ {conf})")
    print(f"  ⏳ OCR EasyOCR en cours...")

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

        text, is_valid, ocr_conf = run_ocr(crop)

        detections.append({
            "id":           i + 1,
            "text":         text or "",
            "is_valid_ref": is_valid,
            "yolo_conf":    round(conf_yolo, 3),
            "ocr_conf":     round(ocr_conf, 3),
            "bbox":         {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "center":       {"x": (x1 + x2) // 2, "y": (y1 + y2) // 2},
        })

        # Progression tous les 50 crops
        if (i + 1) % 50 == 0:
            print(f"     {i+1}/{len(boxes)} crops traités...")

    return img, detections


# ──────────────────────────────────────────────────────────
#  5. VISUALISATION
# ──────────────────────────────────────────────────────────

def draw_results(img, detections):
    """
    Vert  = référence validée (R 516, C 407...)
    Orange = texte lu mais format non reconnu
    """
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
#  6. EXPORTS
# ──────────────────────────────────────────────────────────

def save_json(detections, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)
    print(f"  💾 JSON → {path}")


def save_csv(detections, path):
    fields = ["id", "text", "is_valid_ref", "yolo_conf", "ocr_conf",
              "x1", "y1", "x2", "y2", "cx", "cy"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in detections:
            w.writerow({
                "id":           d["id"],
                "text":         d["text"],
                "is_valid_ref": d["is_valid_ref"],
                "yolo_conf":    d["yolo_conf"],
                "ocr_conf":     d["ocr_conf"],
                "x1": d["bbox"]["x1"],  "y1": d["bbox"]["y1"],
                "x2": d["bbox"]["x2"],  "y2": d["bbox"]["y2"],
                "cx": d["center"]["x"], "cy": d["center"]["y"],
            })
    print(f"  💾 CSV  → {path}")


# ──────────────────────────────────────────────────────────
#  7. POINT D'ENTRÉE
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
    print(f"  PCB Component Reference Extractor  v4")
    print(f"  OCR : EasyOCR")
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
        cv2.imshow("PCB Extractor v4 — appuie sur une touche pour quitter", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"\n✅ Terminé ! Résultats dans : {out_dir}/\n")


if __name__ == "__main__":
    main()
