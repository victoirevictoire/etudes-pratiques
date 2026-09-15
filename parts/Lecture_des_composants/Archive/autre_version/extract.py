"""
extract.py
==========
Pipeline d'extraction de références composants PCB.
Utilise un modèle YOLO pré-entraîné (fourni) pour la détection,
puis Tesseract pour lire le texte dans chaque zone détectée.

Fixes v2 :
  - Upscale ×2 de l'image entière avant YOLO (texte ~15px trop petit)
  - Upscale ×5 des crops (au lieu de ×3)
  - Espace ajouté dans la whitelist Tesseract (refs "R 516", "C 407")
  - Regex mis à jour pour accepter "R 516" avec espace

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

def preprocess_crop(crop_bgr, scale=5):
    """
    Améliore la lisibilité du crop avant OCR :
      - Upscale ×5  → les refs PCB sont très petites (~15px)
      - CLAHE       → améliore le contraste local
      - Débruitage  → réduit le bruit de compression
      - Binarisation adaptative → sépare proprement texte / fond
      - Padding     → évite que Tesseract rate les bords
    """
    h, w = crop_bgr.shape[:2]

    # Upscale ×5
    resized = cv2.resize(crop_bgr, (w * scale, h * scale),
                         interpolation=cv2.INTER_CUBIC)

    # Niveaux de gris
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Amélioration du contraste local
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Débruitage léger
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    # Binarisation adaptative
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2
    )

    # Si le fond est sombre → inverser
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # Padding blanc autour
    padded = cv2.copyMakeBorder(binary, 15, 15, 15, 15,
                                 cv2.BORDER_CONSTANT, value=255)
    return padded


# ──────────────────────────────────────────────────────────
#  2. OCR — Lit le texte dans un crop préprocessé
# ──────────────────────────────────────────────────────────

# Configs Tesseract : du plus strict au plus permissif
# Espace inclus dans la whitelist car refs PCB = "R 516", "C 407"
TESSERACT_CONFIGS = [
    r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz-_. ',
    r'--oem 3 --psm 8',
    r'--oem 3 --psm 6',
]

# Regex : valide les références composants
# Gère avec espace : "R 516", "C 407", "IC 402"
# et sans espace   : "R516", "BC178", "S402", "St401"
COMPONENT_PATTERN = re.compile(
    r'^('
    r'[A-Z]{1,2}\s\d{3,5}[A-Z]?'   # R 516 / C 407 / IC 402A  (avec espace)
    r'|[A-Z]{1,3}\d{2,5}[A-Z]?'    # R516 / BC178 / S402       (sans espace)
    r'|St\s?\d{3}'                  # St 401
    r'|[A-Z]{1,3}\d+[-_]\d+'        # U1-3
    r')$',
    re.IGNORECASE
)


def run_ocr(crop_bgr):
    """
    Essaie plusieurs configs Tesseract sur le crop.
    Retourne (texte, est_une_référence_valide).
    """
    preprocessed = preprocess_crop(crop_bgr)
    pil_img = Image.fromarray(preprocessed)

    candidates = []
    for config in TESSERACT_CONFIGS:
        try:
            raw = pytesseract.image_to_string(pil_img, config=config)
            # Garder alphanum + -_. + espace, nettoyer les espaces multiples
            text = re.sub(r'[^\w\-_\.\s]', '', raw).strip().upper()
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                candidates.append(text)
        except Exception:
            continue

    if not candidates:
        return None, False

    # Priorité aux textes qui matchent le pattern référence
    for c in candidates:
        if COMPONENT_PATTERN.match(c):
            return c, True

    # Sinon → le candidat le plus long
    best = max(candidates, key=len)
    return best, False


def run_ocr_with_rotations(crop_bgr):
    """
    Essaie 4 orientations (0°, 90°, 180°, 270°).
    Le texte PCB peut être dans n'importe quelle direction.
    Retourne (texte, est_valide, angle).
    """
    best_text, best_valid, best_angle = None, False, 0

    for angle in [0, 90, 180, 270]:
        rotated = np.rot90(crop_bgr, k=angle // 90)
        text, is_valid = run_ocr(rotated)

        if not text:
            continue

        # Dès qu'on trouve une référence valide, on arrête
        if is_valid and not best_valid:
            best_text, best_valid, best_angle = text, True, angle
            break

        # Sinon garder le plus long
        if len(text) > len(best_text or ''):
            best_text, best_angle = text, angle

    return best_text, best_valid, best_angle


# ──────────────────────────────────────────────────────────
#  3. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────

def extract(img_path: Path, model_path: Path, conf: float = 0.3):
    """
    Étape 1 : Upscale ×2 de l'image source (texte PCB trop petit sinon).
    Étape 2 : YOLO détecte les bounding boxes de texte.
    Étape 3 : Crop + Tesseract sur chaque zone.
    Retourne (image_originale, liste_de_détections).
    """

    # ── Charger l'image ──
    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        raise FileNotFoundError(f"Image introuvable : {img_path}")

    # ── Upscale ×2 de l'image entière ──
    print(f"  📐 Image originale : {img_orig.shape[1]}×{img_orig.shape[0]} px")
    img = cv2.resize(img_orig,
                     (img_orig.shape[1] * 2, img_orig.shape[0] * 2),
                     interpolation=cv2.INTER_CUBIC)
    print(f"  📐 Image upscalée  : {img.shape[1]}×{img.shape[0]} px")

    # ── Charger le modèle YOLO ──
    print(f"  🤖 Modèle  : {model_path.name}")
    model = YOLO(str(model_path))

    # ── Détection YOLO ──
    yolo_results = model(img, conf=conf, imgsz=1024, verbose=False)[0]
    boxes = yolo_results.boxes
    print(f"  📦 {len(boxes)} zone(s) détectée(s)  (conf ≥ {conf})")

    # ── OCR sur chaque crop ──
    detections = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf_yolo = float(box.conf[0])

        # Marge autour de la boîte
        pad = 6
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(img.shape[1], x2 + pad)
        cy2 = min(img.shape[0], y2 + pad)

        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        text, is_valid, angle = run_ocr_with_rotations(crop)

        # Remettre les coordonnées à l'échelle originale (÷2)
        detections.append({
            "id":           i + 1,
            "text":         text or "",
            "is_valid_ref": is_valid,
            "yolo_conf":    round(conf_yolo, 3),
            "rotation":     angle,
            "bbox": {
                "x1": x1 // 2, "y1": y1 // 2,
                "x2": x2 // 2, "y2": y2 // 2,
            },
            "center": {
                "x": (x1 + x2) // 4,
                "y": (y1 + y2) // 4,
            },
        })

    return img_orig, detections


# ──────────────────────────────────────────────────────────
#  4. VISUALISATION
# ──────────────────────────────────────────────────────────

def draw_results(img, detections):
    """
    Dessine sur l'image :
      - Vert  = référence validée (R 516, C 407...)
      - Orange = texte détecté mais format non reconnu
    """
    output = img.copy()

    for det in detections:
        b = det["bbox"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        text     = det["text"]
        is_valid = det["is_valid_ref"]

        color = (0, 200, 0) if is_valid else (0, 140, 255)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        if text:
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness  = 1
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(output,
                          (x1, y1 - th - 6), (x1 + tw + 4, y1),
                          color, -1)
            cv2.putText(output, text,
                        (x1 + 2, y1 - 3),
                        font, font_scale, (255, 255, 255), thickness)

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
    print(f"  PCB Component Reference Extractor  v2")
    print(f"{'═'*50}")

    img, detections = extract(img_path, model_path, conf=args.conf)

    total  = len(detections)
    valid  = [d for d in detections if d["is_valid_ref"]]
    others = [d for d in detections if d["text"] and not d["is_valid_ref"]]

    print(f"\n{'─'*50}")
    print(f"  Zones détectées   : {total}")
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
        cv2.imshow("PCB Extractor v2 — appuie sur une touche pour quitter", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"\n✅ Terminé ! Résultats dans : {out_dir}/\n")


if __name__ == "__main__":
    main()
