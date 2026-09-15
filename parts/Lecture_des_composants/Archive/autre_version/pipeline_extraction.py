"""
pipeline_extraction.py
======================
Relie prediction.py (classification pages) + extract_easyocr.py (cartes)
+ extract_schema_tiles.py (schémas) pour produire un CSV unifié de coordonnées.

Étapes :
  1. PDF  →  images PNG à 300 DPI  (prediction.py)
  2. Chaque image  →  classification YOLO : carte / schema / ignorer
  3. Carte  →  YOLO + EasyOCR  →  coords  (extract_easyocr.py)
  4. Schéma →  Tesseract tuiles →  coords  (extract_schema_tiles.py)
  5. Fusion dans un CSV unifié  :  ref, source, type, cx, cy, x1, y1, x2, y2

Usage :
  python3 pipeline_extraction.py --pdf doc.pdf
  python3 pipeline_extraction.py --pdf doc.pdf --classif-model models/extraction/best.pt
  python3 pipeline_extraction.py --images carte.png schema.png --types carte schema
"""

import argparse
import csv
import re
import sys
import time
import cv2
import numpy as np
import pytesseract
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ─── Chemins des modèles (modifiables) ───────────────────────────────────────
DEFAULT_COMPOSANTS_MODEL = Path("parts/Lecture_des_composants/best.pt")
DEFAULT_CLASSIF_MODEL    = Path("models/extraction/best.pt")   # peut ne pas exister

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITAIRES COMMUNS
# ─────────────────────────────────────────────────────────────────────────────

def log(msg): print(f"  {msg}", flush=True)
def ok(msg):  print(f"  ✔  {msg}", flush=True)
def warn(msg):print(f"  ⚠  {msg}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ÉTAPE 1 — PDF → PAGES PNG (prediction.py)
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_pages(pdf_path: Path, out_dir: Path, dpi: int = 300) -> list[Path]:
    """Convertit un PDF en images PNG à {dpi} DPI. Retourne la liste des PNG."""
    from pdf2image import convert_from_path

    log(f"Conversion PDF → images ({dpi} DPI) : {pdf_path.name}")
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    paths = []
    for i, page in enumerate(pages):
        p = out_dir / f"page_{i+1:03d}.png"
        page.save(str(p))
        paths.append(p)
    ok(f"{len(paths)} page(s) extraite(s)")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
#  ÉTAPE 2 — CLASSIFICATION DES PAGES (prediction.py)
# ─────────────────────────────────────────────────────────────────────────────

def classify_pages(image_paths: list[Path],
                   classif_model_path: Path) -> dict[Path, str]:
    """
    Classifie chaque image en 'carte', 'schema' ou 'ignorer'.
    Si le modèle n'existe pas → retourne {} (classification manuelle requise).
    """
    if not classif_model_path.exists():
        warn(f"Modèle classification introuvable : {classif_model_path}")
        warn("→ Utilisez --types pour spécifier le type de chaque image manuellement.")
        return {}

    from ultralytics import YOLO
    log(f"Chargement modèle classification : {classif_model_path}")
    model = YOLO(str(classif_model_path))

    results = {}
    for img_path in image_paths:
        import numpy as np
        img = cv2.imread(str(img_path))
        if img is None:
            warn(f"Image illisible : {img_path}")
            results[img_path] = 'ignorer'
            continue

        pred = model.predict(source=img, conf=0.212, verbose=False)
        classes = [pred[0].names[int(b.cls[0])] for b in pred[0].boxes]

        # Priorité : carte > schema > ignorer
        cls = 'ignorer'
        for candidate in ['carte', 'schema']:
            if candidate in classes:
                cls = candidate
                break
        if cls == 'ignorer' and classes:
            cls = classes[0]

        results[img_path] = cls
        log(f"  {img_path.name}  →  {cls}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  ÉTAPE 3 — EXTRACTION CARTE (extract_easyocr.py)
# ─────────────────────────────────────────────────────────────────────────────

# — Correcteur OCR (extract_easyocr.py) —
PREFIX_FIXES = {'0':'O','1':'I','8':'B','6':'G','5':'S'}
DIGIT_FIXES  = {'I':'1','J':'1','L':'1','|':'1','O':'0','Q':'0','D':'0',
                'G':'6','Z':'2','S':'5','B':'8','T':'7'}
VALID_2L     = {'IC','TR','VR','BC','ZD','SW','TP','FR','ST','CF',
                'FL','RL','DL','LED','SCR','FET','MOV'}
POST_CORRECTIONS = {
    'UC402':'IC402','UC405':'IC405','RSI9':'R519','RL26':'R126',
    'R62I':'R621','C41J':'C411','C50G':'C506',
}
COMPONENT_PATTERN = re.compile(
    r'^([A-Z]{1,2}\s?\d{3,5}[A-Z]?|[A-Z]{1,3}\d{2,5}[A-Z]?'
    r'|[A-Z]{2,4}\d{1,4}[A-Z]?|St\s?\d{3}|[A-Z]{1,3}\d+[-_]\d+)$',
    re.IGNORECASE
)

def fix_ocr_confusion(text: str) -> str:
    if not text: return text
    text = text.strip().upper().replace(' ','')
    i = 0
    while i < len(text) and not text[i].isdigit(): i += 1
    if i == len(text): return text
    prefix_raw, rest = text[:i], text[i:]
    if (len(prefix_raw) >= 2 and prefix_raw[-1] == 'I'
            and prefix_raw not in VALID_2L):
        prefix_raw, rest = prefix_raw[:-1], '1' + rest
    if len(rest) > 1 and rest[-1].isalpha() and rest[-1] not in DIGIT_FIXES:
        suffix, digits_raw = rest[-1], rest[:-1]
    else:
        suffix, digits_raw = '', rest
    fixed_prefix = ''.join(PREFIX_FIXES.get(c,c) if c.isdigit() else c for c in prefix_raw)
    fixed_digits = ''.join(c if c.isdigit() else DIGIT_FIXES.get(c,c) for c in digits_raw)
    return fixed_prefix + fixed_digits + suffix

def apply_post_corrections(t: str) -> str:
    return POST_CORRECTIONS.get(t.strip().upper().replace(' ',''), t)

def is_valid_ref(t: str) -> bool:
    return bool(COMPONENT_PATTERN.match(t.strip()))

def merge_nearby_boxes(boxes, gap_ratio=0.8, overlap_y_ratio=0.4):
    if not boxes: return boxes
    boxes = sorted(boxes, key=lambda b: b[0])
    merged, changed = list(boxes), True
    while changed:
        changed, new_merged, used = False, [], [False]*len(merged)
        for i in range(len(merged)):
            if used[i]: continue
            b1 = merged[i]; x1a,y1a,x2a,y2a,ca = b1
            best_j, best_gap = -1, float('inf')
            for j in range(i+1, len(merged)):
                if used[j]: continue
                b2 = merged[j]; x1b,y1b,x2b,y2b,cb = b2
                if x1b < x2a: continue
                h_avg = ((y2a-y1a)+(y2b-y1b))/2; gap = x1b-x2a
                if gap > gap_ratio*h_avg: continue
                overlap_y = min(y2a,y2b)-max(y1a,y1b); min_h = min(y2a-y1a,y2b-y1b)
                if min_h == 0 or overlap_y/min_h < overlap_y_ratio: continue
                if gap < best_gap: best_gap, best_j = gap, j
            if best_j >= 0:
                b2 = merged[best_j]; x1b,y1b,x2b,y2b,cb = b2
                new_merged.append((min(x1a,x1b),min(y1a,y1b),max(x2a,x2b),max(y2a,y2b),(ca+cb)/2))
                used[i] = used[best_j] = True; changed = True
            else:
                new_merged.append(b1); used[i] = True
        merged = new_merged
    return merged

def preprocess_crop(crop_bgr, scale=5):
    h,w = crop_bgr.shape[:2]
    big = cv2.resize(crop_bgr,(w*scale,h*scale),interpolation=cv2.INTER_CUBIC)
    sharp = cv2.filter2D(big,-1,np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
    gray = cv2.cvtColor(sharp,cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    bgr = cv2.cvtColor(enhanced,cv2.COLOR_GRAY2BGR)
    return cv2.copyMakeBorder(bgr,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))

def upscale_image(img, factor):
    h,w = img.shape[:2]
    big = cv2.resize(img,(w*factor,h*factor),interpolation=cv2.INTER_CUBIC)
    return cv2.filter2D(big,-1,np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))

def auto_params(img):
    h,w = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mean,std = float(gray.mean()),float(gray.std())
    dpi_est = int(w/(297/25.4))
    pre_upscale = 3 if dpi_est<80 else (2 if dpi_est<150 else 1)
    eff_px = (w*pre_upscale)*(h*pre_upscale)
    if eff_px > 3_000_000: imgsz,scale = 1280,5
    elif eff_px > 1_000_000: imgsz,scale = 1024,6
    elif eff_px > 300_000: imgsz,scale = 640,8
    else: imgsz,scale = 640,10
    conf = 0.25 if std<25 else (0.30 if std<45 else 0.40)
    log(f"Carte : {w}×{h} px  ~{dpi_est} DPI  upscale×{pre_upscale}  imgsz={imgsz}")
    return {"imgsz":imgsz,"scale":scale,"conf":conf,"invert":mean<80,"pre_upscale":pre_upscale}

def extract_carte(img_path: Path, model_path: Path,
                  reader, conf_thresh=None) -> list[dict]:
    """
    Pipeline complet carte PCB :  YOLO → merge → EasyOCR → correction.
    Retourne liste de {ref, cx, cy, x1, y1, x2, y2, yolo_conf, ocr_conf}.
    """
    from ultralytics import YOLO

    img = cv2.imread(str(img_path))
    if img is None:
        warn(f"Image illisible : {img_path}"); return []

    params = auto_params(img)
    if conf_thresh is not None: params["conf"] = conf_thresh
    if params["invert"]: img = cv2.bitwise_not(img)
    if params["pre_upscale"] > 1:
        img = upscale_image(img, params["pre_upscale"])

    model = YOLO(str(model_path))
    yolo_result = model(img, conf=params["conf"],
                        imgsz=params["imgsz"], verbose=False)[0]
    boxes_raw = [(int(b.xyxy[0][0]),int(b.xyxy[0][1]),
                  int(b.xyxy[0][2]),int(b.xyxy[0][3]),float(b.conf[0]))
                 for b in yolo_result.boxes]
    boxes = merge_nearby_boxes(boxes_raw)
    log(f"  YOLO : {len(boxes_raw)} zones → {len(boxes)} après merge")

    ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
    PAD = 4
    detections = []
    for i, (x1,y1,x2,y2,conf_y) in enumerate(boxes):
        crop = img[max(0,y1-PAD):min(img.shape[0],y2+PAD),
                   max(0,x1-PAD):min(img.shape[1],x2+PAD)]
        if crop.size == 0: continue

        # OCR — tentative CLAHE puis Otsu
        text, valid, ocr_conf = "", False, 0.0
        for pre_fn in [preprocess_crop,
                       lambda c,scale=5: preprocess_crop(c,scale)]:
            try:
                hits = reader.readtext(pre_fn(crop, params["scale"]),
                                       detail=1, paragraph=False,
                                       allowlist=ALLOWLIST, width_ths=0.7)
            except Exception:
                continue
            hits = [(t,c) for (_,t,c) in hits if c >= 0.15]
            hits.sort(key=lambda x: x[1], reverse=True)
            for (t, c) in hits:
                t2 = re.sub(r'\s+','',t.upper())
                fixed = apply_post_corrections(fix_ocr_confusion(t2))
                if is_valid_ref(fixed):
                    text, valid, ocr_conf = fixed, True, c
                    break
            if valid: break
            if hits and not text:
                t2 = re.sub(r'\s+','',hits[0][0].upper())
                text = apply_post_corrections(fix_ocr_confusion(t2))
                ocr_conf = hits[0][1]

        if not text: continue
        cx, cy = (x1+x2)//2, (y1+y2)//2
        # Retour aux coords dans l'image ORIGINALE (avant upscale)
        scale_back = 1.0 / params["pre_upscale"]
        detections.append({
            "ref":       text,
            "is_valid":  valid,
            "cx":        int(cx * scale_back),
            "cy":        int(cy * scale_back),
            "x1":        int(x1 * scale_back),
            "y1":        int(y1 * scale_back),
            "x2":        int(x2 * scale_back),
            "y2":        int(y2 * scale_back),
            "yolo_conf": round(conf_y, 3),
            "ocr_conf":  round(ocr_conf, 3),
        })
        sys.stdout.write(f"\r  OCR {i+1}/{len(boxes)}")
        sys.stdout.flush()

    sys.stdout.write("\n")

    # Déduplication
    seen = {}
    for d in detections:
        if not d["is_valid"]: continue
        key = d["ref"].upper()
        score = d["yolo_conf"] * d["ocr_conf"]
        if key not in seen or score > seen[key]["yolo_conf"]*seen[key]["ocr_conf"]:
            seen[key] = d
    valids = list(seen.values())
    invalids = [d for d in detections if not d["is_valid"]]
    result = valids + invalids
    ok(f"Carte : {len(valids)} refs valides, {len(invalids)} non reconnues")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  ÉTAPE 4 — EXTRACTION SCHÉMA (extract_schema_tiles.py)
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_PATTERN = re.compile(
    r'^(IC\d{1,4}[A-Z]?|TR\d{1,4}[A-Z]?|BC\d{2,4}[A-Z]?'
    r'|[RCLDJ]\d{1,4}[A-Z]?|Q\d{1,3}[A-Z]?'
    r'|VR\d{1,3}|SW\d{1,3}|TP\d{1,3}|ZD\d{1,3})$',
    re.IGNORECASE
)
SCHEMA_OCR_FIX = {'O':'0','I':'1','l':'1','|':'1','G':'6','S':'5','Z':'2','B':'8'}
SCHEMA_STOP = {'GND','VCC','VDD','VSS','IN','OUT','COM','NC','AMP','REF',
               'LEFT','RIGHT','INPUT','OUTPUT','CHANNEL','REC','PLAY','TAPE'}

TESS_PSM11 = ('--oem 3 --psm 11 -c tessedit_char_whitelist='
              'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
TESS_PSM6  = ('--oem 3 --psm 6  -c tessedit_char_whitelist='
              'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')


def clean_schema_ref(word: str):
    t = word.strip().upper().replace(' ','')
    if not t or len(t) < 2 or t in SCHEMA_STOP: return None
    if not SCHEMA_PATTERN.match(t): return None
    i = 0
    while i < len(t) and not t[i].isdigit(): i += 1
    if i == len(t): return None
    prefix, numpart = t[:i], t[i:]
    fixed = ''.join(SCHEMA_OCR_FIX.get(c,c) for c in numpart)
    digits = ''.join(c for c in fixed if c.isdigit())
    suffix = fixed[-1] if fixed and fixed[-1].isalpha() and digits else ''
    if not digits or int(digits) == 0: return None
    if len(digits) > 3 and prefix in ('Q','L','D','TR'): return None
    return prefix + str(int(digits)) + suffix


def filter_tile_blobs(tile_bw, char_min, char_max):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tile_bw, 8)
    mask = np.zeros_like(tile_bw)
    for lbl in range(1, num_labels):
        x,y,w,h,area = stats[lbl]
        if char_min <= h <= char_max and w <= char_max*5 and area <= char_max*char_max*3:
            mask[labels == lbl] = 255
    return cv2.dilate(mask, np.ones((2,2),np.uint8), iterations=1)


def extract_schema(img_path: Path, scale: int = 8,
                   tile_size: int = 600, overlap: int = 150,
                   conf_min: int = 10) -> list[dict]:
    """
    Pipeline schéma : upscale + CLAHE + filtre blobs + OCR tuiles (PSM 11 + PSM 6).
    Retourne liste de {ref, cx, cy, x1, y1, x2, y2, ocr_conf}.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        warn(f"Image illisible : {img_path}"); return []

    h, w = img.shape[:2]
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray0.mean() < 80: img = cv2.bitwise_not(img)

    log(f"  Schéma : {w}×{h} px  scale×{scale}")
    big = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    sharp = cv2.filter2D(big, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
    _, bw = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    char_min, char_max = max(8, scale*3), max(80, scale*10)
    H, W = bw.shape
    step  = tile_size - overlap
    tiles_x = max(1, (W - overlap + step - 1) // step)
    tiles_y = max(1, (H - overlap + step - 1) // step)
    total = tiles_x * tiles_y
    log(f"  Tuiles : {tiles_x}×{tiles_y} = {total}")

    raw = []
    done = 0
    for row in range(tiles_y):
        for col in range(tiles_x):
            x0, y0 = col*step, row*step
            x1t, y1t = min(x0+tile_size, W), min(y0+tile_size, H)
            tile = bw[y0:y1t, x0:x1t]
            if tile.size == 0: done += 1; continue

            tile_clean = filter_tile_blobs(tile, char_min, char_max)
            tile_ocr   = cv2.copyMakeBorder(cv2.bitwise_not(tile_clean),
                                            20,20,20,20,cv2.BORDER_CONSTANT,value=255)

            # Deux passes Tesseract
            all_words = []
            for cfg in [TESS_PSM11, TESS_PSM6]:
                try:
                    d = pytesseract.image_to_data(tile_ocr, config=cfg,
                                                  output_type=pytesseract.Output.DICT)
                    for k, word in enumerate(d['text']):
                        word = word.strip()
                        if not word: continue
                        cf = int(d['conf'][k])
                        if cf < conf_min: continue
                        all_words.append((word, cf,
                                          d['left'][k], d['top'][k],
                                          d['width'][k], d['height'][k]))
                except Exception:
                    pass

            for (word, cf, lx, ly, lw, lh) in all_words:
                ref = clean_schema_ref(word)
                if ref is None: continue
                wx = x0 + lx + lw//2 - 20
                wy = y0 + ly + lh//2 - 20
                raw.append({
                    "ref":      ref,
                    "cx":       int(wx/scale),
                    "cy":       int(wy/scale),
                    "x1":       int((x0+lx-20)/scale),
                    "y1":       int((y0+ly-20)/scale),
                    "x2":       int((x0+lx+lw-20)/scale),
                    "y2":       int((y0+ly+lh-20)/scale),
                    "ocr_conf": cf/100.0,
                })

            done += 1
            sys.stdout.write(f"\r  OCR tuile {done}/{total}")
            sys.stdout.flush()

    sys.stdout.write("\n")

    # Déduplication spatiale par ref
    by_ref = defaultdict(list)
    for d in raw: by_ref[d["ref"].upper()].append(d)
    result = []
    for ref_key, group in by_ref.items():
        group.sort(key=lambda x: x["ocr_conf"], reverse=True)
        kept = []
        for det in group:
            if not any(abs(det["cx"]-k["cx"])<30 and abs(det["cy"]-k["cy"])<30
                       for k in kept):
                kept.append(det)
        result.extend(kept)

    ok(f"Schéma : {len(result)} refs trouvées")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  ÉTAPE 5 — CSV UNIFIÉ DES COORDONNÉES
# ─────────────────────────────────────────────────────────────────────────────

def save_unified_csv(all_detections: list[dict], out_path: Path):
    """
    Sauvegarde toutes les détections dans un CSV unifié.
    Colonnes : ref, type, source_image, cx, cy, x1, y1, x2, y2, conf
    """
    fields = ["ref","type","source_image","cx","cy","x1","y1","x2","y2","conf"]
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_detections)
    ok(f"CSV unifié : {out_path}  ({len(all_detections)} lignes)")


def save_coord_csv(detections: list[dict], out_path: Path):
    """
    Format (identifiant, x, y) compatible avec le pipeline Victoire
    (transformer_csv / fusionner_csv).
    """
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=["name","left","top","width","height"])
        w.writeheader()
        for d in detections:
            if not d.get("is_valid", True): continue
            w.writerow({"name":  d["ref"],
                        "left":  d["x1"], "top":   d["y1"],
                        "width": d["x2"]-d["x1"], "height": d["y2"]-d["y1"]})
    ok(f"CSV coords (format Victoire) : {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline extraction complète : PDF → coordonnées composants"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf",    type=Path, help="Fichier PDF source")
    src.add_argument("--images", type=Path, nargs="+",
                     help="Images PNG/JPG directement (une ou plusieurs)")

    parser.add_argument("--types", nargs="+",
                        choices=["carte","schema","ignorer"],
                        help="Type de chaque image (même ordre que --images). "
                             "Obligatoire si pas de modèle classification.")
    parser.add_argument("--classif-model", type=Path,
                        default=DEFAULT_CLASSIF_MODEL,
                        help=f"Modèle YOLO classification (défaut: {DEFAULT_CLASSIF_MODEL})")
    parser.add_argument("--composants-model", type=Path,
                        default=DEFAULT_COMPOSANTS_MODEL,
                        help=f"Modèle YOLO composants carte (défaut: {DEFAULT_COMPOSANTS_MODEL})")
    parser.add_argument("--dpi",        type=int,   default=300)
    parser.add_argument("--schema-scale",type=int,  default=8,
                        help="Facteur upscale pour OCR schémas (défaut: 8)")
    parser.add_argument("--out",        type=Path,  default=None,
                        help="Répertoire de sortie (défaut: results/pipeline_YYYYMMDD/)")
    args = parser.parse_args()

    # — Répertoire de sortie —
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or Path("results") / f"pipeline_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {'─'*56}")
    print(f"  Pipeline extraction — {ts}")
    print(f"  Sortie : {out_dir}")
    print(f"  {'─'*56}")

    # — EasyOCR (chargé une seule fois) —
    import easyocr
    log("Chargement EasyOCR...")
    reader = easyocr.Reader(['en','fr','de','es','it'], gpu=True, verbose=False)
    ok("EasyOCR prêt")

    # — Étape 1 : obtenir les images —
    if args.pdf:
        image_paths = pdf_to_pages(args.pdf, out_dir, dpi=args.dpi)
    else:
        image_paths = list(args.images)

    # — Étape 2 : classification —
    if args.types:
        if len(args.types) != len(image_paths):
            parser.error(f"--types doit avoir autant de valeurs que d'images "
                         f"({len(image_paths)} image(s))")
        classifs = {p: t for p, t in zip(image_paths, args.types)}
    else:
        classifs = classify_pages(image_paths, args.classif_model)
        if not classifs:
            # Aucun modèle → classification manuelle interactive
            print("\n  Classification manuelle (entrez 'carte', 'schema' ou 'ignorer') :")
            for p in image_paths:
                val = input(f"    {p.name} ? ").strip().lower()
                classifs[p] = val if val in ('carte','schema') else 'ignorer'

    # — Étapes 3 & 4 : extraction —
    all_dets = []

    for img_path, img_type in classifs.items():
        img_path = Path(img_path)
        print(f"\n  {'─'*56}")
        log(f"{img_path.name}  [{img_type}]")

        if img_type == 'ignorer':
            log("→ ignoré")
            continue

        if img_type == 'carte':
            dets = extract_carte(img_path, args.composants_model, reader)
            # CSV format Victoire pour cette image
            coord_csv = out_dir / f"{img_path.stem}_carte_coords.csv"
            save_coord_csv(dets, coord_csv)
            for d in dets:
                all_dets.append({
                    "ref":          d["ref"],
                    "type":         "carte",
                    "source_image": img_path.name,
                    "cx":  d["cx"], "cy":  d["cy"],
                    "x1":  d["x1"], "y1":  d["y1"],
                    "x2":  d["x2"], "y2":  d["y2"],
                    "conf": round(d.get("yolo_conf",0)*d.get("ocr_conf",1), 3),
                })

        elif img_type == 'schema':
            dets = extract_schema(img_path, scale=args.schema_scale)
            coord_csv = out_dir / f"{img_path.stem}_schema_coords.csv"
            save_coord_csv(dets, coord_csv)
            for d in dets:
                all_dets.append({
                    "ref":          d["ref"],
                    "type":         "schema",
                    "source_image": img_path.name,
                    "cx":  d["cx"], "cy":  d["cy"],
                    "x1":  d["x1"], "y1":  d["y1"],
                    "x2":  d["x2"], "y2":  d["y2"],
                    "conf": round(d.get("ocr_conf",0), 3),
                })

    # — Étape 5 : CSV unifié —
    print(f"\n  {'─'*56}")
    unified_csv = out_dir / "coordonnees_unifiees.csv"
    save_unified_csv(all_dets, unified_csv)

    # Résumé
    by_type = defaultdict(list)
    for d in all_dets: by_type[d["type"]].append(d["ref"])
    print(f"\n  {'─'*56}")
    print(f"  RÉSUMÉ")
    print(f"  {'─'*56}")
    for t, refs in by_type.items():
        valids = sorted(set(refs))
        print(f"  {t:8s} : {len(valids)} refs uniques")
        print(f"           {', '.join(valids[:20])}{'...' if len(valids)>20 else ''}")
    print(f"\n  Fichiers générés dans : {out_dir}/")
    print(f"  {'─'*56}\n")


if __name__ == "__main__":
    main()
