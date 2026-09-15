from pathlib import Path
from ultralytics import YOLO
import cv2

# ======================
# CHEMINS
# ======================
REPO_ROOT     = Path(__file__).resolve().parent.parent.parent  # .../etudes-pratiques
MODEL_PATH    = REPO_ROOT / "models/component_names/text_v3_clean/weights/best.pt"
VAL_IMAGES    = REPO_ROOT / "data/datasets/nom_composants/valid/images"
OUT_DIR       = REPO_ROOT / "results/images"
ALL_BOXES_DIR = REPO_ROOT / "results/labels"

# ======================
# SEUILS
# ======================
CONF_THRESHOLD        = 0.10
IOU_THRESHOLD         = 0.5
CONTAINMENT_THRESHOLD = 0.5

OUT_DIR.mkdir(parents=True, exist_ok=True)
ALL_BOXES_DIR.mkdir(parents=True, exist_ok=True)


# ======================
# FONCTIONS NMS
# ======================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA);  inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter_area / float(areaA + areaB - inter_area)


def containment_ratio(inner, outer):
    xA = max(inner[0], outer[0]); yA = max(inner[1], outer[1])
    xB = min(inner[2], outer[2]); yB = min(inner[3], outer[3])
    inter_w = max(0, xB - xA);   inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    inner_area = (inner[2]-inner[0]) * (inner[3]-inner[1])
    return inter_area / inner_area


def box_to_yolo(box, img_w, img_h, class_id=0):
    x1, y1, x2, y2 = box
    xc = ((x1+x2)/2) / img_w
    yc = ((y1+y2)/2) / img_h
    w  = (x2-x1) / img_w
    h  = (y2-y1) / img_h
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def nms(boxes, confs, iou_thresh=0.5, cont_thresh=0.5):
    keep    = []
    indices = sorted(range(len(boxes)), key=lambda i: confs[i], reverse=True)
    while indices:
        cur = indices.pop(0)
        keep.append(cur)
        indices = [
            i for i in indices
            if iou(boxes[cur], boxes[i]) < iou_thresh
            and containment_ratio(boxes[i], boxes[cur]) < cont_thresh
            and containment_ratio(boxes[cur], boxes[i]) < cont_thresh
        ]
    return keep


# ======================
# INFÉRENCE + NMS
# ======================
model = YOLO(str(MODEL_PATH))
IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif"}

total_before = total_after = 0

for img_path in sorted(VAL_IMAGES.iterdir()):
    if img_path.suffix.lower() not in IMG_EXTS:
        continue
    image = cv2.imread(str(img_path))
    if image is None:
        continue
    H, W = image.shape[:2]

    res   = model(image, conf=CONF_THRESHOLD)[0]
    boxes = [list(map(int, b.xyxy[0])) for b in res.boxes]
    confs = [float(b.conf[0])          for b in res.boxes]

    keep_idx = nms(boxes, confs, IOU_THRESHOLD, CONTAINMENT_THRESHOLD)

    with open(ALL_BOXES_DIR / f"{img_path.stem}.txt", "w") as f:
        f.write("\n".join(box_to_yolo(boxes[i], W, H) for i in keep_idx))

    for i in keep_idx:
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(image, f"{confs[i]:.2f}", (x1, max(y1-3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(OUT_DIR / img_path.name), image)

    b, a = len(boxes), len(keep_idx)
    total_before += b; total_after += a
    print(f"{img_path.name} → {b} avant / {a} après NMS")

print(f"\n✅ Total : {total_before} → {total_after} boîtes après NMS")
print(f"   Images : {OUT_DIR}")
print(f"   Labels : {ALL_BOXES_DIR}")
