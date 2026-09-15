from pathlib import Path
from ultralytics import YOLO

# ======================
# CHEMINS
# ======================
BASE_DIR   = Path("/home/rayane/Git/etudes-pratiques")
DATASET = BASE_DIR / "data/datasets/nom_composants/data.yaml"
BASE_MODEL = BASE_DIR / "dataset/yolov8n.pt"  # ou chemin absolu si ailleurs
PROJECT = BASE_DIR / "models/component_names"
RUN_NAME   = "nom_composants_v4"

# ======================
# ENTRAÎNEMENT
# ======================
model = YOLO("yolov8n.pt")  # télécharge auto si absent

model.train(
    data=str(DATASET),
    project=str(PROJECT),
    name=RUN_NAME,
    device=0,              # GPU

    epochs=150,
    patience=30,
    imgsz=1024,
    batch=4,

    optimizer="SGD",
    lr0=1e-2,
    lrf=1e-2,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3,

    hsv_h=0.0,
    hsv_s=0.2,
    hsv_v=0.3,
    degrees=5,
    translate=0.1,
    scale=0.4,
    shear=2,
    perspective=0.0002,
    flipud=0.0,
    fliplr=0.3,
    mosaic=0.5,
    mixup=0.0,
    copy_paste=0.0,

    cache='disk',
    workers=4,
    seed=42,
    verbose=True,
    save=True,
    save_period=25,
)

best = PROJECT / RUN_NAME / "weights" / "best.pt"
print(f"\n✅ Terminé — meilleurs poids : {best}")
