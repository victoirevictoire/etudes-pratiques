"""
compare_models.py
=================
Compare best.pt vs last.pt sur la même image.
Affiche nb zones détectées, distribution des confs, et exemples.

Usage:
  python3 compare_models.py --img carte.png
"""

import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO


def test_model(img, model_path, conf=0.4):
    model  = YOLO(str(model_path))
    result = model(img, conf=conf, imgsz=1024, verbose=False)[0]
    boxes  = result.boxes

    confs = [float(b.conf[0]) for b in boxes]
    if not confs:
        return {"n": 0, "conf_mean": 0, "conf_min": 0, "conf_max": 0, "boxes": []}

    return {
        "n":         len(confs),
        "conf_mean": sum(confs) / len(confs),
        "conf_min":  min(confs),
        "conf_max":  max(confs),
        "n_above_05": sum(1 for c in confs if c >= 0.5),
        "n_above_07": sum(1 for c in confs if c >= 0.7),
        "boxes": [(int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                   int(b.xyxy[0][2]), int(b.xyxy[0][3]),
                   float(b.conf[0])) for b in boxes],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",  required=True)
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    img_path = Path(args.img)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Image introuvable : {args.img}"); return

    base = Path(__file__).parent
    candidates = [
        ("best.pt (local)",  base / "best.pt"),
        ("last.pt (local)",  base / "last.pt"),
        ("best.pt (v4)",     base / "../../models/component_names/nom_composants_v4/weights/best.pt"),
        ("last.pt (v4)",     base / "../../models/component_names/nom_composants_v4/weights/last.pt"),
    ]

    print(f"\n{'='*60}")
    print(f"  Comparaison modeles — {img_path.name}  (conf>={args.conf})")
    print(f"{'='*60}")
    print(f"{'Modele':<22} {'Zones':>6} {'Moy':>6} {'Min':>6} {'Max':>6} {'>0.5':>5} {'>0.7':>5}")
    print(f"{'-'*60}")

    results = {}
    for label, path in candidates:
        path = path.resolve()
        if not path.exists():
            print(f"{label:<22}  [non trouve]")
            continue
        r = test_model(img, path, conf=args.conf)
        results[label] = r
        print(f"{label:<22} {r['n']:>6} {r['conf_mean']:>6.3f} "
              f"{r['conf_min']:>6.3f} {r['conf_max']:>6.3f} "
              f"{r.get('n_above_05',0):>5} {r.get('n_above_07',0):>5}")

    print(f"\n  Conseil : le meilleur modele est celui avec")
    print(f"  le plus de zones ET la confiance moyenne la plus haute.")
    print(f"  Si 'last.pt' a beaucoup plus de zones que 'best.pt',")
    print(f"  il sur-detecte (faux positifs). Preferez best.pt.\n")


if __name__ == "__main__":
    main()
