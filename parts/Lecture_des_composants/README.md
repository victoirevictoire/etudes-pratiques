# PCB Component Reference Extractor

Extraction automatisée des références de composants électroniques sur des schémas PCB ou des cartes électroniques photographiées.

**Pipeline** : `YOLOv8` (détection de zones) → `Merge voisines` → `EasyOCR` (lecture OCR) → `Correction OCR` → `Export`

---

## Fonctionnement

1. **YOLO** localise les zones de texte contenant des références (ex. `R516`, `C401`, `BC178`).
2. **Merge** fusionne les boîtes voisines que YOLO a divisées (ex. `R` + `516` → boîte unique).
3. **EasyOCR** lit le texte de chaque zone (avec upscale et amélioration du contraste).
4. **Correction** applique un correcteur de confusions OCR (`I`↔`1`, `O`↔`0`, etc.) et un dictionnaire d'erreurs connues.
5. **Export** génère une image annotée, un fichier JSON et un CSV.

---

## Installation

```bash
pip install -r requirements.txt
```

> **GPU recommandé** — EasyOCR est nettement plus rapide avec CUDA. Sans GPU, le temps d'exécution peut dépasser 5 minutes.

---

## Utilisation

```bash
python3 extract_easyocr.py --img <image> --model <modele.pt> [--conf <seuil>]
```

### Arguments

| Argument | Obligatoire | Défaut | Description |
|---|---|---|---|
| `--img` | ✅ | — | Chemin vers l'image PCB (PNG, JPG) |
| `--model` | ✅ | — | Chemin vers le modèle YOLO (`best.pt`) |
| `--conf` | ❌ | auto | Seuil de confiance YOLO (0.0 → 1.0). Si absent, calculé automatiquement selon le contraste de l'image. |

### Exemples

```bash
# Lancement standard (paramètres auto-détectés)
python3 extract_easyocr.py --img carte.png --model best.pt

# Forcer un seuil de confiance spécifique
python3 extract_easyocr.py --img carte.png --model best.pt --conf 0.4

# Image faible résolution ou faible contraste
python3 extract_easyocr.py --img schema.png --model best.pt --conf 0.25

# Ouvrir le résultat visuel
xdg-open results/*_annotated.png
```

### Choisir le seuil `--conf`

| Situation | Valeur recommandée |
|---|---|
| Image standard (> 2 MP, bon contraste) | `0.40` (défaut auto) |
| Image petite / faible contraste | `0.25` |
| Trop de faux positifs dans les résultats | `0.50` |
| Trop peu de refs trouvées | `0.25` |

---

## Sortie terminal

```
  ──────────────────────────────────────────────────────
    PCB Component Reference Extractor  v7
  ──────────────────────────────────────────────────────

  ◈  IMAGE   2480×1200 px  ~200 DPI  ● BONNE
  ◈  PARAMS  imgsz=1280  scale=5  conf=0.40

  ◉  YOLO  185 zones  →  155 apres merge (+30 fusions)  1.2s

  ⟳  OCR  [████████████████████████████░░]  155/155  100%

  RÉSULTATS
  ──────────────────────────────────────────────────────
  Zones traitees        149
  Refs valides           61
    dont haute conf      26  (ocr >= 0.7 → fiables)
  Non reconnu            88
  Temps total          42.3s
  ──────────────────────────────────────────────────────

  RÉFÉRENCES TROUVÉES (61 — triées par score)

  Référence              YOLO      OCR    Score  Qualité
  ──────────────────────────────────────────────────────
  R516                   0.73     0.96     0.70  ●●●
  C410                   0.76     0.89     0.67  ●●●
  R415                   0.62     1.00     0.61  ●●●
  ...
```

**Indicateur qualité :**
- `●●●` vert   → score ≥ 0.5 — ref très fiable
- `●●○` jaune  → score 0.3–0.5 — à vérifier
- `●○○` rouge  → score < 0.3 — lecture incertaine

---

## Fichiers générés

```
results/
├── carte_YYYYMMDD_HHMMSS_annotated.png   ← image avec boîtes colorées
├── carte_YYYYMMDD_HHMMSS_results.json    ← toutes les détections
└── carte_YYYYMMDD_HHMMSS_results.csv     ← tableau (Excel / pandas)
```

**Couleurs sur l'image annotée :**
- Vert   — référence valide (`R516`, `C401`, `BC178`...)
- Orange — zone détectée mais texte non reconnu ou invalide

**Structure d'une entrée JSON :**
```json
{
  "id": 1,
  "text": "R516",
  "is_valid_ref": true,
  "yolo_conf": 0.730,
  "ocr_conf":  0.960,
  "score":     0.701,
  "bbox":   { "x1": 748, "y1": 757, "x2": 814, "y2": 796 },
  "center": { "x": 781, "y": 776 }
}
```

---

## Adaptation automatique à l'image

Le script analyse l'image avant traitement et adapte ses paramètres sans intervention manuelle :

| Caractéristique détectée | Action automatique |
|---|---|
| Grande image (> 3 MP) | `imgsz=1280`, `scale=5` |
| Image moyenne (1–3 MP) | `imgsz=1024`, `scale=6` |
| Petite image (< 0.3 MP) | `imgsz=640`, `scale=10` |
| DPI estimé < 80 | Pré-upscale ×3 avant YOLO |
| DPI estimé < 150 | Pré-upscale ×2 avant YOLO |
| Faible contraste (`std < 25`) | `conf=0.25` |
| Fond sombre (texte clair) | Inversion couleurs automatique |
| Crop vertical | Rotation 90° automatique |

Si les résultats sont insuffisants, forcer manuellement via `--conf`.

---

## Ajouter une correction personnalisée

Quand une référence est systématiquement mal lue, ouvrir `extract_easyocr.py`
et ajouter une entrée dans le dictionnaire `POST_CORRECTIONS` :

```python
POST_CORRECTIONS = {
    'UC402':  'IC402',   # U mal lu à la place de I
    'BCI78':  'BC178',   # I confondu avec 1
    # Ajouter ici :
    'MAUVAIS': 'CORRECT',
}
```

---

## Structure du dossier

```
Lecture_des_composants/
├── extract_easyocr.py   ← script principal (v7)
├── best.pt              ← modèle YOLO entraîné (meilleur checkpoint)
├── carte.png            ← image de test principale
├── schema.png           ← image de test secondaire
├── requirements.txt     ← dépendances Python
├── results/             ← résultats générés automatiquement
└── Archive/             ← versions antérieures du pipeline
```

---

## Performances

| Métrique | Valeur |
|---|---|
| Zones YOLO détectées | ~185 (conf=0.4) |
| Fusions merge voisines | ~30 |
| Refs valides trouvées | ~61 |
| Refs haute confiance (fiables) | ~26 |
| Recall estimé | ~40 % |
| Temps d'exécution | ~40–60 s (GPU) |

> **Levier principal d'amélioration** : réentraîner le modèle YOLO avec des boîtes d'annotation serrées autour du texte uniquement. Objectif : passer de 40 % à 60–70 % de recall.
