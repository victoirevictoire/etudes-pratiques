# 📁 Guide de contribution — Etudes Pratiques



## 📂 Arborescence — Où mettre quoi

```
etudes-pratiques/
│
├── data/
│   ├── datasets/
│   │   ├── dataset1/
│   │   │   ├── train/
│   │   │   ├── valid/
│   │   │   └── data.yaml
│   │   └── dataset2.../   ← un autre dataset, par exemple Dataset YOLO détection zones schéma
│   ├── docs_pdf/               ← PDFs sources originaux (ne pas modifier)
│   └── raw_images/             ← Images brutes non annotées
│
├── models/
│   ├── yolov8n.pt              ← Base model YOLO (téléchargé une fois)
│   ├── component_names/
│   └── les autres models
│
├── parts/                      ← Code de travail de chaque partie
│   ├── extraction_nom_composants/   ← Louis
│   │   ├── train.py
│   │   └── yolo_nms.py
│   ├── Extraction_schemas/          ← Équipe schéma
│   ├── victoire/                    ← Victoire (à modifier le nom)
│   ├── les autres à rajouter/                    ← Victoire (à modifier le nom)
│   ├── Lecture_table/       	← Ilyas
│   │	├── IMAGES/
│   │	└── test.py	
│   └── a_trier/                     ← Scripts divers à migrer
│
├── notebook/
│   └── NotebookEP.ipynb        ← Notebook principal, point d'entrée du pipeline
│
├── results/                    ← ⚠️ GITIGNORE — généré en local uniquement
│   ├── images/                 ← Images annotées après inférence
│   └── labels/                 ← Labels YOLO exportés (.txt)
│
├── CONTRIBUTING.md             ← Ce fichier
├── README.md
└── .gitignore
```

---

## 📋 Règles par dossier

### `data/datasets/`
- Les datasets viennent de **Roboflow** — ne jamais modifier les images/labels manuellement
- Chaque dataset a sa structure `train/`, `valid/`, `data.yaml`
- Si tu crées un nouveau dataset, le nommer explicitement (`nom_composants`, `schema_detection`...)

### `models/`
- Seuls `best.pt` et `last.pt` sont commités
- Les checkpoints intermédiaires (`epoch*.pt`) sont dans le `.gitignore`
- Les images de training (`train_batch*.jpg`, `val_batch*.jpg`) sont aussi ignorées
- Si les `.pt` sont trop lourds pour git → partager via Google Drive et mettre le lien dans le README

### `parts/`
- Chacun travaille dans un sous-dossier en fonction de sur quoi il travaille

### `results/`
- **Jamais commité** — chacun génère ses résultats en local
- Créer le dossier manuellement après le clone : `mkdir -p results/images results/labels`

### `notebook/`
- Le notebook orchestre toutes les étapes du pipeline
- Il importe les fonctions depuis `parts/`
- Les images/CSV de test nécessaires au notebook peuvent y être stockés

---

## 📝 Règles git

1. **`git pull` avant de travailler** — toujours synchroniser avant de commencer
2. **Un commit = une chose** — message clair avec préfixe :
   - `feat:` nouvelle fonctionnalité
   - `fix:` correction de bug
   - `data:` ajout/modification de dataset
   - `model:` nouveau modèle ou poids
   - `docs:` documentation
3. **Ne jamais commiter** : `results/`, `*.cache`, `epoch*.pt`, `__pycache__/`
4. **Toujours vérifier** avec `git status` avant `git commit`

---
