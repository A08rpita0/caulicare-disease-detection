# Reproducing `models/model1.h5`

1. Download the dataset (public, no login needed) and extract it so that
   `data/raw/Original Dataset/<class folders>` exists:

   ```bash
   curl -L -o data/raw/cauliflower-dataset.zip https://www.kaggle.com/api/v1/datasets/download/noamaanabdulazeem/cauliflower-dataset
   python -c "import zipfile; zipfile.ZipFile('data/raw/cauliflower-dataset.zip').extractall('data/raw')"
   ```

2. Install: `pip install -r training/requirements.txt`

3. Run from `training/`:

   | Step | Command | Output |
   |---|---|---|
   | Dataset report | `python dataset_report.py` | `reports/dataset_report.json` |
   | Split | `python split.py` | `data/splits.json` (seed 42, duplicate-aware) |
   | Train baseline | `python train.py --variant baseline` | `models/model_original_baseline.h5` (~30 min CPU) |
   | Train variant | `python train.py --variant improved` | `models/model_improved.h5` (optional) |
   | Evaluate | `python evaluate.py ../models/<file>.h5` | `reports/eval_<file>.json` |
   | Promote | copy the validation-best model to `models/model1.h5`, then `python evaluate.py ../models/model1.h5 --metadata` | `models/model_metadata.json` |
   | Calibration check | `python calibrate.py` | `reports/calibration.json` |
   | Curves | `python plot_curves.py` | `reports/curves_*.png` |

`evaluate.py` runs in a fresh process, loads the saved `.h5` and uses the
application's own `services.predictor.preprocess_image`, so the reported numbers
reflect what the deployed app computes. The test split is never used for training
or model selection.

Exact numbers can differ slightly between machines/TF versions even with a fixed seed.

# Rebuilding the similarity reference index

The committed index in `similarity_service/index/` can be regenerated from the same
dataset (run from the repository root, with `requirements-dev.txt` installed):

```bash
python similarity_service/download_model.py          # DINOv2-small, SHA-256 verified
python scripts/build_image_index.py --dataset "data/raw/Original Dataset"
python scripts/evaluate_visual_features.py --dataset "data/raw/Original Dataset"   # optional sanity check
```

`build_image_index.py` embeds the training and validation images listed in
`data/splits.json` (the test split is kept out), writes `embeddings.npy`,
`faiss.index` (when FAISS is installed), `metadata.json` and 200 px thumbnails.
Keep `similarity_service/index/ATTRIBUTION.md` with the thumbnails (CC BY 4.0).
