# Attribution

## Dataset: VegNet (CC BY 4.0)

The disease classifier was trained on, and the reference images used by visual
similarity search (`similarity_service/index/`) are derived from, the VegNet dataset:

- **Title:** VegNet: An organized dataset of cauliflower disease
- **Authors:** Aditya Rajbongshi, Umme Sara, Rashiduzzaman Shakil, Bonna Akter, Mohammad Shorif Uddin
- **Repository:** Mendeley Data, https://data.mendeley.com/datasets/t5sssfgn2v/3
- **DOI:** [10.17632/t5sssfgn2v.3](https://doi.org/10.17632/t5sssfgn2v.3)
- **Article:** "VegNet: An organized dataset of cauliflower disease for a sustainable agro-based
  automation system", *Data in Brief* (2022), https://pubmed.ncbi.nlm.nih.gov/35811654/
- **Licence:** [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

The images were obtained through a Kaggle re-upload of the same 656 original images
([noamaanabdulazeem/cauliflower-dataset](https://www.kaggle.com/datasets/noamaanabdulazeem/cauliflower-dataset)).
The dataset itself is not included in this repository.

**Changes made:** the 557 training and validation images in the reference index were resized
to thumbnails (at most 200 px) and re-encoded as JPEG. No other edits were made. Photos in
`docs/screenshots/` show held-out test images from the same dataset.
Details: [`similarity_service/index/ATTRIBUTION.md`](similarity_service/index/ATTRIBUTION.md).

## Model: DINOv2-small (Apache-2.0)

Visual similarity search uses DINOv2-small by Meta AI
([facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small), Apache-2.0), via the
ONNX export [onnx-community/dinov2-small](https://huggingface.co/onnx-community/dinov2-small).
The model file is not committed; `similarity_service/download_model.py` downloads it and
verifies its SHA-256 checksum.

## Disease information

Disease descriptions in `data/diseases.json` are summarised from the public sources listed
with each entry in that file.
