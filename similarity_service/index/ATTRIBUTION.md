# Reference images: attribution and licence

The thumbnails in `thumbs/` (557 images, resized to at most 200 px) and the embeddings in
`embeddings.npy` / `faiss.index` are derived from the **VegNet** dataset.

- **Dataset:** VegNet: An organized dataset of cauliflower disease
- **Authors:** Aditya Rajbongshi, Umme Sara, Rashiduzzaman Shakil, Bonna Akter, Mohammad Shorif Uddin
- **Repository:** Mendeley Data, https://data.mendeley.com/datasets/t5sssfgn2v/3
- **DOI:** [10.17632/t5sssfgn2v.3](https://doi.org/10.17632/t5sssfgn2v.3)
- **Related article:** "VegNet: An organized dataset of cauliflower disease for a sustainable agro-based
  automation system", *Data in Brief* (2022), https://pubmed.ncbi.nlm.nih.gov/35811654/
- **Licence:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

CauliCare obtained the images through a Kaggle re-upload
([noamaanabdulazeem/cauliflower-dataset](https://www.kaggle.com/datasets/noamaanabdulazeem/cauliflower-dataset)),
which contains the same 656 original images in the same four classes.

**Changes made:** only the training and validation images were used (the test split is excluded),
and each image was resized to a thumbnail and re-encoded as JPEG. No other edits were made.
The embeddings were computed with DINOv2-small by Meta AI
([facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small), Apache-2.0; ONNX export
[onnx-community/dinov2-small](https://huggingface.co/onnx-community/dinov2-small)).

`metadata.json` lists, for each thumbnail, its class label, its split and the original file name.
