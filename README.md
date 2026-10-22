# People agree more about good experiences, but their brains respond more alike to bad ones

Code and data for the manuscript investigating whether individuals converge more
on positive versus negative emotional experiences, behaviorally and neurally.

## Contents

- `Analysis/Emotion_Similarity_Consolidated.ipynb` — the full analysis pipeline
  (Studies 1-4): behavioral appraisal similarity, PCA structure, an independent
  video-dataset replication, appraisal-to-emotion prediction (nested cross-validation
  and a Cowen & Keltner style analysis across images), and fMRI ROI inter-subject
  similarity.
- `Analysis/study3_transformation.py` — the Study 3 analyses (nested cross-validation
  and the across-image analysis), imported by the notebook.
- `Analysis/robustness_studies1to3.py` — the Study 2 check without arousal and the
  Study 3 alternative-model table (`Analysis/figs/robustness_studies1to3.csv`).
- `Analysis/make_manuscript_figures.py` — assembles Figs. 1-4 from the notebook's panels.
- `Analysis/IAPS_rating.csv` — behavioral appraisal ratings for the fMRI subsample
  (Study 1/3/4).
- `Online/CIAPS_Appraisal_Recovered_NNMF_SGD.csv` — the full MTurk sample's
  NNMF-recovered appraisal ratings (Study 1 headline analysis, N=201).
- `Analysis/figs/` — generated figures and the fMRI ROI similarity/permutation
  test results (Study 4), including thresholded statistical maps (`.nii.gz`).
  `image_neural_similarity/` holds the image-by-region similarity matrices for
  each first-level model (`ROI_Image_Similarity_<model>.csv`), the supplementary
  Study 4 tables (`Study4_*.csv`), and, in `original_impulse_2021/`, the outputs
  of the original one-TR impulse analysis.
- `Analysis/fmri/` — the Study 4 pipeline, in the order it is run:
  1. `first_level_betas.py <subject>` — single-trial betas under three timing
     models (one-TR impulse, 4 s boxcar, 4 s boxcar with one model per trial).
     The 4 s boxcar with one model per run is the primary model.
  2. `roi_similarity.py` — inter-subject similarity per image and region for
     each model (and with run means equalized), plus compact subject-level arrays.
  3. `supplement_analyses.py` — exact category-level permutation tests.
  4. `run_variance.py` — run structure of the single-trial betas.
  5. `inter_subject_rsa.py` — inter-subject RSA of neural vs appraisal similarity.
  6. `design_checks.py` — run-assignment overlap, valence-label agreement, and
     split-half reliability of the pairwise similarity measures.

## Not included here

- **Raw fMRI data** (preprocessed images and single-trial beta maps, ~157GB):
  too large for a git repository; available at [repository link to be added].
  Step 1 of the pipeline above needs them.
- **Anonymized subject-level arrays** (`fMRI/derived/roi_pairwise_<model>.npz` for
  the one-TR impulse, 4 s boxcar and 4 s boxcar per image models: pairwise pattern
  correlations per region and image). With the similarity matrices in `Analysis/figs/` they
  reproduce `roi_similarity.py`'s similarity matrices, `supplement_analyses.py`,
  `inter_subject_rsa.py`, and the reliability and label checks in `design_checks.py`.
  They do not reproduce `first_level_betas.py`, `run_variance.py`, the
  run-mean-equalized models, or the run-assignment check, which need the voxel betas
  and onset files. The arrays are too large for git and are archived separately
  at the Open Science Framework (https://osf.io/qn3ve/) [DOI to be added].
- **Study 2 stimulus/response data**: from Cowen & Keltner (2017, *PNAS*),
  "Self-report captures 27 distinct categories of emotion bridged by continuous
  gradients." This is a third-party dataset; see the original publication for
  access to their data.

## License

The code is released under the MIT License (`LICENSE`). The behavioral rating files
(`Analysis/IAPS_rating.csv`, `Online/CIAPS_Appraisal_Recovered_NNMF_SGD.csv`) and the derived
similarity tables are released under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/), the license also used for the archived arrays.
The IAPS images and the Cowen & Keltner data are not part of this repository and keep their own terms.

## Reproducing the analysis

The notebook was built and run with Python 3.11 (pandas, numpy, scikit-learn,
scipy, nltools, nibabel, nilearn, statsmodels, matplotlib, seaborn); the exact
versions are in `environment.yml` (`conda env create -f environment.yml`). Study 4
additionally requires the Neurosynth-derived 50-region parcellation used for the
fMRI ROI analysis (NeuroVault collection 2099; fetched automatically in the
notebook and cached in `fMRI/derived/`).

Studies 1-3 run end to end from the files in this repository. For Study 4, the
notebook reads the similarity matrices and supplementary tables in
`Analysis/figs/image_neural_similarity/`; to regenerate them, run the scripts in
`Analysis/fmri/` in the order listed above (steps 3 and 5, and the parts of step 6
described above, need only the archived subject-level arrays). Six of the 37 fMRI participants are excluded because their
onset files assign 22-24 images to more than one run, so betas could not be
matched to images; 31 are analyzed.
