# Positive events converge in the mind, but diverge in the brain

Code and data for the manuscript investigating whether individuals converge more
on positive versus negative emotional experiences, behaviorally and neurally.

## Contents

- `Analysis/Emotion_Similarity_Consolidated.ipynb` — the full analysis pipeline
  (Studies 1-4): behavioral appraisal similarity, PCA complexity, an independent
  video-dataset replication, a PLS appraisal-to-emotion transformation model, and
  fMRI ROI inter-subject similarity.
- `Analysis/IAPS_rating.csv` — behavioral appraisal ratings for the fMRI subsample
  (Study 1/3/4).
- `Online/CIAPS_Appraisal_Recovered_NNMF_SGD.csv` — the full MTurk sample's
  NNMF-recovered appraisal ratings (Study 1 headline analysis, N=201).
- `Analysis/PLS_transformation_by_image.csv` — per-image PLS transformation
  results (Study 3).
- `Analysis/figs/` — generated figures and the fMRI ROI similarity/permutation
  test results (Study 4), including thresholded statistical maps (`.nii.gz`).

## Not included here

- **Raw fMRI data** (single-trial beta maps, ~157GB): these are human-subjects
  neuroimaging data and are available on request, subject to appropriate data
  use agreements, rather than distributed via a public git repository.
- **Study 2 stimulus/response data**: from Cowen & Keltner (2017, *PNAS*),
  "Self-report captures 27 distinct categories of emotion bridged by continuous
  gradients." This is a third-party dataset; see the original publication for
  access to their data.

## Reproducing the analysis

The notebook was built and run with Python 3.11 (pandas, numpy, scikit-learn,
scipy, nltools, nibabel, matplotlib, seaborn). Study 4 additionally requires the
Neurosynth-derived 50-region parcellation used for the fMRI ROI analysis
(fetched automatically in the notebook).
