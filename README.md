# TumorConnectome

**An exploratory longitudinal analysis of structural–functional connectome coupling and cognitive change following brain tumor surgery.**

TumorConnectome is a research-code repository for investigating longitudinal structural connectivity (SC), functional connectivity (FC), and their relationship to neuropsychological change using the publicly available **Brain Tumor Connectomics (BTC)** preoperative and postoperative datasets.

> **Status:** Exploratory research code / work in progress. Findings are not clinical predictions or validated biomarkers. The original BTC MNI registration transforms and exact template correspondence to the candidate voxelwise DK68 atlas have not been verified.

## Research questions

1. Does global SC–FC coupling change from preoperative to postoperative imaging, and does this change differ between patients and controls?
2. Are individual changes in SC–FC coupling associated with longitudinal changes in cognitive performance?
3. How sensitive are tumor-mask spatial features to thresholding and image-space definitions?

## Data and analysis scope

- **Preoperative:** 36 scans (25 tumor patients and 11 controls).
- **Postoperative matched SC/FC cohort:** 27 participants/scans (17 tumor patients and 10 controls), subject to data availability and cohort inclusion criteria.
- **SC–FC session records:** 63 across both time points.
- **Cognitive-change association:** 16 patients with usable data for the reported sustained-attention analysis.
- **Network parcellation:** 68 Desikan–Killiany cortical regions for SC–FC analyses. Crossmodal node ordering/identity was checked for all 63 SC/FC records.

These counts describe the locally audited analysis cohort, not necessarily all participants in the original data releases. Refer to the analysis scripts for selection rules.

## Preliminary findings

- The between-group difference in longitudinal **global SC–FC coupling change** was not statistically significant in the exploratory analysis (reported *p* approximately 0.65).
- Among 16 patients with matched cognitive measurements, the change in global SC–FC coupling was associated with the change in sustained attention (Spearman **ρ ≈ −0.585**, unadjusted **p ≈ 0.019**).
- The sustained-attention association is **exploratory**: the sample is small, the reported p-value is unadjusted for multiple comparisons, and no causal interpretation or prospective predictive validity is claimed.

These results require independent replication and should not be treated as a clinically validated outcome model.

## Repository layout

```text
TumorConnectome/
├── README.md
├── .gitignore
└── scripit/                # Existing directory name retained for compatibility
    ├── 01_audit_btc_dataset.py
    ├── 02_build_analysis_cohort.py
    ├── 02b_integrate_clinical_metadata.py
    ├── 03_global_sc_fc_coupling.py
    ├── 03b_network_cognition_association.py
    ├── 03c_validate_network_cognition.py
    ├── 04_validate_tumor_masks.py
    ├── 05b_verify_crossmodal_node_alignment_v2.py
    ├── 05_validate_DK68_atlas.py
    └── 06*.py               # Exploratory tumor-space / QC workflows
```

The `scripit/` name is preserved from the current working project. The scripts are research utilities and are not yet packaged as a single turnkey pipeline.

## Workflow

| Stage | Focus | Interpretation |
| --- | --- | --- |
| 01–02 | Dataset audit, cohort assembly, clinical metadata | Data preparation |
| 03–03c | Longitudinal SC–FC coupling and cognitive association | Main exploratory analyses |
| 04–05 | Tumor-mask QC and SC/FC DK68 node alignment | Quality control |
| 06a–06h | Candidate atlas construction, mask thresholds, clinical-volume comparisons, spatial and native-space QC | Exploratory / partially validated workflows |
| 06i | Independent Native T1 → template registration experiments | Initial 6/12-DOF tests failed anatomical QC; a later FSL MNI152 rigid-registration diagnostic improved gross orientation and preserved tumor volume in PAT23, but remains **unvalidated** |

### Registration diagnostic (PAT23 only)

A separate, exploratory registration investigation evaluated whether the native-space lesion could be transferred into a standard reference space:

- Initial direct FLIRT registration to the FreeSurfer `cvs_avg35_inMNI152` reference produced anatomically implausible alignment. The 12-DOF result reduced the PAT23 lesion volume from **103.505 cm³** to approximately **48.24 cm³** at the candidate `>0.5` threshold; the original transform was rejected.
- A subsequent diagnostic used the FSL `MNI152_T1_1mm` reference, first resampling native T1 by its existing image affine onto the reference grid and then estimating a constrained **6-DOF** rigid transform. The estimated matrix had **determinant ≈ 0.9999998**, with singular values near 1.
- Applying this staged transform to the continuous native tumor mask yielded a thresholded volume of **103.174 cm³**, compared with **103.505 cm³** in native space (approximately **99.68% volume retention**). The gross anterior lesion location was visually plausible.
- **This is a single-subject technical diagnostic, not validated anatomical normalization.** Rigid-body volume preservation and plausible lesion location do not establish accurate voxelwise registration. The original BTC standard-space template, transform provenance, exact DK68 correspondence, and cohort-wide registration quality remain unverified. No patient-specific or derived template-space transforms from these experiments are endorsed for regional tumor-burden inference.
- The staged method uses an *additional* transform estimated from an already-resampled T1. Its FLIRT matrix **must not be applied directly to the original native mask** without accounting for that initialization. Experimental image outputs and matrices are not distributed with the code release.

**Important:** Verifying that two images have matching grid dimensions and affines does **not** independently establish that they originate from the same anatomical template. The candidate voxelwise DK68 atlas was **not** accepted for quantitative tumor-to-DK68 regional-burden analysis.

## Getting started

### Requirements

Recommended environment: Python 3.11 with scientific Python packages, including `numpy`, `pandas`, `scipy`, `matplotlib`, and `nibabel`. Some scripts also require packages for `.mat`/connectome processing. The exploratory registration scripts require a separately installed **FSL FLIRT** executable.

Install packages according to imports in the script you intend to run. Dependencies have not yet been frozen or tested as a complete cross-platform environment.

### Data access

Obtain the BTC data from the original public dataset releases, following their licenses, citations, and access terms. **MRI scans, individual-level clinical tables, and local derivative files are not redistributed here.**

The working scripts expect data under a local data root and may require additional local paths or output files from earlier stages. For scripts that provide a command-line interface, inspect available arguments:

```bash
python scripit/03_global_sc_fc_coupling.py --help
```

Not every script is guaranteed to expose `--help` or operate as a standalone command. Read the script header and verify its inputs before execution. Do not assume numerical results are reproducible without the same BTC versions and cohort exclusions.

## Interpretation and limitations

- Small longitudinal patient sample, particularly for cognition-related analyses.
- Exploratory comparisons and potential multiplicity across cognitive and network outcomes.
- Observational data do not identify a causal effect of network reorganization on cognition.
- Tumor masks can contain low positive interpolation values: `>0` may greatly inflate apparent volume. A threshold of `>0.5` showed better agreement with clinical volumes in the local cohort, but it is **not independently validated as a universal segmentation threshold**.
- Native T1–tumor-mask geometric resampling checks found no automatic red flags in 25 patients; this is not equivalent to expert segmentation validation.
- Exact provenance of the supplied BTC MNI tumor-mask registration and identity of the FreeSurfer candidate reference template have not been established.
- Initial direct FLIRT registration tests for PAT23 produced gross anatomical mismatch. A later FSL MNI152 constrained rigid-registration diagnostic preserved approximately 99.68% of the thresholded tumor volume and improved gross lesion orientation, but **was not independently validated for accurate regional localization**. None of these experimental transforms is approved for regional tumor-burden quantification.

## Data citation

Please cite the original BTC dataset and the relevant primary publications when reusing the data:

- Aerts H. et al. *Modeling Brain Dynamics in Brain Tumor Patients Using the Virtual Brain.* **eNeuro** (2018). DOI: [10.1523/ENEURO.0083-18.2018](https://doi.org/10.1523/ENEURO.0083-18.2018).
- Aerts H. et al. *Modeling brain dynamics after tumor resection using The Virtual Brain.* **NeuroImage** (2020), 213:116738. DOI: [10.1016/j.neuroimage.2020.116738](https://doi.org/10.1016/j.neuroimage.2020.116738).

For the precise data repository citation and license, consult the BTC release documentation.

## Project status

**Code-release milestone:** dataset audit, SC/FC alignment, longitudinal coupling, exploratory cognitive association, and tumor-mask QC workflows assembled. Additional validation is required before presenting this as a completed prospective predictive or spatially localized model.

## License

No code license has been selected yet. Until a license is added, standard copyright restrictions apply. Please review third-party code and dependency licenses before choosing a repository license.
