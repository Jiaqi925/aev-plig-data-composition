# Training data composition in protein–ligand binding affinity prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23248275.svg)](https://doi.org/10.5281/zenodo.23248275)

Code and experimental output for an MRes thesis investigating how the
composition of training data affects the performance of a machine-learning
scoring function. Eleven training configurations are compared under an
identical architecture, training protocol and evaluation protocol, so that
differences in performance are attributable to the training set alone.

The model is AEV-PLIG (GATv2, 367-dimensional node features, 5 layers,
7,554,561 parameters), taken unmodified from
[Valsson et al., *Commun. Chem.* **8**:41 (2025)](https://github.com/isakvals/AEV-PLIG).

## Experimental design

Each configuration is trained with 5 random seeds (42–46), giving 55 models.
The train/validation split is fixed across seeds, so seed-to-seed variation
comes only from weight initialisation and batch ordering.

| Config | Name | Training pairs |
|---|---|---|
| 1 | baseline (PDBbind + BindingNet + BindingDB-DCS) | 95,364 |
| 2 | baseline + HiQBind | 116,814 |
| 3–5 | baseline + GatorAffinity, confidence-filtered, 30 / 60 / 100% | 109,843 / 124,326 / 143,638 |
| 6–8 | baseline + GatorAffinity, unfiltered, 30 / 60 / 100% | 111,190 / 127,017 / 148,125 |
| 9 | BindingNet + DCS + HiQBind (no PDBbind) | 98,097 |
| 10 | config 9 + best GatorAffinity variant | 150,858 |
| 11 | PDBbind only | 18,717 |

The 30 / 60 / 100% GatorAffinity subsets are nested, so the three points lie
on a single data-growth curve rather than being independent samples.

Two held-out evaluation sets are used: an out-of-distribution set of 295
complexes, and the FEP benchmark (42 congeneric series, 1,087 ligands).
Metrics are series-weighted Pearson correlation (wmPCC) and Kendall's tau
for FEP, and PCC / tau_b / RMSE / MAE for OOD, each with 95% BCa bootstrap
confidence intervals. Significance is judged from paired differences, not
from overlap of individual intervals.

## Reproducing the evaluation without retraining

This path needs no dataset and no GPU. It recomputes every number in the
thesis from the delivered predictions in about three minutes.

```bash
git clone https://github.com/Jiaqi925/aev-plig-data-composition.git
cd aev-plig-data-composition
export STACEY_ROOT=$(pwd)

conda env create -f AEV-PLIG/aev-plig-linux.yml
conda activate aev-plig

python scripts/07_evaluate.py --configs 1 2 3 4 5 6 7 8 9 10 11 \
                              --seeds 42 43 44 45 46
```

Note that this overwrites `experiments/07_eval/`. Copy that directory first
if you want to compare against the delivered results.

Supplementary analyses, both CPU-only:

```bash
python scripts/09_uniprot_coverage.py   # target coverage of each source
python scripts/10_strat_ci.py           # similarity-stratified intervals
```

`07_evaluate.py` requires mmseqs2 for the stratified analysis. Without it,
that section is skipped and all other metrics are produced normally.

## GatorAffinity pipeline

The synthetic-structure arm of the study is built by two scripts:

- `scripts/03_build_gator_index.py` — pK extraction (Ki/Kd only), pairing
  with downloaded structures, Boltz-1 confidence scoring, internal
  deduplication by BindingDB MonomerID and canonical SMILES.
- `scripts/03b_gator_dedup_split.py` — cross-database deduplication against
  PDBbind and HiQBind using the CleanSplit triple criterion
  (Tanimoto > 0.9 **and** |ΔpK| ≤ 1.0 **and** TM-score > 0.8), evaluation-set
  separation, and the nested 30 / 60 / 100% sampling.

The module docstring of `03b` documents why evaluation-set separation uses
an AND combination of sequence identity and ligand similarity rather than OR,
and reports the hit counts under both readings so the choice can be checked.

Outputs: `experiments/03_gator/` (cleaned and deduplicated indices, duplicate
pairs, test-set flags, and the two stage reports `report_03a.json` /
`report_03b.json`).

## Repository layout

```
scripts/                     numbered pipeline stages plus shared libraries
  leak.py                    evaluation-set PDB IDs to exclude (330)
  01-03b                     index construction, cleaning, deduplication
  04_generate_graphs.py      AEV + PLIG graph featurisation
  05_build_configs.py        assembly of the 11 training indices
  06_train_all.py            concurrent training and inference
  07_evaluate.py             metrics, BCa intervals, paired differences
  08-10                      integrity audit, coverage, stratification
  99_make_smoke_config.py    small config for smoke-testing the pipeline
  lib_data / lib_eval / lib_model

experiments/
  00_meta/                   leakage IDs, sequence pools, test-set indices
  01-03_*/                   cleaned source indices
  05_configs/                the 11 training indices
  06_train/cfgXX/seedYY/     done.json, history.csv, prediction arrays
  07_eval/                   main results (see below)
  07_eval_base11/            reverse paired test, config 11 as reference
  09_uniprot_coverage/       target overlap with the OOD set
  10_strat_ci/               stratified confidence intervals
  audit_report.json          61 integrity checks, all passing

figs/                        figure scripts and the values behind each panel
datasets/index/              index files for PDBbind, BindingNet, HiQBind
AEV-PLIG/                    upstream reference implementation (BSD-3)
```

### Results files

| File | Contents |
|---|---|
| `07_eval/table_main.csv` | per-config metrics with 95% BCa intervals |
| `07_eval/table_single.csv` | per-seed metrics, showing the ensembling gain |
| `07_eval/paired_diff.csv` | paired differences against config 1 |
| `07_eval/per_series/cfgXX.csv` | per-series r / tau / n over the 42 series |
| `07_eval/scaling.csv` | log-linear scaling fit (k, b, R²) |
| `07_eval/strat.csv` | sequence-similarity stratified metrics |
| `07_eval/results.json` | all of the above in structured form |

## Not included in this repository

**Model checkpoints** (55 × 29 MB), deposited on Zenodo:
[10.5281/zenodo.23248276](https://doi.org/10.5281/zenodo.23248276)
(all versions: [10.5281/zenodo.23248275](https://doi.org/10.5281/zenodo.23248275)).
Each checkpoint carries its own pK standardisation constants, so inference
needs no further files. The evaluation above does not require them: it runs
from the prediction arrays already in this repository.

**Raw structure files** (~183 GB) and **graph featurisation output**
(7.6 GB). The latter is a deterministic intermediate, rebuilt by
`scripts/04_generate_graphs.py`. Sources for the former:

| Source | Size | Where to get it |
|---|---|---|
| PDBbind v2020 | ~11 GB | http://www.pdbbind.org.cn/ (registration) |
| BindingNet v1 | ~24 GB | http://bindingnet.huanglab.org.cn/ |
| BindingDB-DCS | ~18 GB | docked congeneric series, BindingDB affinities |
| HiQBind | ~36 GB | https://github.com/THGLab/HiQBind |
| GatorAffinity-DB | ~36 GB | https://github.com/Vina-Lab/GatorAffinity-DB |

The GatorAffinity index CSV (900 MB) is downloaded separately from the same
repository into `datasets/index/gatoraffinity/`. Of its 463,867 entries,
381,291 carry a usable Ki or Kd; 65,001 of those had structures downloaded,
limited by storage. Internal deduplication leaves 54,290, and after
leakage removal and evaluation-set separation the unfiltered version holds
52,773 pairs and the confidence-filtered version 48,286.
`03_build_gator_index.py` skips missing entries automatically.

Third-party evaluation-set inputs and the BindingDB-DCS series files are
also omitted, as they are not ours to redistribute.

External tools: TM-align (structure similarity) and mmseqs2 (sequence
similarity) must be on `PATH` to rerun the deduplication stages. mmseqs2
is also used by the stratified analysis in `07_evaluate.py`.

## Notes

Structure-file paths in the index CSVs are relative to the repository root
and resolved at runtime against `$STACEY_ROOT`. When that variable is unset,
the parent directory of `scripts/` is used, so the repository runs from any
location.

Two source labels in `09_uniprot_coverage.py` and `figs/fig2_coverage.py`
are in Chinese. They are written into `uniprot_coverage.csv` and used as
lookup keys by the figure script, so they are kept as they are to stay
consistent with the delivered data.

## Citation and licence

The upstream AEV-PLIG implementation in `AEV-PLIG/` is distributed under the
3-Clause BSD License, Copyright (c) Ísak Valsson, Oxford Protein Informatics
Group, 2024. `scripts/lib_model.py` is a verbatim copy of its model
definition and carries the same terms. Code in `scripts/` and `figs/` is
released under the same licence.

Jiaqi Wang, MRes Bioengineering, Imperial College London, 2026.
Supervisor: Dr Pedro Ballester.
