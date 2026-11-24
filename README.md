# Reproduction code for

Griffon, M., Delestre, N., Guériau, M. (2026). **An Approach for Evaluating Recommender Systems: Assessing Usefulness and Coherence Through Item Relationships**. In: Zanni-Merk, C., Ismayilov, E., Ismayilova, N., Genaud, S., Abdulkarimova, U. (eds) Artificial Intelligence for Digital Transformations. AIDT 2026. Communications in Computer and Information Science, vol 3023. Springer, Cham. https://doi.org/10.1007/978-3-032-31319-5_15

---

# Erratum (code corrected after publication)

**The Usefulness and ILR values reported in the paper were computed with an implementation error. The code on the `main` branch is corrected; the corrected results differ from the published ones and are given below.**

**What was wrong.** Substitutability and complementarity are symmetric relations: the score of a pair of items does not depend on the order of the two items. Each pair is stored once, as `(smaller item ID, larger item ID)`, but the function `max_relation_score` in `src/metrics/usefulness_ilr.py` looked a pair up only in the order in which it was presented. A pair presented as `(larger ID, smaller ID)` was therefore scored 0 instead of its actual score.

**What is affected.**

* **Affected:** the Usefulness and ILR experiments (step 3 below): the corruption curves and the degradation indicators (Spearman correlation and total drop).
* **Not affected:** the train/test splits, the substitutability and complementarity scores, the IC score, and the precision/recall evaluation of the scores (steps 1 and 2 below).

**Published vs. corrected results.** Same protocol, same splits, same seed and same corruption draws; only the relation lookup differs. *Spearman* is the correlation between the corruption ratio and the mean metric value; *Drop* is the mean value without corruption minus the mean value at full corruption.

| Domain | Metric | Spearman (published) | Spearman (corrected) | Drop (published) | Drop (corrected) |
|---|---|---|---|---|---|
| Electronic | ILR | -0.5438 | -0.6895 | 0.0039 | 0.0084 |
| Electronic | Usefulness | -0.4643 | -0.5000 | 0.0007 | 0.0137 |
| Clothing | ILR | 0.3915 | 0.1725 | 0.0009 | 0.0017 |
| Clothing | Usefulness | -0.3349 | -0.5714 | 0.0007 | 0.0055 |
| Food | ILR | -0.6674 | -0.6641 | 0.0041 | 0.0112 |
| Food | Usefulness | -0.5406 | -0.9286 | 0.0019 | 0.0145 |
| Global | ILR | -0.8155 | -0.8112 | 0.0030 | 0.0073 |
| Global | Usefulness | -0.7857 | -0.8214 | 0.0011 | 0.0114 |

**Impact on the conclusions.** The conclusions of the paper are unchanged, and for the most part strengthened. With the corrected code, the drop under corruption is larger for both metrics in every domain (about 2 to 3 times larger for ILR, 8 to 20 times larger for Usefulness), and the correlation between corruption and Usefulness is more strongly negative in every domain. For ILR, the correlation is more strongly negative on Electronic, essentially unchanged on Food and globally, and remains positive on Clothing (0.17 instead of 0.39).

**Versions.**

* Release [`v1.0-paper`](https://github.com/MadPuppyForce/evaluating-recsys-item-relationships/releases/tag/v1.0-paper): the code as published, which reproduces the values reported in the paper.
* Release [`v1.1-corrected`](https://github.com/MadPuppyForce/evaluating-recsys-item-relationships/releases/tag/v1.1-corrected) and the `main` branch: the corrected code.
* The table above can be regenerated with `scripts/recompute_symmetric_usefulness_ilr.py`, which computes both versions side by side (see step 4 below).

The code was corrected and this erratum published in November 2026.

---

# Overview

This repository contains the code used to reproduce the experiments presented in the paper:

> *An approach for evaluating recommender systems: assessing usefulness and coherence through item relationships*

The goal of this work is to introduce an approach for evaluating recommender systems. We proposed two complementary metrics, **Usefulness** and **Intra-List Relationships** (ILR), designed to capture respectively the alignment of recommendations with the user’s session context and the internal relational coherence of recommendation sets. Both metrics rely on item–item relationships modeled through **substitutability** and **complementarity** scores derived from user session interactions.

The repository provides scripts to:

1. Build train/test splits from raw BundleRec datasets
2. Compute substitutability and complementarity scores
3. Compute the IC score introduced in Sun, Z., Yang, J., Zhang, J., Bozzon, A.: Exploiting both vertical and horizontaldimensions of feature hierarchy for effective recommendation. Proc. AAAI Conf. Artif. Intell. 31(1) (2017)
4. Evaluate the substitutability and complementarity scores
5. Evaluate the ILR and Usefulness metrics

---

# Repository structure

```
mon-papier-repro/
│
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE.txt
│
├── data/
│   ├── raw/              # original datasets
│   ├── processed/        # processed datasets (train/test splits)
│   └── README.md         # dataset download instructions
│
├── src/
│   ├── metrics/
│   │   └── usefulness_ilr.py
│   │
│   └── scores/
│       ├── ic.py
│       └── substitution_complementarity_scores.py
│
├── scripts/
│   ├── build_splits.py
│   ├── run_scores_experiment.py
│   ├── run_metrics_experiment.py
│   └── recompute_symmetric_usefulness_ilr.py   # erratum: published vs. corrected
│
└── results/
```

---

# Environment setup

The project uses **uv** to manage the Python environment.

## Install uv

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Create the virtual environment

```
uv venv
```

## Install dependencies

```
uv sync
```

---

# Dataset installation

The datasets used in this work come from the **BundleRec dataset** introduced in:

* Sun et al., Revisiting Bundle Recommendation: Datasets, Tasks, Challenges and Opportunities for Intent-Aware Product Bundling, SIGIR 2022.
* Sun et al., Revisiting Bundle Recommendation for Intent-aware Product Bundling, ACM TORS 2024.

See:

https://github.com/BundleRec/bundle_recommendation

Detailed instructions are provided in:

```
data/README.md
```

The expected directory structure is:

```
data/raw/
    clothing/
        session_item.csv
        bundle_item.csv
        session_bundle.csv

    electronic/
        session_item.csv
        bundle_item.csv
        session_bundle.csv

    food/
        session_item.csv
        bundle_item.csv
        session_bundle.csv
```

---

# Reproducing the experiments

All random seeds are fixed to **42** to ensure reproducibility.

Experiments should be run in the following order.

---

## 1. Build train/test splits

```
uv run python scripts/build_splits.py
```

This script:

* splits sessions into **train (80%)** and **test (20%)**
* applies a swap-based optimization to improve the balance of unique item occurrences across splits
* saves the processed data into:

```
data/processed/<domain>/
```

---

## 2. Run scores experiments

```
uv run python scripts/run_scores_experiment.py
```
This scrip evaluates the ability of our substitutability and complementarity scores to identify meaningful relationships.
The identified relationships are compared against a ground truth constructed from the annotated bundles.

Outputs include:

* substitutability and complementarity score tables
* precision and recall summary

Outputs are saved in:

```
results/<domain>/
results/
```

---

## 3. Run ILR and Usefulness experiments

```
uv run python scripts/run_metrics_experiment.py
```

This script evaluates the behavior of the metrics under controlled corruption.

Outputs include:

* aggregated metric tables
* plots of metric degradation vs corruption level

Results are saved in:

```
results/<domain>/
results/
```

---

## 4. Compare published and corrected metrics (erratum)

```
uv run python scripts/recompute_symmetric_usefulness_ilr.py
```

This script requires the outputs of step 2. It re-runs the protocol of step 3 and computes, for every corrupted sample, both the published version of the metrics (`ref`, single-orientation lookup) and the corrected version (`sym`, symmetric lookup). See the Erratum section at the top of this file.

Outputs are saved in:

```
results/<domain>/ilr_agg_symcompare.csv
results/<domain>/usefulness_agg_symcompare.csv
results/symcompare_summary.csv
```

---

# Hardware configuration

All experiments were run on:

* **OS:** Ubuntu 22.04
* **CPU:** Intel Core i9-12900K
* **RAM:** 64 GB
* **Python:** 3.10.12

---

# Citation

If you use this code, please cite:

```bibtex
@inproceedings{griffon2026usefulnessilr,
  author    = {Griffon, Marie and Delestre, Nicolas and Gu{\'e}riau, Maxime},
  title     = {An Approach for Evaluating Recommender Systems: Assessing Usefulness and Coherence Through Item Relationships},
  booktitle = {Artificial Intelligence for Digital Transformations},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  pages     = {220--235}
}
```

You should also cite the dataset:

```bibtex
@inproceedings{sun2022revisiting,
  title={Revisiting Bundle Recommendation: Datasets, Tasks, Challenges and Opportunities for Intent-Aware Product Bundling},
  author={Sun, Zhu and Yang, Jie and Feng, Kaidong and Fang, Hui and Qu, Xinghua and Ong, Yew Soon},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2022}
}

@article{sun2024revisiting,
  title={Revisiting Bundle Recommendation for Intent-aware Product Bundling},
  author={Sun, Zhu and Feng, Kaidong and Yang, Jie and Fang, Hui and Qu, Xinghua and Ong, Yew-Soon and Liu, Wenyuan},
  journal={ACM Transactions on Recommender Systems},
  year={2024},
  publisher={ACM New York, NY}
}
```

---

# License

This project is released under the MIT License.
