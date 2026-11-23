# Used-car machine-learning workflow

Car-data cleaning, price regression and K-nearest-neighbour classification. This is separate from the smaller Code Alpha car dataset.

## Project files and local data inputs

Data files listed below are local inputs, not bundled public downloads. See `PUBLIC_DATA_SETUP.md`.

- `assigment1_clean_step1.ipynb`
- `modeling.ipynb`
- `handling_script.py`

## Run

First restore the locally required data described in `PUBLIC_DATA_SETUP.md`.

Use Python 3.11+ in a virtual environment, then install dependencies:

```sh
python -m pip install -r requirements.txt
jupyter notebook
```

Open a notebook from this repository and run it from the repository root.

## Review notes

- Organized from the supplied project archives; duplicate folders, cache files and assignment handouts were not counted as separate projects.
- Saved notebook outputs were cleared so old results are not mistaken for a current successful run.
- Raw datasets are not included in this public source copy. See `PUBLIC_DATA_SETUP.md` for exact local filenames and expected columns. No new license is assigned.
- This is a portfolio/learning project, not a validated decision system. See `VALIDATION.md` for checks and remaining limits.

## Repaired entry point

Run `python train.py` for a fresh train/test baseline with train-only imputation, encoding and scaling. It uses Ridge regression and KNN price-tier classification. The original notebook calls several absent fit_handle_* methods; it is retained as exploratory source rather than claimed to run end to end.
