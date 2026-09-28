# I-SPY 1 pCR benchmark — reproducible analysis

Data: download "I-SPY-1-All-Patient-Clinical-and-Outcome-Data.xlsx" from the TCIA ISPY1 collection
(DOI 10.7937/K9/TCIA.2016.HdHpgJLK) and place it in this folder. You can also set the environment variable ISPY_XLSX to its path.
The data are not redistributed here.

## Run
    pip install -r requirements.txt
    python reproduce.py   > results_reproduction.txt   # re-runs manuscript v2 analyses; creates a.pkl
    python improved.py    > results_improved.txt       # revised analyses (manuscript tables 2, 3, 7, 8 and S1–S3)
    python figures.py; python figures14.py              # Figures 1-5 -> ./figures

## Benchmarking an image-based model in the same patients and folds
    python improved.py --ids my_patients.txt [--reps 50] [--out results_dir]

`my_patients.txt` holds one SUBJECTID per line, for example the patients with usable MRI at T1 and T2.
The script recomputes reference models R1–R5 on that subset and exports two files:
- `folds_subset.csv`: SUBJECTID, repetition, fold. Repetition r uses StratifiedKFold(5, shuffle=True, random_state=r), stratified on pCR × subtype.
- `oof_predictions_subset.csv`: out-of-fold predictions of R1–R5 averaged over repetitions.

Train and test your model on exactly these folds. Then compare its averaged out-of-fold predictions with pred_R1 and pred_R5 (ΔAUC with bootstrap CI, calibration, decision curves).
Without --ids, the primary set of 210 patients is used and the files are named *_primary.csv.

## Environment
Python 3.12, scikit-learn 1.8.0, statsmodels 0.15.0, SciPy 1.17.1, NumPy 2.4.4, pandas 2.3.3, lifelines 0.30.3.
