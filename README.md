# URLGuard

[Reading order](READING-ORDER.md) · [Team roles](ROLES.md) · [Two-person viva practice](VIVA-PRACTICE.md) · [Code walkthrough](CODE-WALKTHROUGH.md)

Machine learning based phishing URL detection. Machine Learning Lab mini project.

Jaypee Institute of Information Technology, Noida

Department of Computer Applications. Machine Learning Lab. MCA, 1st Year, 1st Semester.

Use this file to explain the project in a viva. The numbers below come from the saved training run (2026-11-04 04:59:46 UTC). They are not typed in by hand. Longer notes are in [DATASET.md](DATASET.md), [MODEL_REPORT.md](MODEL_REPORT.md), and [VIVA.md](VIVA.md).

## One-minute explanation

URLGuard checks a pasted link and estimates whether it looks legitimate or phishing. The browser does not run Python and it does not open the website. It sends the URL text to a local FastAPI service. FastAPI turns the text into 12 numbers with `features_from_normalized`, then calls a saved scikit-learn model. The saved model is **Random Forest**. On the held-out test split its accuracy is 0.9964, precision 0.9984, recall 0.9932, and F1 0.9958. This is an educational estimate, not a security guarantee.

## How one check works

1. The user types a URL on the Scanner page and presses Analyze URL.
2. The Next.js page sends `POST /api/predict` with JSON `{ "url": "..." }` to FastAPI at http://127.0.0.1:5000.
3. `normalize_url` accepts only HTTP and HTTPS. A missing scheme becomes `https://`. A leading `www.` is removed.
4. `features_from_normalized` returns the same 12 numbers, in the same order, that were used in training.
5. The saved model runs `predict` for the class and `predict_proba` for the phishing score. The phishing score is the column for class 1, read from `model.classes_`, not a guessed index.
6. The page shows SAFE or PHISHING, the score, the model name, and the URL traits that were measured. The submitted URL is plain text, not a clickable link.

Training is a different program. `python train.py` fits the models. `python app.py` only loads `backend/models/best_model.joblib`. Starting the API does not train again.

```text
Next.js frontend  http://localhost:3000
        |
        |  HTTP JSON
        v
FastAPI + Uvicorn  http://127.0.0.1:5000
        |
        |  features_from_normalized(normalized)
        v
Saved scikit-learn model  backend/models/best_model.joblib
```

## What each file does

- `backend/download_data.py` downloads UCI dataset 967 with `fetch_ucirepo` and saves only `url` and the original label in `data/urls.csv`. It does not invent rows.
- `backend/features.py` has `features_from_normalized`. Training and the API both call this function. It never opens the URL.
- `backend/train.py` cleans the table, builds X and y, splits the rows, trains three models, picks one on validation F1, tests that model once, and saves the Joblib file plus the reports.
- `backend/app.py` is the FastAPI service. `load_artifacts` runs once at startup.
- `backend/write_docs.py` writes this README and the other reports from `results/metrics.json`.
- `frontend/` is the Next.js site. Scanner is the check. Model Insights shows the real validation table, test scores, and confusion matrix.

## Technology

- Frontend: Next.js App Router, React, TypeScript, Tailwind CSS, shadcn/ui, Lucide icons
- Backend: Python 3.11.15, FastAPI, Uvicorn
- Machine learning: scikit-learn 1.9.1, Pandas 3.1.0, NumPy 2.4.6, Joblib, Matplotlib
- Dataset download: ucimlrepo
- Seed: `random_state = 42`

## Dataset, in the words you can say

The dataset is the PhiUSIIL Phishing URL Dataset, UCI id 967, by Arvind Prasad and Shalini Chandra. Official page: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset. License: Creative Commons Attribution 4.0 International (CC BY 4.0).

The file has 235,795 rows. UCI says 134,850 are legitimate and 100,945 are phishing. The downloaded counts match that.

Original UCI labels: **1 = legitimate**, **0 = phishing**. The UCI page says: "Label 1 corresponds to a legitimate URL, label 0 to a phishing URL."

This project flips the label so phishing is the positive class:

```text
phishing_label = 1 - original_label
```

After that, **0 = SAFE** and **1 = PHISHING**. Precision, recall, and F1 all treat phishing as the positive class.

Cleaning removed 0 invalid rows and 0 rows with conflicting labels. After a leading `www.` was removed, 571 extra copies of the same host were dropped. Final cleaned rows: 235,224 (134,850 safe, 100,374 phishing).

Modeling used a stratified sample of 30,000 real rows, not synthetic rows and not the full file. Split with `random_state = 42`:

| Split | Rows | SAFE | PHISHING |
|---|---:|---:|---:|
| Training (70%) | 21,000 | 12,039 | 8,961 |
| Validation (15%) | 4,500 | 2,580 | 1,920 |
| Test (15%) | 4,500 | 2,580 | 1,920 |

Identical URLs across those three splits: 0. X is the feature matrix. y is the label vector. The model is fit only on the training rows.

## Why a leading www. is removed

In the raw file, every legitimate URL starts with `www.` Many phishing URLs do not. If that prefix stayed in the features, the model would learn "no www means phishing." Then a real site such as `x.com` or `github.com` would be flagged only because it has no `www`, while `www.google.com` would look safe. `normalize_url` deletes a leading `www.` before counting features, and training drops duplicate hosts after that. `www.x.com` and `x.com` are scored as the same host. This is a cleaning rule, not a hardcoded list of safe websites.

## The 12 features

Same function in training and in FastAPI. No DNS, no page download, no HTML.

- **URL length** (`url_length`). Total number of characters in the normalized URL. Some phishing links are very long because they hide the real destination or add extra tracking text.
- **Number of dots** (`dot_count`). How many '.' characters appear in the full URL string. Extra dots can come from extra host labels or from a long, messy link.
- **Number of hyphens** (`hyphen_count`). How many '-' characters appear in the full URL string. Hyphens are sometimes used to imitate a real brand name inside a hostname or path.
- **Number of digits** (`digit_count`). How many numeric characters appear in the URL. Automatically generated phishing links often contain many digits.
- **Number of @ symbols** (`at_count`). How many '@' characters appear in the URL. An @ symbol can make the real hostname harder to notice, because the browser uses the host after the @.
- **Number of subdomains** (`subdomain_count`). Extra hostname labels beyond the last two labels, after a leading www is removed. example.com and www.example.com both count as 0. login.example.com counts as 1. This is a simple approximation and does not understand suffixes such as co.uk. IP addresses count as 0. www is removed because every legitimate row in this dataset starts with www. Extra subdomains are sometimes used to imitate a trusted brand.
- **HTTPS usage** (`uses_https`). 1 if the scheme is https, otherwise 0. A missing scheme is treated as https before this check. The scheme is one signal. HTTPS encrypts traffic, but phishing websites can use HTTPS too, so it is not proof of safety.
- **IP address usage** (`is_ip_address`). 1 if the hostname is an IPv4 or IPv6 address according to Python's ipaddress module, otherwise 0. Most public websites use a domain name. A raw IP address can be suspicious, though it is not always malicious.
- **Suspicious keywords** (`suspicious_keyword_count`). Total substring count of these words, matched without regard to case: login, verify, update, secure, account, confirm, password. A word inside a longer word can also match. These are weak indicators, not proof. Words such as login, verify, and password appear in many phishing lures. They also appear on real login pages.
- **Number of question marks** (`question_mark_count`). How many '?' characters appear in the URL. A question mark usually starts a query string, which can carry extra parameters.
- **Number of equals signs** (`equals_count`). How many '=' characters appear in the URL. Equals signs usually join a query parameter to its value.
- **URL path length** (`path_length`). Number of characters in the URL path returned by urllib.parse, including the leading slash when one is present. A very long path can be used to bury the real destination of a link.

## The three algorithms

All three see the same training rows and are compared on the same validation rows.

**Logistic Regression** is the linear baseline. It learns a weight for each feature, adds them, and turns the sum into a probability with the logistic function. Settings: `StandardScaler` inside a Pipeline, `solver=lbfgs`, `max_iter=1000`, `random_state=42`. Scaling matters because URL length is much larger than a 0/1 feature. The scaler is fit on training rows only.

**K-Nearest Neighbors** stores the training examples. For a new URL it finds the 5 nearest rows and lets them vote. Settings: `n_neighbors=5` and `StandardScaler` in a Pipeline. The phishing score can only be 0, 0.2, 0.4, 0.6, 0.8, or 1, because it is a vote of five neighbors.

**Random Forest** builds 100 decision trees and averages their votes. A tree asks yes/no questions such as whether URL length is above a threshold. Settings: `n_estimators=100`, `max_depth=20`, `random_state=42`, `n_jobs=-1`. Trees do not use StandardScaler. The depth cap was set in the script before the test set was scored.

## How the best model was chosen

Validation chooses the model. The test set is scored once, after that choice, and is not used to change the model.

1. Main score: phishing-class F1.
2. If the top F1 scores are within 0.005, use phishing recall.
3. If recall is also within 0.01, keep the simpler model. Simpler means Logistic Regression, then KNN, then Random Forest.

Validation results:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9656 | 0.9873 | 0.9313 | 0.9585 |
| K-Nearest Neighbors | 0.9889 | 0.9947 | 0.9792 | 0.9869 |
| Random Forest | 0.9936 | 0.9984 | 0.9865 | 0.9924 |

**Selected model: Random Forest.** Random Forest had the highest validation F1-score for the phishing class (0.9924).

Its validation scores: accuracy 0.9936, precision 0.9984, recall 0.9865, F1 0.9924.

## Held-out test result

These numbers are for the 4,500 test rows inside the 30,000-row sample. They are not a test of all 235,795 original records.

| Metric | Score |
|---|---:|
| Accuracy | 0.9964 |
| Precision | 0.9984 |
| Recall | 0.9932 |
| F1 | 0.9958 |

Confusion matrix. Rows are the true class. Columns are the prediction. Order is SAFE, then PHISHING.

|  | Predicted SAFE | Predicted PHISHING |
|---|---:|---:|
| Actual SAFE | 2,577 true negatives | 3 false positives |
| Actual PHISHING | 13 false negatives | 1,907 true positives |

Say it this way: of 4,500 new URLs, 4,484 were correct. The model missed 13 phishing URLs and flagged 3 legitimate URLs.

Formulas, with phishing as the positive class:

- Accuracy = correct predictions / total predictions
- Precision = TP / (TP + FP) = 1,907 / (1,907 + 3)
- Recall = TP / (TP + FN) = 1,907 / (1,907 + 13)
- F1 = 2 x precision x recall / (precision + recall)

Recall matters because a false negative is a phishing URL called safe. Accuracy alone can hide that.

`predict` returns 0 or 1. `predict_proba` returns a score for each class. FastAPI shows the class-1 score as the estimated phishing probability. For Random Forest that score is the fraction of trees that vote phishing. It is not a calibrated guarantee.

## API routes

- `GET /api/health` returns `{ "status": "ok" }`.
- `POST /api/predict` accepts `{ "url": "https://example.com" }` and returns the class, phishing score, model name, reasons, and measured traits.
- `GET /api/model-info` returns the saved metrics, including the validation table and test scores.
- `GET /api/confusion-matrix` and `GET /api/comparison-chart` return the PNG files written by training.
- `GET /docs` is the FastAPI page that lists these routes.

## What to say if a familiar site looks wrong

The model only sees URL structure. It does not know brand names. HTTPS is one feature, not proof of safety. Dataset labels are historical. A random split of URLs can still put related domains in both training and test, so the score can look optimistic. Do not treat the result as a security guarantee.

## Run it

Python 3.11 and Node.js. Internet is needed the first time, for the dataset download and `npm install`. After training, prediction works offline.

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_data.py
python train.py
python app.py
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. The API is http://127.0.0.1:5000. If port 5000 is busy, run `PORT=5050 python app.py` and set `NEXT_PUBLIC_API_URL=http://127.0.0.1:5050` in `frontend/.env.local`. If port 3000 is busy, run `npm run dev -- -p 3010`.

Tests, from `backend` after training:

```bash
python -m unittest discover -s tests -v
```

The prediction test calls the saved model directly and checks that the API returns the same class and the same phishing score.

## Limits to mention

- The model can be wrong. This test run has 3 false positives and 13 false negatives.
- HTTPS does not prove a site is legitimate.
- Labels can be out of date.
- Only the URL string is analyzed.
- Related domains can appear on both sides of a random split.
- The probability is a model score, not a certified risk.

## Other write-ups

- [DATASET.md](DATASET.md)
- [MODEL_REPORT.md](MODEL_REPORT.md)
- [VIVA.md](VIVA.md)
