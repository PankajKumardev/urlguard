# URLGuard Model Report

Jaypee Institute of Information Technology, Noida

Department of Computer Applications

Machine Learning Lab · MCA, 1st Year, 1st Semester

These numbers were written by `train.py` from the real evaluation files in `backend/results`. They were not typed in by hand.

Generated: 2026-11-04 04:59:46 UTC

Python 3.11.15 · scikit-learn 1.9.1 · pandas 3.1.0 · NumPy 2.4.6

Random seed: 42

## A. Project Introduction

Phishing is an attempt to trick someone into opening a dishonest link, often so that a password, payment detail, or personal information can be stolen. A phishing URL is the address used in that attempt. It may imitate a real brand, hide the destination in a long path, or use a raw IP address instead of a normal domain name.

A person can learn a few warning signs, but there are too many URLs to inspect one by one. Machine learning helps by learning patterns from a large set of addresses that have already been labeled. The program does not understand a website the way a person does. It only sees the numeric features we calculate from the URL text.

URLGuard is a college lab project that demonstrates this idea. A user pastes a URL into a website. The FastAPI backend turns that text into the same features used in training and sends them to a saved scikit-learn model. The page then shows SAFE or PHISHING, an estimated phishing probability, and a short description of the URL traits that were measured.

The result is an educational risk estimate. It is not a promise that a website is safe, and it is not a professional security product.

## B. Dataset Information

- Official dataset name: PhiUSIIL Phishing URL (Website)
- Authors: Arvind Prasad and Shalini Chandra
- Official UCI page: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- Dataset citation: Prasad, A. & Chandra, S. (2024). PhiUSIIL Phishing URL (Website) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.1016/j.cose.2023.103545.
- Introductory paper: Prasad, A., & Chandra, S. (2023). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. Computers & Security, 103545. https://doi.org/10.1016/j.cose.2023.103545
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- License URL: https://creativecommons.org/licenses/by/4.0/
- UCI id: 967
- Download method: ucimlrepo fetch_ucirepo(id=967)
- SHA-256 of `backend/data/urls.csv`: `0112072734d5e58a0c319e82f3aea97cb6ae2803c21a32d1d52446afa382c40d`

The dataset is reliable for this lab because it comes from the UCI Machine Learning Repository, it has a published paper, a named author team, a Creative Commons license, and a documented class mapping. The file downloaded for this project contains 235,795 rows. That matches the published size of 235,795 rows, with 134,850 legitimate URLs and 100,945 phishing URLs.

On the original UCI scale, label 1 is legitimate and label 0 is phishing. The official sentence is: "Label 1 corresponds to a legitimate URL, label 0 to a phishing URL." Confirmation used for this run: The official UCI dataset page states 'Label 1 corresponds to a legitimate URL, label 0 to a phishing URL'. The downloaded file matches the published size: 134850 rows with label 1 and 100945 rows with label 0.

In the downloaded file, before cleaning, label 1 appeared 134,850 times and label 0 appeared 100,945 times.

After cleaning there were 235,224 usable rows: 134,850 safe (legitimate) and 100,374 phishing, using the project labels.

Modeling used 30,000 real rows from that cleaned file. This was a stratified subset chosen with `random_state = 42`. It was not the full dataset, and no synthetic URLs were added. In the subset there were 17,199 safe URLs and 12,801 phishing URLs.

The labels are historical. A URL that was legitimate when the dataset was built might later be abandoned, and a site that was phishing might later be taken down. The model also never opens the page, so it cannot see what the website does today.

## C. Data Preprocessing

Cleaning was done with ordinary Pandas steps in `train.py`.

1. Load `data/urls.csv`, which stores the URL and the original UCI label.
2. Remove rows with a missing URL or a missing label. Removed: 0.
3. Remove rows whose label is not 0 or 1. Removed: 0.
4. Remove URLs that `features.normalize_url` cannot parse as HTTP or HTTPS. Removed: 0.
5. Combined invalid removals (missing, bad label, or malformed URL): 0.
6. Remove every row whose URL appears with more than one label. Conflicting URLs: 0. Rows removed: 0. This is done before duplicate removal so that a conflict cannot be hidden by keeping an arbitrary copy.
7. Remove extra copies of the same URL string. Duplicate count (extra copies removed): 571.
8. Convert the label with `phishing_label = 1 - original_label`. After this step, 0 means SAFE and 1 means PHISHING.

| Item | Count |
|---|---:|
| Original rows | 235,795 |
| Removed invalid rows | 0 |
| Duplicate extras removed | 571 |
| Conflicting-label rows removed | 0 |
| Final cleaned rows | 235,224 |
| Legitimate / SAFE rows after conversion | 134,850 |
| Phishing rows after conversion | 100,374 |
| Rows actually used for modeling | 30,000 |
| Training rows (70%) | 21,000 |
| Validation rows (15%) | 4,500 |
| Test rows (15%) | 4,500 |

Class counts inside the modeling sample:

| Split | SAFE (0) | PHISHING (1) | Total |
|---|---:|---:|---:|
| Modeling sample | 17,199 | 12,801 | 30,000 |
| Training | 12,039 | 8,961 | 21,000 |
| Validation | 2,580 | 1,920 | 4,500 |
| Test | 2,580 | 1,920 | 4,500 |

The same cleaned sample was split once. Training, validation, and test URLs do not overlap. Identical-URL overlap count: 0. A random split still does not stop related domains, such as two different pages on the same website, from landing in different splits. That can make the score look a little more optimistic than a split by domain would.

Feature extraction uses `extract_features` from `features.py`. The original dataset also contains features taken from webpage source code. Those columns were not used, because the website will only receive a URL string. The longest cleaned URL had 6,097 characters. The API refuses pastes longer than 2,048 characters. Cleaned URLs longer than that limit: 8. They can remain in training, but the website will not score a paste that long.

One important cleaning choice: a leading `www.` label is removed before features are counted, and duplicate hosts are removed after that. In the downloaded file, every legitimate URL starts with `www.` and many phishing URLs do not. If that prefix were left in, the model would treat `www` as proof of safety. Then a real site such as `x.com` or `github.com` would be scored as phishing only because it has no `www`. After this step, `www.example.com` and `example.com` are the same training host.

This report does not claim that cleaning raised accuracy. Cleaning was done so that missing, broken, repeated, and contradictory rows were not used. Accuracy before and after cleaning was not compared.

## D. Feature Engineering

Every model sees the same 12 numbers, in the order below. Keyword matching is a simple lowercase substring count of: login, verify, update, secure, account, confirm, password. It can match inside a longer word. That limitation is accepted so the code stays easy to explain.

If a user omits the scheme, `example.com` is treated as `https://example.com` before the features are counted. An unsupported scheme is rejected. The code never connects to the host.

### url_length — URL length

How it is calculated: Total number of characters in the normalized URL.

Why it may help: Some phishing links are very long because they hide the real destination or add extra tracking text.

### dot_count — Number of dots

How it is calculated: How many '.' characters appear in the full URL string.

Why it may help: Extra dots can come from extra host labels or from a long, messy link.

### hyphen_count — Number of hyphens

How it is calculated: How many '-' characters appear in the full URL string.

Why it may help: Hyphens are sometimes used to imitate a real brand name inside a hostname or path.

### digit_count — Number of digits

How it is calculated: How many numeric characters appear in the URL.

Why it may help: Automatically generated phishing links often contain many digits.

### at_count — Number of @ symbols

How it is calculated: How many '@' characters appear in the URL.

Why it may help: An @ symbol can make the real hostname harder to notice, because the browser uses the host after the @.

### subdomain_count — Number of subdomains

How it is calculated: Extra hostname labels beyond the last two labels, after a leading www is removed. example.com and www.example.com both count as 0. login.example.com counts as 1. This is a simple approximation and does not understand suffixes such as co.uk. IP addresses count as 0. www is removed because every legitimate row in this dataset starts with www.

Why it may help: Extra subdomains are sometimes used to imitate a trusted brand.

### uses_https — HTTPS usage

How it is calculated: 1 if the scheme is https, otherwise 0. A missing scheme is treated as https before this check.

Why it may help: The scheme is one signal. HTTPS encrypts traffic, but phishing websites can use HTTPS too, so it is not proof of safety.

### is_ip_address — IP address usage

How it is calculated: 1 if the hostname is an IPv4 or IPv6 address according to Python's ipaddress module, otherwise 0.

Why it may help: Most public websites use a domain name. A raw IP address can be suspicious, though it is not always malicious.

### suspicious_keyword_count — Suspicious keywords

How it is calculated: Total substring count of these words, matched without regard to case: login, verify, update, secure, account, confirm, password. A word inside a longer word can also match. These are weak indicators, not proof.

Why it may help: Words such as login, verify, and password appear in many phishing lures. They also appear on real login pages.

### question_mark_count — Number of question marks

How it is calculated: How many '?' characters appear in the URL.

Why it may help: A question mark usually starts a query string, which can carry extra parameters.

### equals_count — Number of equals signs

How it is calculated: How many '=' characters appear in the URL.

Why it may help: Equals signs usually join a query parameter to its value.

### path_length — URL path length

How it is calculated: Number of characters in the URL path returned by urllib.parse, including the leading slash when one is present.

Why it may help: A very long path can be used to bury the real destination of a link.

## E. Algorithms Used

All three models were trained on the same training rows and compared on the same validation rows. PHISHING (1) is the positive class for precision, recall, and F1.

### Logistic Regression

Logistic Regression is a linear model for binary classification. It multiplies each feature by a learned weight, adds those products, and passes the total through a logistic function. The output is a probability between 0 and 1. If the phishing probability is higher than the safe probability, the predicted class is PHISHING.

It was included as the simple baseline. If a straight-line model is already strong, a more complicated model has to beat it by enough to be worth the extra complexity.

Advantages: it is fast, the prediction is a probability, and the idea is easy to explain. Limitations: it only learns a linear boundary, so it can miss patterns that depend on combinations of features. Features are on very different scales, so this model is placed in a Pipeline with `StandardScaler`. The scaler is fit on the training rows only.

Hyperparameters:

- scaler: StandardScaler inside a Pipeline, fit on training data only
- solver: lbfgs
- max_iter: 1000
- random_state: 42

### K-Nearest Neighbors

K-Nearest Neighbors does not learn a formula with weights. It stores the training examples. For a new URL, it finds the K closest training rows in feature space and lets those neighbors vote. Here K is 5, so the five nearest examples vote. The phishing probability is the fraction of those five neighbors that are phishing. That score is not a carefully calibrated risk.

Scaling matters because KNN uses distance. URL length can be 20 or 200, while HTTPS is only 0 or 1. Without scaling, length would dominate the distance. `StandardScaler` is therefore used in a Pipeline, again fit on training rows only.

Advantages: the idea is intuitive, and it can follow a non-linear boundary. Limitations: prediction gets slower as the training set grows, and the result depends on K and on how the features are scaled. That is why this project uses 30,000 records rather than all 235,795 for the comparison.

Hyperparameters:

- scaler: StandardScaler inside a Pipeline, fit on training data only
- n_neighbors: 5

### Random Forest

A decision tree asks a series of yes/no questions, such as whether URL length is above a threshold. Each question sends the example left or right until it reaches a leaf, and the leaf votes for a class. A single tree can memorize the training rows.

A Random Forest builds many trees. Each tree sees a bootstrap sample of the training rows and considers a random subset of features at each split. The forest averages the trees. This usually lowers variance compared with one tree. The phishing probability is the fraction of trees that vote for phishing.

Advantages: it handles non-linear patterns and does not need scaled features. Limitations: it is harder to explain than one weight per feature, it can still overfit, and a large forest uses more memory. `max_depth` is capped at 20. That cap was chosen in the script before the test set was scored. Many depths were not searched.

Hyperparameters:

- n_estimators: 100
- max_depth: 20
- random_state: 42
- n_jobs: -1
- scaler: None. Trees do not need scaled features.

## F. Model Training

X is the matrix of URL features. Each row of X is one URL, and each column is one feature from `FEATURE_NAMES`. y is the label vector. After conversion, y is 0 for SAFE and 1 for PHISHING.

The modeling sample has 30,000 rows. It was split with `train_test_split`, stratified so each split keeps a similar mix of safe and phishing URLs.

- Training: 21,000 rows (12,039 safe, 8,961 phishing). The models are fit on these rows only.
- Validation: 4,500 rows (2,580 safe, 1,920 phishing). This split chooses the model.
- Test: 4,500 rows (2,580 safe, 1,920 phishing). This split is scored once, after the choice is fixed.

`random_state = 42` makes the sample and the split repeatable. The same seed does not make the model correct. It only means that running the script again should give the same split.

Training and testing use different records so the score measures URLs the model did not fit on. If the same rows were used for both, the model could look strong simply because it had memorized them. The validation split is separate from the test split so that choosing the model does not quietly reuse the final exam. Overlap of identical URLs across the three splits: 0.

Logistic Regression and KNN scale features inside a Pipeline. Random Forest uses the raw numbers. The saved file is the Random Forest itself. Its features are not scaled, because a tree compares each feature with a threshold.

## G. Model Comparison

The table uses validation predictions only. PHISHING is the positive class. These scores are for the 30,000-row subset, not for the full UCI file.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9656 | 0.9873 | 0.9313 | 0.9585 |
| K-Nearest Neighbors | 0.9889 | 0.9947 | 0.9792 | 0.9869 |
| Random Forest | 0.9936 | 0.9984 | 0.9865 | 0.9924 |

The highest validation accuracy was 0.9936 (99.36%), from Random Forest.

The highest validation precision was 0.9984 (99.84%), from Random Forest.

The highest validation recall was 0.9865 (98.65%), from Random Forest.

The highest validation F1-score was 0.9924 (99.24%), from Random Forest.

Precision and recall pull in different directions. A model can catch more phishing URLs (higher recall) while also flagging more legitimate URLs (lower precision). F1 is the harmonic mean of precision and recall, so it falls when either one is weak. Accuracy alone can hide a weak phishing recall, which is why it was not the selection rule.

The chart of these validation scores is saved at `backend/results/model_comparison.png`. The axis starts at 0 so small gaps are not exaggerated. The table in `backend/results/model_comparison.csv` holds the same validation numbers.

## H. Best Model Selected

**BEST PERFORMING MODEL: Random Forest**

- Validation F1-score: 0.9924 (99.24%)
- Validation recall: 0.9865 (98.65%)
- Validation precision: 0.9984 (99.84%)
- Validation accuracy: 0.9936 (99.36%)

Why this model was selected: Random Forest had the highest validation F1-score for the phishing class (0.9924).

The rule was fixed before the test set was scored. The main score is validation F1 for the phishing class. If the best F1 values are within 0.005, phishing recall breaks the tie. If recall is also within 0.01, the simpler model is kept. Simpler means Logistic Regression, then K-Nearest Neighbors, then Random Forest. The test set was not consulted during this choice.

## I. Final Test Results

The selected model was evaluated once on the 4,500 held-out test URLs. The model was not changed after these numbers were seen.

| Metric | Score |
|---|---:|
| Accuracy | 0.9964 |
| Precision | 0.9984 |
| Recall | 0.9932 |
| F1-score | 0.9958 |

Confusion matrix for the test set. Rows are the true class and columns are the predicted class. Order is SAFE (0), then PHISHING (1).

|  | Predicted SAFE | Predicted PHISHING |
|---|---:|---:|
| Actual SAFE | 2,577 (TN) | 3 (FP) |
| Actual PHISHING | 13 (FN) | 1,907 (TP) |

- True positives: 1,907. Phishing URLs correctly flagged.
- True negatives: 2,577. Legitimate URLs correctly left as safe.
- False positives: 3. Legitimate URLs incorrectly flagged as phishing.
- False negatives: 13. Phishing URLs incorrectly called safe.

In plain language, the model saw 4,500 new URLs. It labeled 4,484 of them correctly. It missed 13 phishing URLs and raised 3 false alarms. From the counts, accuracy = (2,577 + 1,907) / 4,500 = 0.9964. Precision = 1,907 / (1,907 + 3) = 0.9984. Recall = 1,907 / (1,907 + 13) = 0.9932. F1 from those two values is 0.9958. The scikit-learn scores recorded for this run are accuracy 0.9964, precision 0.9984, recall 0.9932, and F1 0.9958.

The picture of this matrix is `backend/results/confusion_matrix.png`.

These test scores describe the held-out portion of the 30,000-row sample. They are not an evaluation of all 235,795 original records.

The scores are high. Many URLs in this dataset are easy to separate with simple structural counts, and a random split can still put related domains on both sides. That can make the result look stronger than the model would be on brand-new, unrelated websites. The number is reported as measured. It is not a claim of real-world security performance.

## J. Evaluation Metrics Explained

Accuracy = Correct predictions / Total predictions

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 × Precision × Recall / (Precision + Recall)

Here the positive class is PHISHING. Precision answers: when the model says phishing, how often is it right? Recall answers: of the phishing URLs, how many did the model catch? A missed phishing URL is a false negative. That is why recall matters in this project. Accuracy can look high even when many phishing URLs are missed, especially if one class is much larger than the other. This dataset is not extremely unbalanced, but accuracy is still not the selection rule. F1 requires both precision and recall to be reasonably strong.

## K. Prediction Example

This example is one real row from the held-out test set. It was not invented.

The true class is phishing, so the live URL is redacted. Do not try to reconstruct it. The features below were calculated on the original string before redaction.

- Displayed URL: `http://55****om/[redacted-path]`
- True dataset label: PHISHING
- Predicted label: PHISHING
- Estimated phishing probability: 1.0000
- Prediction was correct: yes

| Feature | Value |
|---|---:|
| URL length | 17 |
| Number of dots | 1 |
| Number of hyphens | 0 |
| Number of digits | 2 |
| Number of @ symbols | 0 |
| Number of subdomains | 0 |
| HTTPS usage | 0 |
| IP address usage | 0 |
| Suspicious keywords | 0 |
| Number of question marks | 0 |
| Number of equals signs | 0 |
| URL path length | 0 |

The original phishing URL is not printed. Features were calculated on the full URL before redaction.

The probability was read from the column for class 1 in `predict_proba`, not from a hardcoded index.

## L. Limitations

- The model can be wrong. The test set itself contains 3 false positives and 13 false negatives.
- HTTPS does not guarantee legitimacy. Phishing sites can have certificates. The HTTPS feature is only one of the 12 inputs.
- Dataset labels are historical and may not describe a URL's current state.
- The system only analyzes the URL string. It does not read the page, check DNS, inspect certificates, or follow redirects.
- Related domains can appear on both sides of a random URL split, so the score can be optimistic compared with a domain-wise split.
- The estimated probability is a model score. KNN reports a neighbor vote and Random Forest reports a tree vote. Neither score is presented as a calibrated real-world probability.
- Keyword counts are simple substrings, and the subdomain count treats the last two hostname labels as the domain. Suffixes such as `.co.uk` are not handled specially.
- The model cannot guarantee that a website is safe. URLGuard is a lab demonstration.

## M. Conclusion

The project used the public PhiUSIIL dataset, cleaned the URL and label columns, and built 12 features that can be computed from a URL string alone. Logistic Regression, K-Nearest Neighbors, and Random Forest were trained on the same 21,000 rows and compared on the same 4,500 validation rows.

The selected model is **Random Forest**. On the validation split its phishing F1 was 0.9924 and its phishing recall was 0.9865. On the untouched test split of 4,500 URLs, accuracy was 0.9964, precision was 0.9984, recall was 0.9932, and F1 was 0.9958.

The main lesson is the experimental habit: use real data, keep training and testing apart, compare models on validation data, and report the test score only after the choice is fixed. A high score on URL structure is still not a security guarantee.
