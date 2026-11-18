# Viva Preparation

Jaypee Institute of Information Technology, Noida

Computer Applications · Machine Learning Lab · MCA, 1st Year, 1st Semester

Short answers for the URLGuard lab. Numbers come from the training run saved in `backend/results/metrics.json` (2026-11-04 04:59:46 UTC). Speak them in your own words.

## Questions

### 1. What is machine learning?

Machine learning is a way of building a program that learns a pattern from examples instead of being given a fixed rule for every case. In this project the examples are URLs with labels, and the program learns how to label a new URL.

### 2. What is supervised learning?

Supervised learning uses examples that already have answers. Here the answer is the label: safe or phishing. The model sees both the features and the label while it trains. It is then tested on rows whose labels it did not use for fitting.

### 3. What is classification?

Classification means predicting a category. URLGuard predicts a category for a URL. It does not predict a price or a temperature.

### 4. What is binary classification?

Binary classification has exactly two classes. This project has SAFE and PHISHING. In the project coding, SAFE is 0 and PHISHING is 1.

### 5. What is phishing?

Phishing is an attempt to make a person trust a dishonest message or link. A phishing URL is the address used in that attempt. It may copy a brand name, add a login path, or hide the real host.

### 6. What is the purpose of URLGuard?

URLGuard is a Machine Learning Lab project that shows the full path from a real dataset to a small website. The user pastes a URL, the backend extracts features, and a saved model estimates whether the URL looks legitimate or phishing. It is a teaching tool, not a security guarantee.

### 7. Which dataset did you use?

I used the PhiUSIIL Phishing URL Dataset from the UCI Machine Learning Repository, dataset id 967. The authors are Arvind Prasad and Shalini Chandra. The page is https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset.

### 8. Why did you choose this dataset?

It is a public academic dataset with a paper, a license (CC BY 4.0), real URLs, and a documented label. It is large enough to train and test a classifier. I can also defend where every row came from.

### 9. How many records are in the dataset?

The downloaded file has 235,795 rows, matching the published 235,795. After cleaning, 235,224 rows remained. Training and evaluation used a stratified subset of 30,000 real rows: 21,000 train, 4,500 validation, and 4,500 test. I do not claim that the scores were measured on all 235,795 rows.

### 10. What do the labels 0 and 1 represent?

In the original UCI file, 1 means legitimate and 0 means phishing. The UCI page says: "Label 1 corresponds to a legitimate URL, label 0 to a phishing URL." In this project I convert with `phishing_label = 1 - original_label`. After that, 0 means SAFE and 1 means PHISHING. The positive class for precision, recall, and F1 is PHISHING.

### 11. What is data preprocessing?

Preprocessing means preparing the table before training. I removed missing values, bad labels, URLs that could not be parsed, URLs with conflicting labels, and duplicate URL strings. Then I converted the label and extracted features. Removed invalid rows: 0. Duplicate extras removed: 571. Conflicting rows removed: 0.

### 12. Why do we remove duplicates?

If the same URL is copied in both the training split and the test split, the test is too easy. The model may simply remember that exact address. I drop extra copies before the split. I also delete a URL completely when its copies disagree about the label, because keeping one copy at random would teach the wrong answer.

### 13. What is feature extraction?

Feature extraction turns raw input into numbers a model can use. The raw input here is a URL string. `features_from_normalized` returns 12 numbers, such as length, dot count, and whether the host is an IP address. The same function is used for training and for a live prediction.

### 14. What are the input features?

The features are: url_length, dot_count, hyphen_count, digit_count, at_count, subdomain_count, uses_https, is_ip_address, suspicious_keyword_count, question_mark_count, equals_count, path_length. There are 12 of them. They are all calculated from the URL text. I do not use the dataset's webpage-content features, because the website never opens the page. A leading www. is removed first. Every legitimate URL in the raw file starts with www, so keeping it would make the model call any bare domain phishing.

### 15. What are X and y?

X is the table of features. One row is one URL and one column is one feature. y is the label for each row: 0 for SAFE and 1 for PHISHING. In `train.py` the full sample is stored in the variables `X` and `y`, and then split into `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, and `y_test`.

### 16. What is train-test splitting?

The rows are divided so the model is fit on one part and scored on another part. I use 70% for training, 15% for validation, and 15% for testing. The split is stratified, so each part keeps a similar proportion of phishing URLs. In this run that is 21,000, 4,500, and 4,500 rows.

### 17. Why do we need validation data?

Validation data is used to compare Logistic Regression, KNN, and Random Forest. If I compared them on the test set and then kept the winner, the test set would have helped me choose, and the final number would be less honest. I choose on validation. I touch the test set only once, after the choice is fixed.

### 18. What does random_state=42 mean?

It is the seed for the random sample and the random split. Seed 42 was chosen in advance and written in the script. It does not mean the model is better. It means another person can run the script and get the same split. The recorded seed for this run is 42.

### 19. What is Logistic Regression?

It is a linear classifier. It learns a weight for each feature, adds the weighted features, and turns the sum into a probability with the logistic function. I use it as the simple baseline, with `max_iter=1000`, `solver='lbfgs'`, `random_state=42`, and a `StandardScaler` inside a Pipeline.

### 20. What is KNN?

K-Nearest Neighbors stores the training examples. For a new URL it finds the 5 closest examples and lets them vote. K is 5. The features are scaled, because distance would otherwise be dominated by large numbers such as URL length.

### 21. What is Random Forest?

A decision tree asks yes/no questions about the features. A Random Forest builds 100 such trees on random samples of the training data and averages their votes. I set `n_estimators=100`, `max_depth=20`, `random_state=42`, and `n_jobs=-1`. Trees do not need scaled features.

### 22. What is overfitting?

Overfitting means the model fits the training rows too closely, including their accidents, and then does worse on new rows. A very deep tree can overfit. I limit the forest depth to 20, and I judge the models on validation data rather than on the training score.

### 23. What is underfitting?

Underfitting means the model is too simple to learn the real pattern, so it does poorly even where it should be able to do well. A useful check is a weak training score and a weak validation score together. Logistic Regression can underfit if the class boundary is not close to a straight line in these 12 features.

### 24. What is StandardScaler?

StandardScaler subtracts the training mean of each feature and divides by the training standard deviation. After scaling, a typical value is near 0 and the spread is near 1. I use it for Logistic Regression and KNN so that URL length does not overwhelm a 0/1 feature. The scaler is fit on training rows only, inside a Pipeline, so validation and test rows do not leak into the mean and standard deviation. Random Forest skips it.

### 25. What is accuracy?

Accuracy = correct predictions / total predictions. On the held-out test split, Random Forest had accuracy 0.9964 (99.64%). That is 4,484 correct out of 4,500 test URLs.

### 26. What is precision?

Precision = TP / (TP + FP). It is the fraction of phishing alarms that were truly phishing. Test precision was 0.9984. From the counts, TP = 1,907 and FP = 3.

### 27. What is recall?

Recall = TP / (TP + FN). It is the fraction of real phishing URLs that the model caught. Test recall was 0.9932. TP = 1,907 and FN = 13. Recall matters because a false negative means a phishing URL was called safe.

### 28. What is F1-score?

F1 = 2 × precision × recall / (precision + recall). It is high only when both precision and recall are reasonably high. Validation F1 was the main rule for choosing the model. The selected model's validation F1 was 0.9924. Its test F1 was 0.9958.

### 29. What is a confusion matrix?

It is a 2-by-2 table of actual class against predicted class. For the test set, with rows actual SAFE then actual PHISHING, and columns predicted SAFE then predicted PHISHING, the counts are TN 2,577, FP 3, FN 13, TP 1,907. The picture is `backend/results/confusion_matrix.png`.

### 30. What are TP, TN, FP and FN?

TP, true positive: phishing correctly flagged (1,907). TN, true negative: legitimate correctly called safe (2,577). FP, false positive: legitimate incorrectly flagged (3). FN, false negative: phishing incorrectly called safe (13).

### 31. Which model performed best in your project?

The selected model is Random Forest. This was decided from the validation scores, not chosen in advance.

### 32. Why did that model perform best according to your results?

Random Forest had the highest validation F1-score for the phishing class (0.9924). Its validation F1 was 0.9924 and its validation recall was 0.9865. If two F1 scores are within 0.005, I use phishing recall. If recall is also within 0.01, I keep the simpler model.

### 33. What accuracy did you actually achieve?

On the held-out test split of the 30,000-row sample, Random Forest accuracy was 0.9964 (99.64%). Test precision was 0.9984, test recall was 0.9932, and test F1 was 0.9958. The validation accuracy of this same model was 0.9936. These are not full-dataset scores.

### 34. How is phishing probability calculated?

The API calls `predict_proba` and reads the column whose class label is 1. It does not assume that column 1 is phishing without checking `model.classes_`. For Logistic Regression the number is the logistic function of the weighted features. For KNN it is the fraction of the 5 neighbors that are phishing. For Random Forest it is the fraction of trees that vote phishing. The number shown in the website is that score. It is an estimate, not a calibrated guarantee.

### 35. What does predict() do?

`predict` returns the class. In this project that is 0 or 1, which the API turns into the words SAFE or PHISHING.

### 36. What does predict_proba() do?

`predict_proba` returns a probability for each class. The two values sum to 1. I use the value attached to class 1 as the estimated phishing probability.

### 37. What is Joblib?

Joblib saves a Python object to disk and loads it back. I save the fitted model, its name, and the feature order in `backend/models/best_model.joblib`. FastAPI loads that file when the server starts. I do not download a model from the internet.

### 38. How does FastAPI use the trained model?

When `app.py` starts, `load_artifacts` reads the joblib file once. A request to `POST /api/predict` checks the URL, calls `features_from_normalized`, builds the 12 numbers in the saved order, then calls `predict` and `predict_proba`. It does not call `fit` again. If I train a new model, I have to restart the API so it loads the new file.

### 39. Why can a legitimate URL be incorrectly classified?

A legitimate URL can share traits with phishing URLs in this feature set. It might be long, contain a hyphen, use a keyword such as `login`, or have several subdomains. The model only sees those 12 numbers, so different URLs can land in the same region of feature space. On the test set this showed up as 3 false positives.

### 40. What are the limitations of the system?

It can be wrong. HTTPS is not proof of safety. Labels can be out of date. Only the URL string is analyzed. A random split can place related domains on both sides. The probability is a model score, not a certified risk. The result must not be treated as a security guarantee.

### 41. Why did you not use the original 54 features?

Many of those features come from the HTML of the page. The live application only receives a URL, and it is not allowed to open that URL. A model trained on page features could not be applied to the text the user pastes. I therefore build features that training and prediction can both compute.

### 42. Why was the test set not used to choose the model?

The test set is the final check. If I looked at it, changed the model, and looked again, I would be tuning on the test set. The number would then describe a set I had already peeked at. Validation is for the choice. The test score is reported once.

### 43. Why did you use about 30,000 records?

The official file has 235,795 rows. KNN must compare a new point with the training points, so the full file would make the lab slow on a normal laptop. A stratified sample of 30,000 real rows keeps the class mix and finishes in reasonable time. The report states that the scores belong to this sample.

### 44. Does HTTPS mean a website is safe?

No. HTTPS means the connection is encrypted. A phishing site can still use HTTPS. In the feature list, `uses_https` is one signal among 12. The result page says the same thing.

### 45. Does the server retrain the model for every URL?

No. Training is `python train.py`. Prediction is `python app.py`. The server loads the saved file at startup and only calls `predict` and `predict_proba` after that.

## Important Formulas

Accuracy = Correct predictions / Total predictions

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 × Precision × Recall / (Precision + Recall)

Project label conversion: phishing_label = 1 - original_label

Worked test counts for Random Forest: TP = 1,907, TN = 2,577, FP = 3, FN = 13.

## Algorithms in Simple Language

### Logistic Regression

1. Each feature gets a weight.
2. Multiply each feature by its weight and add them up.
3. The logistic function turns that sum into a probability between 0 and 1.
4. The class with the higher probability wins.
5. Scaling is needed because the features have different units.
6. It is fast and easy to explain, but the boundary is linear.
7. In this project it is the simplest of the three models.

### K-Nearest Neighbors

1. Store the training examples.
2. K is 5 in this project.
3. For a new URL, find the 5 nearest training rows.
4. Let those neighbors vote.
5. The phishing probability is the share of neighbors that are phishing.
6. Scale the features first, or long URLs will dominate the distance.
7. Prediction gets slower when the training set is huge.

### Random Forest

1. A decision tree asks a chain of yes/no questions.
2. One tree can memorize the training data.
3. A forest grows many trees, here 100.
4. Each tree sees a random sample of rows and a random subset of features.
5. The trees vote, and the average is the forest's score.
6. Depth is limited to 20 so a tree cannot grow without bound.
7. Scaling is not required for the trees.

## Feature Extraction Walkthrough

This is a parsing demonstration, not a training sample and not a verified safety classification. The address `https://example.com/login?user=1` was passed to `extract_features` while the report was written. No model prediction is attached to it.

| Feature | Value from the function | How this value was reached |
|---|---:|---|
| URL length | 32 | `len` of the normalized URL. The scheme was already present, so nothing was added. |
| Number of dots | 1 | Count of `.` in the full string. The host `example.com` contributes the dot. |
| Number of hyphens | 0 | Count of `-`. This address has none. |
| Number of digits | 1 | Count of numeric characters. The query value `1` is the digit. |
| Number of @ symbols | 0 | Count of `@`. There is none, so the host is not hidden behind a user name. |
| Number of subdomains | 0 | The host labels are `example` and `com`. Extra labels beyond the last two: 0. |
| HTTPS usage | 1 | The scheme is `https`, so the feature is 1. This does not prove the site is safe. |
| IP address usage | 0 | `ipaddress` does not accept the hostname `example.com`, so the feature is 0. |
| Suspicious keywords | 1 | Lowercase substring count. `login` matches once. `user` is not in the keyword list. |
| Number of question marks | 1 | One `?` starts the query string. |
| Number of equals signs | 1 | One `=` joins `user` to `1`. |
| URL path length | 6 | `urlparse` reads the path `/login`, and `len("/login")` is 6. |

The keyword list is: login, verify, update, secure, account, confirm, password.

## Code Explanation

### features.py

1. `normalize_url` trims the text, rejects spaces and unknown schemes, and adds `https://` when the scheme is missing.
2. `hostname_is_ip` uses `ipaddress` on the parsed hostname only, not on the whole string.
3. `count_subdomains` counts hostname labels beyond the last two. IP hosts count as 0.
4. `features_from_normalized` parses the URL and fills the 12 counters.
5. The returned dictionary is always built in the order of `FEATURE_NAMES`.
6. Nothing in this file opens a network connection.

### train.py

1. `clean_dataset` removes missing, invalid, conflicting, and duplicate rows, and prints the counts.
2. `convert_labels` applies `phishing_label = 1 - original_label` only if the source note and the published counts still match.
3. `stratified_sample` keeps 30,000 real rows with the cleaned class mix.
4. `make_feature_matrix` calls `feature_vector` and builds X.
5. `train_test_split` creates the 70/15/15 stratified split with seed 42.
6. The script fits Logistic Regression, KNN, and Random Forest on `X_train` and `y_train`.
7. `score_predictions` computes accuracy, precision, recall, and F1 with phishing as the positive class.
8. `select_best_model` applies the F1 rule, then the recall tie-break, then the simplicity tie-break.
9. Only the chosen model is scored on `X_test`.
10. The model is saved with Joblib and immediately loaded again to check that the predictions match.
11. Charts, `metrics.json`, and this document are written from those results.

### app.py

1. `load_artifacts` runs once at startup and stores the model in memory.
2. `GET /api/health` returns `{"status": "ok"}`.
3. `POST /api/predict` reads the JSON URL, validates it, and extracts features.
4. `class_probability` calls `predict` and `predict_proba`, then selects the class-1 column.
5. `build_reasons` and `build_details` describe the measured URL traits. They do not claim to be the cause of the decision.
6. `GET /api/model-info` returns the saved metrics, including the real test scores and the validation table.
7. Empty, invalid, unsupported, and oversized URLs return a JSON error. A missing model returns a clear message instead of a fake score.

## Full Project Workflow

1. Load the bundled URL dataset and its label information.
2. Cleaning in `train.py`.
3. Label conversion to 0 = SAFE and 1 = PHISHING.
4. Feature extraction with `features.py`.
5. Stratified split into training, validation, and test.
6. Model training for Logistic Regression, KNN, and Random Forest.
7. Model comparison on the validation split.
8. Best-model selection by phishing F1, then recall, then simplicity.
9. One test-set evaluation of the selected model.
10. Saving the model with Joblib and saving the metrics.
11. FastAPI loads the saved model and scores a new URL.
12. The Next.js page shows the prediction, the probability, and the measured URL traits.

Selected model for this run: **Random Forest**.

Test accuracy 0.9964 · precision 0.9984 · recall 0.9932 · F1 0.9958.
