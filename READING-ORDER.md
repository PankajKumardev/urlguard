# URLGuard reading order

Read the project in two passes. First learn how a model is trained. Then follow one URL from the website into that saved model.

## First pass — training

1. **backend/data/urls.csv** — Open a few rows. Each row has a URL and its original label. Use DATASET.md to understand the source.
2. **backend/features.py** — Start with FEATURE_NAMES and SUSPICIOUS_KEYWORDS. These define the 12 measurements and the words we count.
3. **backend/features.py** — Read normalize_url. Follow how example.com becomes https://example.com and how the function removes a leading www.
4. **backend/features.py** — Read hostname_is_ip and count_subdomains. These help measure the hostname.
5. **backend/features.py** — Read extract_features, features_from_normalized and feature_vector. Write down the 12 numbers for one URL in their declared order.
6. **backend/train.py** — Read clean_dataset and convert_labels. Understand which rows are removed and why project label 1 means phishing.
7. **backend/train.py** — Read stratified_sample and make_feature_matrix. X contains feature rows and y contains the expected labels.
8. **backend/train.py** — Start main and follow the two train_test_split calls. The 30,000 sampled rows become 21,000 training, 4,500 validation and 4,500 test rows.
9. **backend/modeling.py** — Read candidate_models. Pankaj studies the scaled Logistic Regression and KNN pipelines. Shivam studies the Random Forest settings.
10. **backend/modeling.py** — Read train_candidates. Follow fit on training data, predict on validation data and score_predictions.
11. **backend/modeling.py** — Read select_best_model. Validation F1 chooses the winner, with recall and simplicity handling close scores.
12. **backend/train.py** — Return to main after train_candidates. Follow the chosen model’s test predictions, confusion matrix and metrics.
13. **backend/train.py** — Follow joblib.dump and joblib.load. The trained estimator is saved with its feature names and checked after reloading.
14. **backend/results/metrics.json** — Look at validation_comparison, test_metrics and counts. These are the results the website displays.

## Second pass — one website prediction

15. **frontend/app/page.tsx** — See how the home page includes the scanner through ScannerHero.
16. **frontend/components/ScannerHero.tsx** — Find where UrlScanner is displayed.
17. **frontend/components/UrlScanner.tsx** — Read onSubmit. Follow the entered URL into POST /api/predict and see how loading and errors are handled.
18. **backend/app.py** — Read load_artifacts. The saved model is loaded once and kept in memory.
19. **backend/app.py** — Read predict. Follow JSON input through URL validation and features_from_normalized.
20. **backend/app.py** — Read class_probability. It puts the feature vector into a one-row matrix and calls predict and predict_proba.
21. **backend/app.py** — Read build_reasons and build_details, then return to predict’s response. These describe measured URL traits for display.
22. **frontend/components/PredictionResult.tsx** — See how the returned class, probability and details become the result card.
23. **frontend/app/model-insights/page.tsx** — Follow the request to /api/model-info.
24. **backend/app.py** — Read model_info and the two image routes. They serve the saved metrics and charts.

## Supporting files to read last

25. **backend/tests/test_features.py** — Shows examples of expected URL measurements.
26. **backend/tests/test_prediction.py** — Checks the model selection rules, saved artifacts and prediction API.

Spend most of your ML preparation on steps 2–13 and 18–20. Use ROLES.md to see which model each member owns.
