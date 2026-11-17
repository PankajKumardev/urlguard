# URLGuard — team roles

We downloaded the PhiUSIIL dataset from the UCI Machine Learning Repository. Each row gives a URL and a label indicating whether it is legitimate or phishing.

| Feature | Team member | Implementation |
| --- | --- | --- |
| Dataset preparation | Pankaj | Clean the saved URL records, remove invalid rows, duplicates and conflicting labels, and convert the labels so phishing is class 1. [train.py](backend/train.py) |
| Feature extraction and data splitting | Pankaj | Normalize each URL, convert its text into 12 ordered numeric features, and prepare stratified training, validation, and test sets. [features.py](backend/features.py) · [train.py](backend/train.py) |
| URL entry form | Pankaj | Accept a URL, submit it to the prediction API, and handle loading and input errors. [UrlScanner.tsx](frontend/components/UrlScanner.tsx) |
| Model training and selection | Shivam | Train Logistic Regression, K-Nearest Neighbors, and Random Forest. Compare phishing F1 on validation data and apply the recall and simplicity rules for close scores. [modeling.py](backend/modeling.py) |
| Evaluation and prediction API | Shivam | Evaluate the selected classifier on the test set, save it with its feature order, and serve predictions using the fitted model. [train.py](backend/train.py) · [app.py](backend/app.py) |
| Prediction results and Model Insights | Shivam | Display the predicted class and phishing probability. Present validation scores and the held-out confusion matrix on the Model Insights page. [PredictionResult.tsx](frontend/components/PredictionResult.tsx) · [Model Insights page](frontend/app/model-insights/page.tsx) |

**Shared file boundary.** In `train.py`, Pankaj owns the steps that produce the three datasets. Shivam’s part begins at the call to `train_candidates` and covers model selection, test evaluation, and saving the model and reports.
