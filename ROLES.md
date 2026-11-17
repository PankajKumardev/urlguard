# URLGuard — team roles

We downloaded the PhiUSIIL dataset from the UCI Machine Learning Repository. Each row gives a URL and a label indicating whether it is legitimate or phishing.

| Feature | Team member | Implementation |
| --- | --- | --- |
| Dataset preparation | Pankaj | Clean the saved URL records, remove invalid rows, duplicates and conflicting labels, and convert the labels so phishing is class 1. [train.py](backend/train.py) |
| Feature extraction and data splitting | Pankaj | Normalize each URL, convert its text into 12 ordered numeric features, and prepare stratified training, validation, and test sets. [features.py](backend/features.py) · [train.py](backend/train.py) |
| URL entry form | Pankaj | Accept a URL, submit it to the prediction API, and handle loading and input errors. [UrlScanner.tsx](frontend/components/UrlScanner.tsx) |
| Logistic Regression and K-Nearest Neighbors | Pankaj | Configure and train the two scaled pipelines. Logistic Regression learns feature weights. KNN classifies each URL using its five nearest training examples. [modeling.py](backend/modeling.py) |
| Random Forest | Shivam | Configure and train the forest of 100 decision trees with a maximum depth of 20. Evaluate its validation predictions. [modeling.py](backend/modeling.py) |
| Model comparison | Both | Use the same training and validation splits for all three candidates. Compare phishing F1 and apply the recall and simplicity rules for near ties. [modeling.py](backend/modeling.py) |
| Evaluation and prediction API | Shivam | Evaluate the selected classifier on the test set, save it with its feature order, and serve predictions using the fitted model. [train.py](backend/train.py) · [app.py](backend/app.py) |
| Prediction results and Model Insights | Shivam | Display the predicted class and phishing probability. Present validation scores and the held-out confusion matrix on the Model Insights page. [PredictionResult.tsx](frontend/components/PredictionResult.tsx) · [Model Insights page](frontend/app/model-insights/page.tsx) |

**Where the work meets.** In `candidate_models`, Pankaj owns the Logistic Regression and KNN pipelines, and Shivam owns the Random Forest configuration. All three use the same fit-and-predict loop in `train_candidates`. After selection, Shivam evaluates the winner on the held-out test set and connects it to the API.
