# URLGuard code walkthrough

[backend/features.py](#backendfeaturespy) · [backend/modeling.py](#backendmodelingpy) · [backend/train.py](#backendtrainpy) · [backend/app.py](#backendapppy) · [backend/download_data.py](#backenddownloaddatapy) · [backend/write_docs.py](#backendwritedocspy) · [Website and test files](#website-and-test-files)

**Start with features.py, then modeling.py.** Pankaj predicts the vector for example.com/login. Shivam explains how three fitted models are compared and how the winner scores that vector.

The numbers beside the source are the line numbers in the repository. Read each short passage and trace the values in the numbered code below it.

## [backend/features.py](backend/features.py)

Converts raw text into the same twelve numeric columns for training and live prediction.

The module contract defines which schemes and host variants receive the same treatment. The imports split URLs into components and recognize IPv4 and IPv6 addresses.

```text
   1  """Turn a URL string into a fixed list of simple numeric features.
   3  Training and live prediction both use features_from_normalized(). The function
   4  reads the URL text only. It does not open, visit, or download the address.
   6  Scheme rule:
   7  - A missing scheme is treated as https:// (example.com becomes https://example.com).
   8  - A leading // is treated as https:// (protocol-relative URL).
   9  - Only http and https are accepted.
  10  - A leading www. label is removed. Every legitimate URL in the PhiUSIIL file
  11    starts with www, so keeping it would let the model treat that prefix as
  12    proof of safety. www.x.com and x.com are scored as the same host.
  13  """
  15  from urllib.parse import urlparse, urlunparse
  16  import ipaddress
```

Read these names down the list. Their position is the column position the fitted model expects.

```text
  19  FEATURE_NAMES = [
  20      "url_length",
  21      "dot_count",
  22      "hyphen_count",
  23      "digit_count",
  24      "at_count",
  25      "subdomain_count",
  26      "uses_https",
  27      "is_ip_address",
  28      "suspicious_keyword_count",
  29      "question_mark_count",
  30      "equals_count",
  31      "path_length",
  32  ]
```

Each word contributes its number of appearances in the lowercase URL. The length limit applies to the API input.

```text
  34  SUSPICIOUS_KEYWORDS = [
  35      "login",
  36      "verify",
  37      "update",
  38      "secure",
  39      "account",
  40      "confirm",
  41      "password",
  42  ]
  44  # The website refuses very long pastes. Training does not use this limit.
  45  API_MAX_URL_LENGTH = 2048
```

Each metadata record gives a machine name, display label, measurement and interpretation. The reports use these to explain the twelve columns.

```text
  47  FEATURE_INFO = [
  48      {
  49          "name": "url_length",
  50          "label": "URL length",
  51          "description": "Total number of characters in the normalized URL.",
  52          "why": "Some phishing links are very long because they hide the real destination or add extra tracking text.",
  53      },
  54      {
  55          "name": "dot_count",
  56          "label": "Number of dots",
  57          "description": "How many '.' characters appear in the full URL string.",
  58          "why": "Extra dots can come from extra host labels or from a long, messy link.",
  59      },
  60      {
  61          "name": "hyphen_count",
  62          "label": "Number of hyphens",
  63          "description": "How many '-' characters appear in the full URL string.",
  64          "why": "Hyphens are sometimes used to imitate a real brand name inside a hostname or path.",
  65      },
  66      {
  67          "name": "digit_count",
  68          "label": "Number of digits",
  69          "description": "How many numeric characters appear in the URL.",
  70          "why": "Automatically generated phishing links often contain many digits.",
  71      },
  72      {
  73          "name": "at_count",
  74          "label": "Number of @ symbols",
  75          "description": "How many '@' characters appear in the URL.",
  76          "why": "An @ symbol can make the real hostname harder to notice, because the browser uses the host after the @.",
  77      },
  78      {
  79          "name": "subdomain_count",
  80          "label": "Number of subdomains",
  81          "description": (
  82              "Extra hostname labels beyond the last two labels, after a leading www is removed. "
  83              "example.com and www.example.com both count as 0. login.example.com counts as 1. "
  84              "This is a simple approximation and does not understand suffixes such as co.uk. "
  85              "IP addresses count as 0. "
  86              "www is removed because every legitimate row in this dataset starts with www."
  87          ),
  88          "why": "Extra subdomains are sometimes used to imitate a trusted brand.",
  89      },
  90      {
  91          "name": "uses_https",
  92          "label": "HTTPS usage",
  93          "description": "1 if the scheme is https, otherwise 0. A missing scheme is treated as https before this check.",
  94          "why": "The scheme is one signal. HTTPS encrypts traffic, but phishing websites can use HTTPS too, so it is not proof of safety.",
  95      },
  96      {
  97          "name": "is_ip_address",
  98          "label": "IP address usage",
  99          "description": "1 if the hostname is an IPv4 or IPv6 address according to Python's ipaddress module, otherwise 0.",
 100          "why": "Most public websites use a domain name. A raw IP address can be suspicious, though it is not always malicious.",
 101      },
 102      {
 103          "name": "suspicious_keyword_count",
 104          "label": "Suspicious keywords",
 105          "description": (
 106              "Total substring count of these words, matched without regard to case: "
 107              + ", ".join(SUSPICIOUS_KEYWORDS)
 108              + ". A word inside a longer word can also match. These are weak indicators, not proof."
 109          ),
 110          "why": "Words such as login, verify, and password appear in many phishing lures. They also appear on real login pages.",
 111      },
 112      {
 113          "name": "question_mark_count",
 114          "label": "Number of question marks",
 115          "description": "How many '?' characters appear in the URL.",
 116          "why": "A question mark usually starts a query string, which can carry extra parameters.",
 117      },
 118      {
 119          "name": "equals_count",
 120          "label": "Number of equals signs",
 121          "description": "How many '=' characters appear in the URL.",
 122          "why": "Equals signs usually join a query parameter to its value.",
 123      },
 124      {
 125          "name": "path_length",
 126          "label": "URL path length",
 127          "description": "Number of characters in the URL path returned by urllib.parse, including the leading slash when one is present.",
 128          "why": "A very long path can be used to bury the real destination of a link.",
 129      },
 130  ]
```

Validate a raw value. Missing values and non-string inputs raise errors before string operations can run.

```text
 133  def normalize_url(url, max_length=None):
 134      """Return a cleaned URL string, or raise ValueError if it cannot be used."""
 135      if url is None:
 136          raise ValueError("Please enter a URL.")
 137      if not isinstance(url, str):
 138          raise ValueError("The url field must be a string.")
```

Trim outside spaces. Reject empty or missing-value text, embedded whitespace and overlong API submissions.

```text
 140      text = url.strip()
 141      if text == "" or text.lower() == "nan":
 142          raise ValueError("Please enter a URL.")
 143      if any(character.isspace() for character in text):
 144          raise ValueError("The URL contains spaces. Remove the spaces and try again.")
 145      if max_length is not None and len(text) > max_length:
 146          raise ValueError(
 147              "This URL is too long to analyze. Please submit a URL under "
 148              + str(max_length)
 149              + " characters."
 150          )
```

Add https to a protocol-relative address. Parse the string. If no scheme was entered, prefix https:// and parse again.

```text
 152      if text.startswith("//"):
 153          text = "https:" + text
 155      parsed = urlparse(text)
 156      if parsed.scheme == "" and "://" not in text:
 157          text = "https://" + text
 158          parsed = urlparse(text)
```

Accept HTTP or HTTPS and require a hostname. Pass the checked string to the www normalizer.

```text
 160      scheme = parsed.scheme.lower()
 161      if scheme not in ("http", "https"):
 162          raise ValueError("Only HTTP and HTTPS URLs can be analyzed.")
 163      if not parsed.hostname:
 164          raise ValueError("This URL does not have a valid hostname.")
 165      return strip_leading_www(text)
```

Read the parsed hostname. Return the original text if there is no leading www label.

```text
 168  def strip_leading_www(url):
 169      """Drop a leading www. label. www.example.com and example.com become the same host."""
 170      parsed = urlparse(url)
 171      host = parsed.hostname or ""
 172      if not host.lower().startswith("www."):
 173          return url
```

Separate the optional user information at the last @. Keep it when rebuilding the network address. Remove exactly four characters from a host beginning with www. Preserve the remaining URL components.

```text
 174      netloc = parsed.netloc
 175      prefix, _, hostport = netloc.rpartition("@")
 176      if prefix:
 177          prefix = prefix + "@"
 178      if not hostport.lower().startswith("www."):
 179          return url
 180      return urlunparse(parsed._replace(netloc=prefix + hostport[4:]))
```

Return false for an absent hostname. Remove surrounding whitespace and brackets. A successful IP parse returns true. An invalid IP spelling raises ValueError and returns false.

```text
 183  def hostname_is_ip(hostname):
 184      """Return True when the hostname is an IP address, not merely text that contains digits."""
 185      if hostname is None:
 186          return False
 187      candidate = hostname.strip().strip("[]")
 188      try:
 189          ipaddress.ip_address(candidate)
 190          return True
 191      except ValueError:
 192          return False
```

IP addresses and empty hosts have zero subdomains. Split a domain at dots, drop empty labels and subtract the final two labels, with a minimum of zero. This approximation can overcount for co.uk.

```text
 195  def count_subdomains(hostname, is_ip):
 196      """Count extra hostname labels using a simple two-label approximation.
 198      The last two dot-separated labels are treated as the domain and its suffix.
 199      Labels before those two are counted as subdomains.
 200      """
 201      if is_ip or not hostname:
 202          return 0
 203      labels = [part for part in hostname.split(".") if part != ""]
 204      return max(0, len(labels) - 2)
```

Normalize once, then give the resulting text to the shared measuring function.

```text
 207  def extract_features(url, max_length=None):
 208      """Return one feature dictionary. Keys always follow FEATURE_NAMES."""
 209      normalized = normalize_url(url, max_length=max_length)
 210      return features_from_normalized(normalized)
```

Split the normalized string, read the hostname, make a lowercase copy for matching and check whether the host is an IP.

```text
 213  def features_from_normalized(normalized):
 214      """Measure one URL after normalize_url has validated it."""
 215      parsed = urlparse(normalized)
 216      hostname = parsed.hostname or ""
 217      lowered = normalized.lower()
 218      is_ip = hostname_is_ip(hostname)
```

Start the keyword counter at zero. Add the count of each listed word, including repetitions within a URL.

```text
 220      keyword_total = 0
 221      for keyword in SUSPICIOUS_KEYWORDS:
 222          keyword_total += lowered.count(keyword)
```

Start a dictionary. Measure normalized text length, then count dots and hyphens over the full text.

```text
 224      values = {
 225          "url_length": len(normalized),
 226          "dot_count": normalized.count("."),
 227          "hyphen_count": normalized.count("-"),
```

Count digit characters and @ symbols. Use the hostname helper for the subdomain count.

```text
 228          "digit_count": sum(1 for character in normalized if character.isdigit()),
 229          "at_count": normalized.count("@"),
 230          "subdomain_count": count_subdomains(hostname, is_ip),
```

Encode HTTPS and IP status as zero or one. Store the keyword total.

```text
 231          "uses_https": 1 if parsed.scheme.lower() == "https" else 0,
 232          "is_ip_address": 1 if is_ip else 0,
 233          "suspicious_keyword_count": keyword_total,
```

Count question marks and equals signs. Measure only the path for the final column. Emit integers in the declared feature order.

```text
 234          "question_mark_count": normalized.count("?"),
 235          "equals_count": normalized.count("="),
 236          "path_length": len(parsed.path or ""),
 237      }
 238      return {name: int(values[name]) for name in FEATURE_NAMES}
```

For a training URL, call the shared extractor and read the fields in model order to produce one list of numbers.

```text
 241  def feature_vector(url, max_length=None):
 242      """Return the feature values in the same order used during training."""
 243      features = extract_features(url, max_length=max_length)
 244      return [features[name] for name in FEATURE_NAMES]
```

## [backend/modeling.py](backend/modeling.py)

Build three candidates, train on known labels and select one using held-out validation predictions.

Import each classifier separately, metric functions for measuring errors, Pipeline to chain steps, and StandardScaler to learn training-set means and scales.

```text
   1  """Fit three classifiers on training rows and choose one using validation rows."""
   3  from sklearn.ensemble import RandomForestClassifier
   4  from sklearn.linear_model import LogisticRegression
   5  from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
   6  from sklearn.neighbors import KNeighborsClassifier
   7  from sklearn.pipeline import Pipeline
   8  from sklearn.preprocessing import StandardScaler
```

Fix the random seed. Define the allowed shortfall from the leading F1 and recall.

```text
  11  RANDOM_STATE = 42
  12  NEAR_F1 = 0.005
  13  NEAR_RECALL = 0.01
```

Record the project’s preference order for practically tied models. A lower number wins the final tie.

```text
  15  # Lower number means a simpler model. Used only when scores are nearly the same.
  16  SIMPLICITY = {
  17      "Logistic Regression": 0,
  18      "K-Nearest Neighbors": 1,
  19      "Random Forest": 2,
  20  }
```

Compare true labels with predictions. Accuracy is the fraction correct. Precision counts correct phishing flags among all flags. Recall counts detected phishing cases among actual cases. F1 combines precision and recall. Zero division produces zero.

```text
  23  def score_predictions(y_true, y_pred):
  24      """Score phishing as the positive class (label 1)."""
  25      return {
  26          "accuracy": float(accuracy_score(y_true, y_pred)),
  27          "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
  28          "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
  29          "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
  30      }
```

The selection function takes validation rows containing model names and scores. Its docstring gives the order of decisions.

```text
  33  def select_best_model(rows):
  34      """Choose a model from validation scores only.
  36      1. The main score is the phishing-class F1.
  37      2. If F1 scores are within 0.005, use phishing recall as the tie-breaker.
  38      3. If recall is also within 0.01, choose the simpler model.
  39      """
```

Find the highest phishing F1. Keep rows within the F1 margin. A single surviving row wins and receives a printable explanation.

```text
  40      best_f1 = max(row["f1"] for row in rows)
  41      contenders = [row for row in rows if best_f1 - row["f1"] <= NEAR_F1]
  42      if len(contenders) == 1:
  43          winner = contenders[0]
  44          reason = f"{winner['model']} had the highest phishing validation F1 ({best_f1:.4f})."
```

With multiple contenders find their highest recall. Keep those within the recall margin. Choose the lowest simplicity rank among the survivors.

```text
  45      else:
  46          best_recall = max(row["recall"] for row in contenders)
  47          finalists = [row for row in contenders if best_recall - row["recall"] <= NEAR_RECALL]
  48          winner = min(finalists, key=lambda row: SIMPLICITY[row["model"]])
```

Record which tie rule settled the result. Return the name, explanation and winning row.

```text
  49          if len(finalists) == 1:
  50              reason = f"{winner['model']} won on phishing recall after a near tie on F1."
  51          else:
  52              reason = f"{winner['model']} is the simpler model among those close on F1 and recall."
  53      return winner["model"], reason, winner
```

For Logistic Regression, scale the features then fit a linear classifier. Allow up to 1,000 solver iterations.

```text
  56  def candidate_models():
  57      logistic = Pipeline([
  58          ("scaler", StandardScaler()),
  59          ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
  60      ])
```

For KNN, scale the features then use the five nearest training rows for each prediction.

```text
  61      neighbors = Pipeline([
  62          ("scaler", StandardScaler()),
  63          ("classifier", KNeighborsClassifier(n_neighbors=5)),
  64      ])
```

Build 100 trees with maximum depth 20. The seed makes the experiment repeatable and n_jobs=-1 allows all available processors.

```text
  65      forest = RandomForestClassifier(
  66          n_estimators=100, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1
  67      )
```

Pair each display name with its estimator so one loop can train all three.

```text
  68      return [
  69          ("Logistic Regression", logistic),
  70          ("K-Nearest Neighbors", neighbors),
  71          ("Random Forest", forest),
  72      ]
```

Receive the training matrix and labels and the separate validation pair. Prepare a dictionary for trained estimators and lists for score rows and reports.

```text
  75  def train_candidates(X_train, y_train, X_val, y_val):
  76      fitted = {}
  77      validation_rows = []
  78      validation_reports = []
```

Take the next name and model. Print progress. Fit only on the training pair. Predict classes for the validation matrix.

```text
  79      for name, model in candidate_models():
  80          print("Training", name + "...")
  81          model.fit(X_train, y_train)
  82          predictions = model.predict(X_val)
```

Compare with the validation labels. Attach the name, append the scores and keep the fitted instance under its name.

```text
  83          scores = score_predictions(y_val, predictions)
  84          scores["model"] = name
  85          validation_rows.append(scores)
  86          fitted[name] = model
```

Build a per-class report from the true and predicted validation labels in SAFE, PHISHING order. Format to four decimals and handle empty denominators.

```text
  87          report = classification_report(
  88              y_val,
  89              predictions,
  90              labels=[0, 1],
  91              target_names=["SAFE", "PHISHING"],
  92              digits=4,
  93              zero_division=0,
  94          )
```

Store the named report and print the F1. Once all candidates finish, the complete score table is available.

```text
  95          validation_reports.append("VALIDATION — " + name + "\n" + report)
  96          print(name, "validation F1:", format(scores["f1"], ".4f"))
  98      print("Validation comparison (these numbers choose the model):")
```

Select from the validation rows, retrieve the already fitted winner, print the reason and return the estimator and report data to train.py.

```text
 100      chosen_name, reason, chosen_scores = select_best_model(validation_rows)
 101      chosen_model = fitted[chosen_name]
 102      print("Selected model:", chosen_name)
 103      print("Reason:", reason)
 104      return chosen_model, chosen_name, reason, chosen_scores, validation_rows, validation_reports
```

## [backend/train.py](backend/train.py)

Runs the experiment from CSV to saved estimator and evidence files. For the viva, follow clean_dataset, make_feature_matrix and main first.

Import numeric, file, plotting and ML tools. Paths are relative to this module so the same training command can run from another working directory. The seed and selection functions come from modeling.py.

```text
   1  """Train three classic models and keep the one with the best validation F1.
   3  Run this after download_data.py. It does not start the website.
   4  The test set is scored once, after the model has already been chosen.
   5  """
   7  import hashlib
   8  import json
   9  import os
  10  import platform
  11  from datetime import datetime, timezone
  13  import joblib
  14  import matplotlib
  16  matplotlib.use("Agg")
  17  import matplotlib.pyplot as plt
  18  import numpy as np
  19  import pandas as pd
  20  import sklearn
  21  from sklearn.metrics import classification_report, confusion_matrix
  22  from sklearn.model_selection import train_test_split
  24  from features import FEATURE_INFO, FEATURE_NAMES, feature_vector, normalize_url
  25  from modeling import NEAR_F1, NEAR_RECALL, RANDOM_STATE, score_predictions, select_best_model, train_candidates
  26  from write_docs import write_all_docs
  29  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  30  DATA_PATH = os.path.join(BASE_DIR, "data", "urls.csv")
  31  SOURCE_INFO_PATH = os.path.join(BASE_DIR, "data", "source_info.json")
  32  MODEL_DIR = os.path.join(BASE_DIR, "models")
  33  RESULTS_DIR = os.path.join(BASE_DIR, "results")
  34  MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
  36  SAMPLE_SIZE = 30000
  37  OFFICIAL_LABEL_SENTENCE = "Label 1 corresponds to a legitimate URL, label 0 to a phishing URL"
```

Work on a copy. Convert labels to numeric values, count the original classes, trim URL strings and identify rows missing either required value.

```text
  40  def clean_dataset(df):
  41      """Drop missing, invalid, conflicting, and duplicate URL rows.
  43      Conflicting labels are removed before duplicate rows. If duplicates were
  44      deleted first, one of the disagreeing labels could be kept by accident.
  45      """
  46      summary = {"original_rows": int(len(df))}
  47      df = df.copy()
  48      numeric_labels = pd.to_numeric(df["label"], errors="coerce")
  49      summary["file_label_1"] = int((numeric_labels == 1).sum())
  50      summary["file_label_0"] = int((numeric_labels == 0).sum())
  52      df["label"] = numeric_labels
  53      df["url"] = df["url"].astype("string").str.strip()
  54      missing = df["url"].isna() | (df["url"] == "") | df["label"].isna()
  55      summary["removed_missing"] = int(missing.sum())
```

Filter unusable labels and convert the remaining labels to integers. Normalize each URL, using a missing value for a failed parse. Count and drop those failures.

```text
  56      df = df.loc[~missing].copy()
  58      bad_label = ~df["label"].isin([0, 1])
  59      summary["removed_bad_label"] = int(bad_label.sum())
  60      df = df.loc[~bad_label].copy()
  61      df["label"] = df["label"].astype(int)
  62      df["url"] = df["url"].astype(str)
  64      normalized_urls = []
  65      for url in df["url"]:
  66          try:
  67              normalized_urls.append(normalize_url(url))
  68          except ValueError:
  69              normalized_urls.append(None)
  70      df["url"] = normalized_urls
  71      summary["removed_invalid"] = int(df["url"].isna().sum())
```

Group rows by normalized URL. Multiple distinct labels for one URL make its target ambiguous, so discard all rows in such groups. Keep one row per remaining URL.

```text
  72      df = df.dropna(subset=["url"])
  74      label_variety = df.groupby("url")["label"].nunique()
  75      conflict_urls = label_variety[label_variety > 1].index
  76      conflict_rows = df["url"].isin(conflict_urls)
  77      summary["conflict_url_count"] = int(len(conflict_urls))
  78      summary["removed_conflicts"] = int(conflict_rows.sum())
  79      df = df.loc[~conflict_rows].copy()
  81      summary["duplicate_count"] = int(df.duplicated(subset=["url"]).sum())
  82      df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
```

Add removal totals and URL length statistics to the summary and return it with the frame.

```text
  84      summary["removed_invalid_rows"] = (
  85          summary["removed_missing"] + summary["removed_bad_label"] + summary["removed_invalid"]
  86      )
  87      summary["rows_before_label_conversion"] = int(len(df))
  88      summary["min_url_length"] = int(df["url"].str.len().min())
  89      summary["max_url_length"] = int(df["url"].str.len().max())
  90      summary["urls_longer_than_2048"] = int((df["url"].str.len() > 2048).sum())
  91      return df, summary
```

Confirm the source metadata gives the expected class meaning and counts. Copy the frame, subtract each original label from one, and return project labels.

```text
  94  def convert_labels(df, source_info):
  95      """Flip the UCI codes into 0 = SAFE and 1 = PHISHING."""
  96      sentence = str(source_info.get("official_label_sentence", ""))
  97      if sentence != OFFICIAL_LABEL_SENTENCE or not source_info.get("label_mapping_confirmed"):
  98          raise SystemExit(
  99              "Refusing to convert labels. The source file does not confirm that "
 100              "1 = legitimate and 0 = phishing."
 101          )
 102      if int(source_info.get("downloaded_label_1", -1)) != 134850:
 103          raise SystemExit("Label 1 count does not match the published legitimate count (134,850).")
 104      if int(source_info.get("downloaded_label_0", -1)) != 100945:
 105          raise SystemExit("Label 0 count does not match the published phishing count (100,945).")
 107      converted = df.copy()
 108      converted["label"] = 1 - converted["label"].astype(int)
 109      return converted
```

Use all rows when the data already fits. Otherwise compute each class share times the sample target. Round down, then give leftover places to the largest fractional remainders.

```text
 112  def stratified_sample(df, sample_size, random_state):
 113      """Keep about sample_size real rows, with the same class mix as the cleaned data."""
 114      if len(df) <= sample_size:
 115          shuffled = df.sample(frac=1.0, random_state=random_state)
 116          return shuffled.reset_index(drop=True)
 118      labels = sorted(int(label) for label in df["label"].unique())
 119      proportions = df["label"].value_counts(normalize=True)
 120      raw = {label: float(proportions[label]) * sample_size for label in labels}
 121      allocation = {label: int(np.floor(raw[label])) for label in labels}
 122      leftover = sample_size - sum(allocation.values())
 123      order = sorted(labels, key=lambda label: (raw[label] - allocation[label], label), reverse=True)
 124      for label in order:
 125          if leftover == 0:
 126              break
 127          allocation[label] += 1
 128          leftover -= 1
```

Sample within each class using the fixed seed. Join the groups and shuffle them so row order is mixed.

```text
 130      parts = []
 131      for label in labels:
 132          group = df[df["label"] == label]
 133          take = min(allocation[label], len(group))
 134          parts.append(group.sample(n=take, random_state=random_state))
 135      sampled = pd.concat(parts, axis=0)
 136      sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
 137      return sampled
```

Convert each URL to one ordered feature vector. Stack the vectors into a numeric matrix, check for missing numbers and return it.

```text
 140  def make_feature_matrix(urls):
 141      rows = [feature_vector(url) for url in urls]
 142      matrix = np.asarray(rows, dtype=float)
 143      if np.isnan(matrix).any():
 144          raise SystemExit("Feature matrix contains missing values.")
 145      return matrix
```

Get probability columns from the estimator and read the column whose class label is one.

```text
 148  def phishing_probability_from_model(model, matrix):
 149      """Read the probability column that belongs to class 1."""
 150      probabilities = model.predict_proba(matrix)
 151      classes = [int(label) for label in model.classes_]
 152      if 1 not in classes:
 153          raise SystemExit("The trained model does not contain the phishing class.")
 154      return probabilities[:, classes.index(1)]
```

Hide most of a phishing hostname and its path for display in the report.

```text
 157  def mask_phishing_url(url):
 158      """Hide a phishing URL before it is written into the report."""
 159      from urllib.parse import urlparse
 161      parsed = urlparse(url)
 162      host = parsed.hostname or "redacted"
 163      if len(host) <= 4:
 164          masked_host = "xxxx"
 165      else:
 166          masked_host = host[:2] + "****" + host[-2:]
 167      return parsed.scheme + "://" + masked_host + "/[redacted-path]"
```

Prefer a correctly detected phishing example from the held-out predictions. Keep the first actual phishing row as a fallback.

```text
 170  def choose_example(urls, y_true, y_pred, model, matrix):
 171      """Pick one real held-out row and hide the URL when it is phishing."""
 172      chosen = None
 173      fallback = None
 174      for index in range(len(y_true)):
 175          if int(y_true[index]) == 1 and fallback is None:
 176              fallback = index
 177          if int(y_true[index]) == 1 and int(y_pred[index]) == 1:
 178              chosen = index
 179              break
 180      if chosen is None:
 181          chosen = 0 if fallback is None else fallback
```

Read its true label, prediction and probability. Verify the reported class agrees with the ordering of the two probabilities.

```text
 183      true_label = int(y_true[chosen])
 184      predicted_label = int(y_pred[chosen])
 185      probability = float(phishing_probability_from_model(model, matrix[chosen : chosen + 1])[0])
 186      full_probabilities = model.predict_proba(matrix[chosen : chosen + 1])[0]
 187      classes = [int(label) for label in model.classes_]
 188      safe_probability = float(full_probabilities[classes.index(0)])
 189      if predicted_label == 1 and probability < safe_probability:
 190          raise SystemExit("Phishing probability does not match the predicted class.")
 191      if predicted_label == 0 and probability > safe_probability:
 192          raise SystemExit("Safe prediction does not match the probability columns.")
```

Mask an actual phishing example or shorten a long legitimate display string. Package the original numeric features, labels and displayed text for the report.

```text
 194      raw_url = urls[chosen]
 195      if true_label == 1:
 196          shown = mask_phishing_url(raw_url)
 197          redacted = True
 198      elif len(raw_url) > 100:
 199          shown = raw_url[:80] + "..."
 200          redacted = False
 201      else:
 202          shown = raw_url
 203          redacted = False
 205      names = {0: "SAFE", 1: "PHISHING"}
 206      return {
 207          "source": "held-out test set",
 208          "url_display": shown,
 209          "url_was_redacted": redacted,
 210          "display_was_shortened": (not redacted) and shown != raw_url,
 211          "features": {
 212              name: int(matrix[chosen, feature_index])
 213              for feature_index, name in enumerate(FEATURE_NAMES)
 214          },
 215          "feature_vector": [int(value) for value in matrix[chosen].tolist()],
 216          "true_label": names[true_label],
 217          "predicted_label": names[predicted_label],
 218          "phishing_probability": round(probability, 6),
 219          "correct": true_label == predicted_label,
 220          "note": (
 221              "The original phishing URL is not printed. Features were calculated on the full URL before redaction."
 222              if redacted
 223              else "This safe URL comes from the held-out test split of the real dataset."
 224          ),
 225      }
```

Draw the matrix as a two-by-two image. Label actual and predicted axes, annotate each cell with its count and error type, save and close the figure.

```text
 228  def save_confusion_matrix(cm, model_name, path):
 229      fig, ax = plt.subplots(figsize=(6.4, 5.2))
 230      image = ax.imshow(cm, cmap="Blues")
 231      fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
 232      ax.set_xticks([0, 1])
 233      ax.set_yticks([0, 1])
 234      ax.set_xticklabels(["Predicted SAFE", "Predicted PHISHING"])
 235      ax.set_yticklabels(["Actual SAFE", "Actual PHISHING"])
 236      cell_names = [["True negative", "False positive"], ["False negative", "True positive"]]
 237      darkest = cm.max() if cm.max() > 0 else 1
 238      for row in range(2):
 239          for column in range(2):
 240              color = "white" if cm[row, column] > darkest / 2 else "#1A1916"
 241              ax.text(
 242                  column,
 243                  row,
 244                  str(int(cm[row, column])) + "\n" + cell_names[row][column],
 245                  ha="center",
 246                  va="center",
 247                  color=color,
 248                  fontsize=11,
 249              )
 250      ax.set_title("Test confusion matrix — " + model_name)
 251      fig.tight_layout()
 252      fig.savefig(path, dpi=140, bbox_inches="tight")
 253      plt.close(fig)
```

Draw four adjacent validation-score bars per model. Set the score axis, labels and legend, then save the comparison image.

```text
 256  def save_comparison_chart(rows, path):
 257      metric_keys = [
 258          ("accuracy", "Accuracy"),
 259          ("precision", "Precision"),
 260          ("recall", "Recall"),
 261          ("f1", "F1"),
 262      ]
 263      colors = ["#1E4E8C", "#4F7CAC", "#C47B4A", "#2F6F4E"]
 264      x = np.arange(len(rows))
 265      width = 0.18
 266      fig, ax = plt.subplots(figsize=(8.4, 4.8))
 267      for index, (key, label) in enumerate(metric_keys):
 268          offset = (index - 1.5) * width
 269          values = [row[key] for row in rows]
 270          ax.bar(x + offset, values, width, label=label, color=colors[index])
 271      ax.set_ylim(0, 1)
 272      ax.set_ylabel("Score")
 273      ax.set_xticks(x)
 274      ax.set_xticklabels([row["model"] for row in rows])
 275      ax.set_title("Validation scores used for model selection")
 276      ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
 277      ax.spines["top"].set_visible(False)
 278      ax.spines["right"].set_visible(False)
 279      fig.tight_layout()
 280      fig.savefig(path, dpi=140, bbox_inches="tight")
 281      plt.close(fig)
```

Count each numeric class. The next helper reads the dataset in chunks and hashes its bytes so a later reader can identify the data file.

```text
 284  def class_count(labels):
 285      values = np.asarray(labels)
 286      return int((values == 0).sum()), int((values == 1).sum())
 289  def sha256_file(path):
 290      digest = hashlib.sha256()
 291      with open(path, "rb") as handle:
 292          for chunk in iter(lambda: handle.read(1024 * 1024), b""):
 293              digest.update(chunk)
 294      return digest.hexdigest()
```

Print each model’s four validation metrics in a table. The next function prints the removal and retained-row counts.

```text
 297  def print_table(rows):
 298      print("")
 299      print(f"{'Model':<24} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
 300      for row in rows:
 301          print(
 302              f"{row['model']:<24} "
 303              f"{row['accuracy']:>10.4f} "
 304              f"{row['precision']:>10.4f} "
 305              f"{row['recall']:>10.4f} "
 306              f"{row['f1']:>10.4f}"
 307          )
 308      print("")
 311  def print_cleaning(summary):
 312      print("Original rows:", summary["original_rows"])
 313      print("Removed invalid rows:", summary["removed_invalid_rows"])
 314      print("  missing URL or label:", summary["removed_missing"])
 315      print("  bad label values:", summary["removed_bad_label"])
 316      print("  malformed URLs:", summary["removed_invalid"])
 317      print("Duplicate count:", summary["duplicate_count"])
 318      print("Removed conflicting-label rows:", summary["removed_conflicts"])
 319      print("Final rows:", summary["final_rows"])
 320      print("Legitimate URL count:", summary["safe_count"])
 321      print("Phishing URL count:", summary["phishing_count"])
```

Check the dataset and its metadata exist. Load both, clean rows, convert labels, update class counts and print the cleaning summary.

```text
 324  def main():
 325      if not os.path.exists(DATA_PATH) or not os.path.exists(SOURCE_INFO_PATH):
 326          raise SystemExit("Dataset file not found. Run python download_data.py first.")
 328      with open(SOURCE_INFO_PATH, encoding="utf-8") as handle:
 329          source_info = json.load(handle)
 331      print("Loading", DATA_PATH)
 332      raw = pd.read_csv(DATA_PATH)
 333      cleaned, summary = clean_dataset(raw)
 334      cleaned = convert_labels(cleaned, source_info)
 335      summary["final_rows"] = int(len(cleaned))
 336      summary["safe_count"] = int((cleaned["label"] == 0).sum())
 337      summary["phishing_count"] = int((cleaned["label"] == 1).sum())
 338      print_cleaning(summary)
 339      print("Label conversion: phishing_label = 1 - original_label")
 340      print("Project labels: 0 = SAFE, 1 = PHISHING")
```

Select the reproducible stratified subset. Extract URL strings, target vector y and feature matrix X in the same row order.

```text
 342      sampled = stratified_sample(cleaned, SAMPLE_SIZE, RANDOM_STATE)
 343      print("Records selected for modeling:", len(sampled))
 344      print("This is a real stratified subset, not the full dataset, and not synthetic data.")
 346      urls = sampled["url"].tolist()
 347      # X holds the feature values. y holds the project labels: 0 = SAFE, 1 = PHISHING.
 348      y = sampled["label"].to_numpy(dtype=int)
 349      X = make_feature_matrix(urls)
```

The first split holds out 30 percent, keeping URLs aligned with their features and labels. The seed fixes the split and stratify preserves the label proportions.

```text
 351      (
 352          X_train,
 353          X_temp,
 354          y_train,
 355          y_temp,
 356          url_train,
 357          url_temp,
 358      ) = train_test_split(
 359          X,
 360          y,
 361          urls,
 362          test_size=0.30,
 363          random_state=RANDOM_STATE,
 364          stratify=y,
 365      )
```

Divide that temporary portion equally into validation and test sets. Each is 15 percent of the original sample.

```text
 366      (
 367          X_val,
 368          X_test,
 369          y_val,
 370          y_test,
 371          url_val,
 372          url_test,
 373      ) = train_test_split(
 374          X_temp,
 375          y_temp,
 376          url_temp,
 377          test_size=0.50,
 378          random_state=RANDOM_STATE,
 379          stratify=y_temp,
 380      )
```

Intersect URL sets pairwise to check exact leakage. Confirm the sizes sum correctly, count classes and print the resulting split sizes.

```text
 382      overlap = len(set(url_train) & set(url_val))
 383      overlap += len(set(url_train) & set(url_test))
 384      overlap += len(set(url_val) & set(url_test))
 385      if overlap != 0:
 386          raise SystemExit("The same URL appeared in more than one split. Stopping.")
 387      if len(url_train) + len(url_val) + len(url_test) != len(sampled):
 388          raise SystemExit("Split sizes do not add up to the modeling sample.")
 390      train_safe, train_phishing = class_count(y_train)
 391      val_safe, val_phishing = class_count(y_val)
 392      test_safe, test_phishing = class_count(y_test)
 393      sample_safe, sample_phishing = class_count(y)
 394      print("Training rows:", len(y_train), "| SAFE", train_safe, "| PHISHING", train_phishing)
 395      print("Validation rows:", len(y_val), "| SAFE", val_safe, "| PHISHING", val_phishing)
 396      print("Test rows:", len(y_test), "| SAFE", test_safe, "| PHISHING", test_phishing)
```

Train and compare candidates. The returned model has already been fitted. Print the validation table.

```text
 398      chosen_model, chosen_name, reason, chosen_scores, validation_rows, validation_reports = train_candidates(
 399          X_train, y_train, X_val, y_val
 400      )
 401      print_table(validation_rows)
```

Predict once on the test matrix with the chosen estimator. Compute scores and unpack the confusion matrix with actual classes on rows and predictions on columns.

```text
 403      # The test set is used once, after selection. It is not used to choose again.
 404      print("Evaluating the selected model once on the held-out test set...")
 405      test_predictions = chosen_model.predict(X_test)
 406      test_scores = score_predictions(y_test, test_predictions)
 407      cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])
 408      true_negative = int(cm[0, 0])
 409      false_positive = int(cm[0, 1])
 410      false_negative = int(cm[1, 0])
 411      true_positive = int(cm[1, 1])
```

Build and print the test report. Each format operation controls the printed precision.

```text
 412      test_report = classification_report(
 413          y_test,
 414          test_predictions,
 415          labels=[0, 1],
 416          target_names=["SAFE", "PHISHING"],
 417          digits=4,
 418          zero_division=0,
 419      )
 420      print("Test accuracy:", format(test_scores["accuracy"], ".4f"))
 421      print("Test precision:", format(test_scores["precision"], ".4f"))
 422      print("Test recall:", format(test_scores["recall"], ".4f"))
 423      print("Test F1:", format(test_scores["f1"], ".4f"))
 424      print("Confusion matrix [[TN, FP], [FN, TP]]:")
 425      print(cm)
```

Select and display one held-out example using the test predictions.

```text
 427      example = choose_example(url_test, y_test, test_predictions, chosen_model, X_test)
 428      print("Held-out example URL:", example["url_display"])
 429      print(
 430          "True:",
 431          example["true_label"],
 432          "| Predicted:",
 433          example["predicted_label"],
 434          "| Phishing probability:",
 435          example["phishing_probability"],
 436      )
```

Create output directories and bundle the fitted model with feature order, positive class, seed and software version.

```text
 438      os.makedirs(MODEL_DIR, exist_ok=True)
 439      os.makedirs(RESULTS_DIR, exist_ok=True)
 440      uses_scaler = hasattr(chosen_model, "named_steps") and "scaler" in getattr(chosen_model, "named_steps", {})
 441      bundle = {
 442          "model": chosen_model,
 443          "model_name": chosen_name,
 444          "feature_names": list(FEATURE_NAMES),
 445          "positive_class": 1,
 446          "sklearn_version": sklearn.__version__,
 447          "random_state": RANDOM_STATE,
 448          "uses_internal_scaler": bool(uses_scaler),
 449      }
```

Save and reload the bundle. Check its first five test predictions against the in-memory ones and verify the feature order.

```text
 450      joblib.dump(bundle, MODEL_PATH)
 451      loaded = joblib.load(MODEL_PATH)
 452      reloaded = loaded["model"].predict(X_test[:5])
 453      if not np.array_equal(reloaded, test_predictions[:5]):
 454          raise SystemExit("The saved model did not reload with the same predictions.")
 455      if loaded["feature_names"] != list(FEATURE_NAMES):
 456          raise SystemExit("Saved feature order does not match features.py.")
 457      print("Saved and reloaded model:", MODEL_PATH)
```

Arrange and round the validation comparison. Save the CSV and produce both charts.

```text
 459      metric_names = ["accuracy", "precision", "recall", "f1"]
 460      comparison_frame = pd.DataFrame(validation_rows)[["model", *metric_names]].round(6)
 461      rounded_validation = comparison_frame.to_dict("records")
 462      comparison_frame.columns = ["Model", "Accuracy", "Precision", "Recall", "F1"]
 463      comparison_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
 464      comparison_frame.to_csv(comparison_path, index=False)
 466      cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
 467      chart_path = os.path.join(RESULTS_DIR, "model_comparison.png")
 468      save_confusion_matrix(cm, chosen_name, cm_path)
 469      save_comparison_chart(rounded_validation, chart_path)
```

For every split, hash each normalized URL and keep its split name. Reject duplicate hashes and write the split audit CSV.

```text
 471      hash_rows = []
 472      for split_name, split_urls in (
 473          ("train", url_train),
 474          ("validation", url_val),
 475          ("test", url_test),
 476      ):
 477          for url in split_urls:
 478              digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
 479              hash_rows.append({"split": split_name, "url_sha256": digest})
 480      hash_frame = pd.DataFrame(hash_rows)
 481      if hash_frame["url_sha256"].duplicated().any():
 482          raise SystemExit("URL hash overlap found while saving the split check.")
 483      hash_frame.to_csv(os.path.join(RESULTS_DIR, "split_assignments.csv"), index=False)
```

Format the cleaning counts and split sizes as text. Write the cleaning and per-class evaluation reports.

```text
 485      report_text = []
 486      report_text.append("URLGuard cleaning summary")
 487      report_text.append("Original rows: " + str(summary["original_rows"]))
 488      report_text.append("Removed invalid rows: " + str(summary["removed_invalid_rows"]))
 489      report_text.append("Duplicate count: " + str(summary["duplicate_count"]))
 490      report_text.append("Removed conflicting-label rows: " + str(summary["removed_conflicts"]))
 491      report_text.append("Final rows: " + str(summary["final_rows"]))
 492      report_text.append("Legitimate URL count: " + str(summary["safe_count"]))
 493      report_text.append("Phishing URL count: " + str(summary["phishing_count"]))
 494      report_text.append("Modeling rows: " + str(len(sampled)))
 495      report_text.append("Training rows: " + str(len(y_train)))
 496      report_text.append("Validation rows: " + str(len(y_val)))
 497      report_text.append("Test rows: " + str(len(y_test)))
 498      with open(os.path.join(RESULTS_DIR, "cleaning_summary.txt"), "w", encoding="utf-8") as handle:
 499          handle.write("\n".join(report_text) + "\n")
 501      with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as handle:
 502          handle.write("\n\n".join(validation_reports))
 503          handle.write("\n\nTEST — " + chosen_name + "\n")
 504          handle.write(test_report)
```

Record the actual candidate configurations so readers can reproduce them.

```text
 506      hyperparameters = {
 507          "Logistic Regression": {
 508              "scaler": "StandardScaler inside a Pipeline, fit on training data only",
 509              "solver": "lbfgs",
 510              "max_iter": 1000,
 511              "random_state": RANDOM_STATE,
 512          },
 513          "K-Nearest Neighbors": {
 514              "scaler": "StandardScaler inside a Pipeline, fit on training data only",
 515              "n_neighbors": 5,
 516          },
 517          "Random Forest": {
 518              "n_estimators": 100,
 519              "max_depth": 20,
 520              "random_state": RANDOM_STATE,
 521              "n_jobs": -1,
 522              "scaler": "None. Trees do not need scaled features.",
 523          },
 524      }
```

Assemble the report metadata. Project information, versions, label conventions and model settings explain how to interpret the scores.

```text
 526      metrics = {
 527          "project": "URLGuard — Machine Learning Based Phishing URL Detection System",
 528          "college": "Jaypee Institute of Information Technology, Noida",
 529          "department": "Computer Applications",
 530          "course": "Machine Learning Lab",
 531          "program": "MCA, 1st Year, 1st Semester",
 532          "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
 533          "sklearn_version": sklearn.__version__,
 534          "pandas_version": pd.__version__,
 535          "numpy_version": np.__version__,
 536          "python_version": platform.python_version(),
 537          "random_state": RANDOM_STATE,
 538          "sample_size_target": SAMPLE_SIZE,
 539          "near_tie_f1": NEAR_F1,
 540          "near_tie_recall": NEAR_RECALL,
 541          "simplicity_order": ["Logistic Regression", "K-Nearest Neighbors", "Random Forest"],
 542          "positive_class": {"name": "PHISHING", "value": 1},
 543          "negative_class": {"name": "SAFE", "value": 0},
 544          "model_name": chosen_name,
 545          "selection_reason": reason,
 546          "uses_internal_scaler": bool(uses_scaler),
 547          "feature_names": list(FEATURE_NAMES),
 548          "features": FEATURE_INFO,
 549          "hyperparameters": hyperparameters,
 550          "dataset": {
 551              "name": source_info.get("dataset_name", "PhiUSIIL Phishing URL Dataset"),
```

Store dataset provenance, label mapping, cleaning totals and the number of examples in each split.

```text
 552              "uci_id": source_info.get("uci_id", 967),
 553              "official_url": source_info.get("official_url"),
 554              "citation": source_info.get("citation"),
 555              "paper": source_info.get("paper"),
 556              "authors": source_info.get("authors"),
 557              "license": source_info.get("license"),
 558              "license_url": source_info.get("license_url"),
 559              "doi": source_info.get("doi"),
 560              "confirmation_method": source_info.get("confirmation_method"),
 561              "official_label_sentence": source_info.get("official_label_sentence"),
 562              "published_rows": source_info.get("published_rows"),
 563              "published_legitimate": source_info.get("published_legitimate"),
 564              "published_phishing": source_info.get("published_phishing"),
 565              "file_sha256": sha256_file(DATA_PATH),
 566              "download_method": source_info.get("download_method"),
 567          },
 568          "label_mapping": {
 569              "original": "1 = legitimate, 0 = phishing",
 570              "project": "0 = SAFE, 1 = PHISHING",
 571              "conversion": "phishing_label = 1 - original_label",
 572          },
 573          "cleaning": summary,
 574          "counts": {
 575              "modeling_rows": int(len(sampled)),
 576              "modeling_safe": sample_safe,
 577              "modeling_phishing": sample_phishing,
 578              "train_rows": int(len(y_train)),
 579              "validation_rows": int(len(y_val)),
 580              "test_rows": int(len(y_test)),
 581              "train_safe": train_safe,
 582              "train_phishing": train_phishing,
 583              "validation_safe": val_safe,
 584              "validation_phishing": val_phishing,
 585              "test_safe": test_safe,
 586              "test_phishing": test_phishing,
 587              "overlap_count": int(overlap),
 588          },
 589          "validation_comparison": rounded_validation,
 590          "selected_validation": {
 591              "accuracy": round(float(chosen_scores["accuracy"]), 6),
```

Save validation scores separately from held-out results. Include the four confusion counts, example, split proportions and limitations.

```text
 592              "precision": round(float(chosen_scores["precision"]), 6),
 593              "recall": round(float(chosen_scores["recall"]), 6),
 594              "f1": round(float(chosen_scores["f1"]), 6),
 595          },
 596          "test_metrics": {
 597              "accuracy": round(float(test_scores["accuracy"]), 6),
 598              "precision": round(float(test_scores["precision"]), 6),
 599              "recall": round(float(test_scores["recall"]), 6),
 600              "f1": round(float(test_scores["f1"]), 6),
 601              "confusion_matrix": [
 602                  [true_negative, false_positive],
 603                  [false_negative, true_positive],
 604              ],
 605              "true_negative": true_negative,
 606              "false_positive": false_positive,
 607              "false_negative": false_negative,
 608              "true_positive": true_positive,
 609          },
 610          "prediction_example": example,
 611          "split": {"train": 0.70, "validation": 0.15, "test": 0.15, "method": "stratified"},
 612          "warnings": [
 613              "Scores describe a stratified subset of about 30,000 real records, not the full 235,795-row dataset.",
 614              "A random split of URLs can still place related domains in both training and testing.",
 615              "The phishing probability is a model score, not a calibrated guarantee of real-world risk.",
 616          ],
 617      }
```

Write the JSON used by Model Insights. Refresh the Markdown reports and print completion. The final guard runs main only when called as a script.

```text
 619      metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
 620      with open(metrics_path, "w", encoding="utf-8") as handle:
 621          json.dump(metrics, handle, indent=2)
 622      print("Saved", metrics_path)
 624      write_all_docs(metrics)
 625      print("Wrote MODEL_REPORT.md, DATASET.md, VIVA.md, and README.md")
 626      print("Training finished. Start the website with: python app.py")
 629  if __name__ == "__main__":
 630      main()
```

## [backend/app.py](backend/app.py)

Serve a prediction using the saved estimator and return enough context for the result screen.

Import the server, JSON responses, prediction library and the shared URL rules. Set paths for the model, metrics and two chart images.

```text
   1  """Small FastAPI service that scores one URL with the saved model.
   3  The model is loaded once, when this file starts. A prediction does not train it again.
   4  Feature extraction reads the URL text only and does not visit the address.
   5  """
   7  import json
   8  import os
   9  import traceback
  10  from urllib.parse import urlparse
  12  import joblib
  13  import numpy as np
  14  import uvicorn
  15  from fastapi import FastAPI, Request
  16  from fastapi.middleware.cors import CORSMiddleware
  17  from fastapi.responses import JSONResponse, Response
  19  from features import API_MAX_URL_LENGTH, FEATURE_NAMES, features_from_normalized, normalize_url
  22  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  23  MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
  24  METRICS_PATH = os.path.join(BASE_DIR, "results", "metrics.json")
  25  CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "results", "confusion_matrix.png")
  26  COMPARISON_CHART_PATH = os.path.join(BASE_DIR, "results", "model_comparison.png")
```

Set the response disclaimer and local development origins. Keep the model, report and loading error in memory.

```text
  27  MAX_BODY_BYTES = 32 * 1024
  29  DISCLAIMER = (
  30      "URLGuard uses structural URL features and machine learning. "
  31      "Predictions may be incorrect. Do not rely on this result as a security guarantee."
  32  )
  34  app = FastAPI(title="URLGuard API")
  35  # Local lab pages may use port 3000, or another port if 3000 is already taken.
  36  app.add_middleware(
  37      CORSMiddleware,
  38      allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
  39      allow_methods=["*"],
  40      allow_headers=["*"],
  41  )
  43  bundle = None
  44  metrics = None
  45  load_error = None
```

Reset the cached state before loading. If the model file is missing, record the message that prediction requests should receive.

```text
  48  def load_artifacts():
  49      """Load the model and metrics one time."""
  50      global bundle, metrics, load_error
  51      bundle = None
  52      metrics = None
  53      load_error = None
  55      if not os.path.exists(MODEL_PATH):
  56          load_error = "The trained model file was not found. Run python train.py before starting predictions."
  57          print(load_error)
```

Load the saved bundle. Check its feature order against the current extractor before making the estimator available. Record any loading failure.

```text
  58      else:
  59          try:
  60              loaded = joblib.load(MODEL_PATH)
  61              if loaded.get("feature_names") != list(FEATURE_NAMES):
  62                  load_error = "The saved model uses a different feature list. Run python train.py again."
  63                  print(load_error)
  64              else:
  65                  bundle = loaded
  66                  print("Loaded model:", bundle.get("model_name"))
  67          except Exception as error:
  68              load_error = "The model file could not be loaded. Run python train.py again."
  69              print(load_error)
  70              print(error)
```

Read the saved metrics for Model Insights if the report exists.

```text
  72      if os.path.exists(METRICS_PATH):
  73          with open(METRICS_PATH, encoding="utf-8") as handle:
  74              metrics = json.load(handle)
  75      else:
  76          print("metrics.json was not found. Model Insights will be unavailable until you train.")
  78      print("The model stays in memory. Restart the API if you train again.")
```

Wrap one ordered vector in a one-row matrix. Get class probabilities and find the desired class by its label. Return that probability together with the model’s predicted label.

```text
  81  def class_probability(model, vector, class_value):
  82      """Return predict_proba for one class, using the model's own class order."""
  83      matrix = np.asarray([vector], dtype=float)
  84      probabilities = model.predict_proba(matrix)[0]
  85      classes = [int(label) for label in model.classes_]
  86      if class_value not in classes:
  87          raise RuntimeError("The model does not contain class " + str(class_value) + ".")
  88      predicted = int(model.predict(matrix)[0])
  89      return float(probabilities[classes.index(class_value)]), predicted
```

Translate measured traits into short display sentences. IP and HTTPS always get a sentence. Other traits add sentences only when the corresponding conditions hold.

```text
  92  def build_reasons(features):
  93      """Describe URL traits that were measured. These do not prove why the model decided."""
  94      reasons = []
  95      if features["is_ip_address"] == 1:
  96          reasons.append("The hostname is an IP address.")
  97      else:
  98          reasons.append("No IP address was found in the hostname.")
 100      if features["uses_https"] == 1:
 101          reasons.append("HTTPS is used. This encrypts the connection, but it does not prove the site is legitimate.")
 102      else:
 103          reasons.append("The URL does not use HTTPS.")
 105      if features["at_count"] > 0:
 106          reasons.append("The URL contains an @ symbol.")
 107      if features["subdomain_count"] >= 2:
 108          reasons.append("The hostname contains multiple extra subdomain labels.")
 109      elif features["subdomain_count"] == 1:
 110          reasons.append("The hostname contains an extra subdomain label.")
 111      if features["suspicious_keyword_count"] > 0:
 112          reasons.append("The URL contains a login-related or account-related keyword.")
 113      if features["url_length"] >= 75:
 114          reasons.append("The URL is unusually long.")
 115      if features["hyphen_count"] >= 4:
 116          reasons.append("The URL contains several hyphens.")
 117      if features["digit_count"] >= 8:
 118          reasons.append("The URL contains many digits.")
 119      if features["question_mark_count"] > 0 or features["equals_count"] > 0:
 120          reasons.append("The URL contains query parameters.")
 121      if features["path_length"] >= 40:
 122          reasons.append("The URL path is long.")
 123      return reasons
```

Build five label–value pairs for the result card.

```text
 126  def build_details(features):
 127      return [
 128          {"label": "URL length", "value": str(features["url_length"]) + " characters"},
 129          {"label": "HTTPS", "value": "Yes" if features["uses_https"] == 1 else "No"},
 130          {"label": "IP address", "value": "Yes" if features["is_ip_address"] == 1 else "No"},
 131          {"label": "Subdomains", "value": str(features["subdomain_count"])},
 132          {"label": "Suspicious keywords", "value": str(features["suspicious_keyword_count"])},
 133      ]
```

Detect the addition of HTTPS and the removal of a leading www from parsed hosts for display notes.

```text
 136  def scheme_was_added(original, normalized):
 137      text = original.strip().lower()
 138      return not text.startswith("http://") and not text.startswith("https://") and normalized.startswith("https://")
 141  def www_was_removed(original, normalized):
 142      original_host = (urlparse(original.strip()).hostname or "").lower()
 143      normalized_host = (urlparse(normalized).hostname or "").lower()
 144      return original_host.startswith("www.") and not normalized_host.startswith("www.")
```

Build JSON errors with an HTTP status. Middleware rejects a declared content length above 32 KiB before passing the request on.

```text
 147  def error(message, status_code):
 148      return JSONResponse({"error": message}, status_code=status_code)
 151  @app.middleware("http")
 152  async def limit_request_size(request: Request, call_next):
 153      length = request.headers.get("content-length")
 154      if length is not None and length.isdigit() and int(length) > MAX_BODY_BYTES:
 155          return error("The request is too large.", 413)
 156      return await call_next(request)
```

Health returns a small readiness-of-service response.

```text
 159  @app.get("/api/health")
 160  def health():
 161      return {"status": "ok"}
```

Read the JSON request. Reject malformed JSON, missing url or a value of the wrong type.

```text
 164  @app.post("/api/predict")
 165  async def predict(request: Request):
 166      try:
 167          payload = await request.json()
 168      except Exception:
 169          return error("Send a JSON body with a url field.", 400)
 171      if not isinstance(payload, dict) or "url" not in payload:
 172          return error('Send JSON like {"url": "https://example.com"}.', 400)
 174      url = payload.get("url")
 175      if not isinstance(url, str):
 176          return error("The url field must be a string.", 400)
```

Normalize with the API length limit and measure the result once. Return a client error for invalid input or 503 if the estimator is unavailable.

```text
 178      try:
 179          normalized = normalize_url(url, max_length=API_MAX_URL_LENGTH)
 180          features = features_from_normalized(normalized)
 181      except ValueError as exc:
 182          return error(str(exc), 400)
 184      if bundle is None or bundle.get("model") is None:
 185          return error(load_error or "The trained model is not available.", 503)
 187      try:
```

Order the twelve values, request the probability for phishing class 1 and handle prediction failures or unexpected labels.

```text
 188          vector = [features[name] for name in FEATURE_NAMES]
 189          phishing_probability, predicted = class_probability(bundle["model"], vector, 1)
 190      except Exception:
 191          traceback.print_exc()
 192          return error("The model could not score this URL.", 500)
 194      if predicted not in (0, 1):
 195          return error("The model returned an unexpected label.", 500)
 197      notes = []
 198      if scheme_was_added(url, normalized):
```

Assemble normalization notes and return the class, rounded probability, model name, observed traits, display details and normalized text.

```text
 199          notes.append("No scheme was entered, so the URL was analyzed as HTTPS.")
 200      if www_was_removed(url, normalized):
 201          notes.append(
 202              "A leading www. was removed before scoring. Every legitimate URL in the training file starts with www, so that prefix is not treated as proof of safety."
 203          )
 204      note = " ".join(notes) if notes else None
 206      return {
 207          "prediction": "PHISHING" if predicted == 1 else "SAFE",
 208          "phishing_probability": round(phishing_probability, 6),
 209          "model": bundle.get("model_name", "Trained model"),
 210          "reasons": build_reasons(features),
 211          "details": build_details(features),
 212          "analyzed_url": normalized,
 213          "scheme_note": note,
 214          "disclaimer": DISCLAIMER,
 215      }
```

Return saved evaluation data and the local paths where the browser can request its two chart images.

```text
 218  @app.get("/api/model-info")
 219  def model_info():
 220      if metrics is None:
 221          return error("Model results were not found. Run python train.py first.", 503)
 222      payload = dict(metrics)
 223      payload["confusion_matrix_image"] = "/api/confusion-matrix"
 224      payload["comparison_chart_image"] = "/api/comparison-chart"
 225      return payload
```

Read an existing PNG as bytes. A missing image gets an error. The next two handlers use this helper for their respective charts.

```text
 228  def png_response(path, missing_message):
 229      if not os.path.exists(path):
 230          return error(missing_message, 503)
 231      with open(path, "rb") as handle:
 232          payload = handle.read()
 233      return Response(content=payload, media_type="image/png")
 236  @app.get("/api/confusion-matrix")
 237  def confusion_matrix_image():
 238      return png_response(CONFUSION_MATRIX_PATH, "The confusion matrix image was not found. Run python train.py.")
 241  @app.get("/api/comparison-chart")
 242  def comparison_chart_image():
 243      return png_response(COMPARISON_CHART_PATH, "The comparison chart was not found. Run python train.py.")
```

List the API routes at the root. Load artifacts on import. When run directly, read PORT and start Uvicorn on the local interface.

```text
 246  @app.get("/")
 247  def index():
 248      return {
 249          "name": "URLGuard API",
 250          "health": "/api/health",
 251          "predict": "/api/predict",
 252          "model_info": "/api/model-info",
 253      }
 256  load_artifacts()
 259  if __name__ == "__main__":
 260      # Port 5000 matches the lab instructions. Set PORT if that port is already taken.
 261      port = int(os.environ.get("PORT", "5000"))
 262      print("URLGuard API: http://127.0.0.1:" + str(port))
 263      uvicorn.run(app, host="127.0.0.1", port=port)
```

## [backend/download_data.py](backend/download_data.py)

Dataset preparation. These functions obtain the public training file and record how its labels were confirmed.

| Source lines | Function | What the lines do |
| --- | --- | --- |
| 37–45 | `column_lookup` | Find a matching source column despite spelling or case differences. |
| 48–54 | `label_sentence_is_present` | Check the source description for the expected class mapping. |
| 57–71 | `manual_download_instructions` | Describe how to supply the CSV manually. |
| 74–79 | `download_with_ucimlrepo` | Fetch UCI dataset 967 and return its parts. |
| 82–89 | `read_manual_csv` | Read the manually supplied CSV. |
| 92–110 | `collect_label_text` | Gather label descriptions from the available source metadata. |
| 113–129 | `build_frame_from_parts` | Pair URL and label columns into one frame. |
| 132–138 | `save_outputs` | Write the two-column CSV and provenance JSON. |
| 141–151 | `summarize_original_labels` | Count the two original target labels. |
| 154–167 | `metadata_to_dict` | Convert metadata to a regular dictionary for lookup. |
| 170–289 | `main` | Choose a data source, verify counts and labels, record provenance, and write files. |

## [backend/write_docs.py](backend/write_docs.py)

Reporting after training. This file converts a results dictionary into readable Markdown.

| Source lines | Function | What the lines do |
| --- | --- | --- |
| 15–17 | `write` | Write one document with a final newline. |
| 20–21 | `four` | Format a metric to four decimal places. |
| 24–25 | `percent` | Show a proportion as a percentage. |
| 28–29 | `comma` | Format row counts with digit grouping. |
| 32–46 | `leader_sentence` | Find and describe the leading score for one metric. |
| 49–68 | `comparison_table` | Turn validation rows into a Markdown comparison table. |
| 71–75 | `hyperparameter_lines` | Format model settings for the report. |
| 78–79 | `safe_cell` | Escape table-cell content. |
| 82–423 | `model_report` | Build the model report from dataset, parameters, validation and test evidence. |
| 426–520 | `dataset_doc` | Describe the source, class mapping and cleaning decisions. |
| 523–854 | `viva_doc` | Build viva questions and answers with measured values. |
| 857–1084 | `readme_doc` | Build setup instructions, feature definitions and a results summary. |
| 1087–1091 | `write_all_docs` | Write the four named Markdown documents to the repository root. |

## Website and test files

| File | What to follow |
| --- | --- |
| frontend/app/page.tsx and components/ScannerHero.tsx | Build the landing page and show the scanner. |
| frontend/components/UrlScanner.tsx | Pankaj follows onSubmit through validation, POST /api/predict, loading and errors. |
| frontend/components/PredictionResult.tsx | Shivam shows where the model name, probability, observed URL traits and result appear. |
| frontend/app/model-insights/page.tsx | Fetches /api/model-info and presents dataset counts, validation scores, held-out results and chart images. |
| frontend/components/SiteHeader.tsx and ThemeToggle.tsx | Navigation and theme controls. |
| backend/tests/test_features.py | Checks normalization, feature values and feature order. |
| backend/tests/test_prediction.py | Checks model selection, saved data, split membership and API responses. |
