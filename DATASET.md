# Dataset Notes

Jaypee Institute of Information Technology, Noida · Computer Applications

## Source

- UCI dataset name: PhiUSIIL Phishing URL (Website)
- UCI id: 967
- Official URL: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- Authors: Arvind Prasad and Shalini Chandra
- Dataset citation: Prasad, A. & Chandra, S. (2024). PhiUSIIL Phishing URL (Website) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.1016/j.cose.2023.103545.
- Introductory paper: Prasad, A., & Chandra, S. (2023). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. Computers & Security, 103545. https://doi.org/10.1016/j.cose.2023.103545
- License: Creative Commons Attribution 4.0 International (CC BY 4.0) (https://creativecommons.org/licenses/by/4.0/)
- How this copy was obtained: ucimlrepo fetch_ucirepo(id=967)
- SHA-256 of `backend/data/urls.csv`: `0112072734d5e58a0c319e82f3aea97cb6ae2803c21a32d1d52446afa382c40d`

The dataset was donated to UCI on 3 March 2024. The repository page describes 235,795 URL records, of which 134,850 are legitimate and 100,945 are phishing. The file saved by `download_data.py` has the same counts.

## Label Mapping

Exact original mapping, from the UCI dataset page under Additional Variable Information:

> Label 1 corresponds to a legitimate URL, label 0 to a phishing URL

So the original codes are:

- 1 = legitimate
- 0 = phishing

The project uses the opposite coding so that the positive class is phishing:

- 0 = SAFE
- 1 = PHISHING

Conversion, applied in `train.py` only after the source sentence and the published class counts were checked:

```text
phishing_label = 1 - original_label
```

Confirmation recorded for this file: The official UCI dataset page states 'Label 1 corresponds to a legitimate URL, label 0 to a phishing URL'. The downloaded file matches the published size: 134850 rows with label 1 and 100945 rows with label 0.

`data/urls.csv` still stores the original UCI codes. The conversion happens in memory during training. The website and the metrics use the project codes.

## Download

From the `backend` directory, with the virtual environment activated:

```bash
python download_data.py
```

The script calls `fetch_ucirepo(id=967)` from the `ucimlrepo` package. It keeps two columns, `url` and `label`, and writes `data/urls.csv` plus `data/source_info.json`.

If the automatic download fails, the script stops and tells you to download `PhiUSIIL_Phishing_URL_Dataset.csv` from the UCI page and save it as `backend/data/PhiUSIIL_Phishing_URL_Dataset.csv`. Run the script again after that. It will read the official file. It will not create substitute rows.

## Counts For This Project

| Stage | Rows |
|---|---:|
| Published and downloaded rows | 235,795 |
| Original label 1 (legitimate) | 134,850 |
| Original label 0 (phishing) | 100,945 |
| Removed invalid rows | 0 |
| Duplicate extras removed | 571 |
| Conflicting-label rows removed | 0 |
| Rows after cleaning | 235,224 |
| SAFE after conversion | 134,850 |
| PHISHING after conversion | 100,374 |
| Stratified modeling sample | 30,000 |
| Sample SAFE / PHISHING | 17,199 / 12,801 |

## Sampling

The full cleaned table is larger than this lab needs, and K-Nearest Neighbors compares a new URL with the stored training rows. `train.py` therefore takes a stratified sample of 30,000 real rows. Stratified means the sample keeps approximately the same safe/phishing mix as the cleaned data. `random_state = 42` makes the sample repeatable. Every selected row comes from the UCI file. None were generated.

The sample is then split 70% / 15% / 15% into training, validation, and test, again stratified, with the same seed. Duplicate URLs are removed before this split. Identical-URL overlap across the three splits in this run: 0.

Results in the model report describe this sample. They do not describe a test of all 235,795 rows.

## Why Only URL-Derived Features Are Used

The original PhiUSIIL table has 54 features. Many of them come from the webpage source, such as line counts, titles, iframes, and popup counts. URLGuard cannot see a page's source, because the safety rule is that a submitted URL is never opened, downloaded, or rendered.

Training therefore ignores those columns. `download_data.py` does not even save them. Both training and the FastAPI prediction call `extract_features(url)` and use the 12 numbers listed in `features.py`. If training had used page-content features, the saved model could not score a URL typed into the website.

## What This File Is Not

The training data is not fake, simulated, or randomly labeled. The only invented URL in the documentation is the parsing demonstration `https://example.com/login?user=1` in `VIVA.md`, and that demonstration is explicitly not a training row and not a model prediction.
