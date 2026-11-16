"""Download the official PhiUSIIL dataset and save url + original label.

The file data/urls.csv keeps the UCI label values:
    1 = legitimate
    0 = phishing

train.py performs the project conversion phishing_label = 1 - original_label
only after this source mapping has been confirmed.

This script never creates fake URLs.
"""

import json
import os
import sys

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "urls.csv")
SOURCE_INFO_PATH = os.path.join(DATA_DIR, "source_info.json")
MANUAL_CSV_PATH = os.path.join(DATA_DIR, "PhiUSIIL_Phishing_URL_Dataset.csv")

UCI_PAGE = "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset"
UCI_ID = 967
OFFICIAL_LABEL_SENTENCE = (
    "Label 1 corresponds to a legitimate URL, label 0 to a phishing URL"
)

PUBLISHED_ROWS = 235795
PUBLISHED_LEGITIMATE = 134850
PUBLISHED_PHISHING = 100945


def column_lookup(frame, accepted_names):
    """Return a Series for the first matching column name, ignoring case."""
    if frame is None:
        return None
    lookup = {str(column).lower(): column for column in frame.columns}
    for name in accepted_names:
        if name.lower() in lookup:
            return frame[lookup[name.lower()]]
    return None


def label_sentence_is_present(text):
    """True only when the text contains the official class-label sentence."""
    if not text:
        return False
    lowered = " ".join(str(text).lower().split())
    expected = " ".join(OFFICIAL_LABEL_SENTENCE.lower().split())
    return expected in lowered


def manual_download_instructions():
    print("")
    print("The official dataset could not be downloaded automatically.")
    print("Download the original file from the UCI Machine Learning Repository:")
    print(UCI_PAGE)
    print("")
    print("Steps:")
    print("1. Open the UCI link above.")
    print("2. Download PhiUSIIL_Phishing_URL_Dataset.csv (the official dataset file).")
    print("3. Save that file as:")
    print("   " + MANUAL_CSV_PATH)
    print("4. Run this script again: python download_data.py")
    print("")
    print("The script will read that official CSV. It will not invent URLs or labels.")
    print("")


def download_with_ucimlrepo():
    """Fetch dataset 967. Return (features, targets, metadata, variables) or raise."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=UCI_ID)
    return dataset.data.features, dataset.data.targets, dataset.metadata, dataset.variables


def read_manual_csv():
    """Read a CSV the user placed after a failed automatic download."""
    if not os.path.exists(MANUAL_CSV_PATH):
        return None
    print("Reading the manually downloaded official CSV:")
    print(MANUAL_CSV_PATH)
    frame = pd.read_csv(MANUAL_CSV_PATH)
    return frame


def collect_label_text(metadata, variables):
    """Join the metadata fields that describe the class labels."""
    parts = []
    if isinstance(metadata, dict):
        for key in ("additional_info", "abstract", "summary"):
            value = metadata.get(key)
            if value:
                parts.append(str(value))
    if variables is not None:
        try:
            for _, row in variables.iterrows():
                name = str(row.get("name", "")).lower()
                description = str(row.get("description", "") or "")
                if name in ("label", "class") or description:
                    if name in ("label", "class"):
                        parts.append(name + ": " + description)
        except Exception:
            parts.append(str(variables))
    return "\n".join(parts)


def build_frame_from_parts(features, targets):
    url_series = column_lookup(features, ["URL", "url"])
    label_series = column_lookup(targets, ["label", "class", "target"])
    if label_series is None:
        label_series = column_lookup(features, ["label", "class"])
    if url_series is None or label_series is None:
        feature_columns = [] if features is None else list(features.columns)
        target_columns = [] if targets is None else list(targets.columns)
        raise RuntimeError(
            "Could not find the URL and label columns. "
            + "Feature columns: "
            + str(feature_columns)
            + ". Target columns: "
            + str(target_columns)
        )
    frame = pd.DataFrame({"url": url_series, "label": label_series})
    return frame


def save_outputs(frame, source_info):
    os.makedirs(DATA_DIR, exist_ok=True)
    frame.to_csv(CSV_PATH, index=False)
    with open(SOURCE_INFO_PATH, "w", encoding="utf-8") as handle:
        json.dump(source_info, handle, indent=2)
    print("Saved", len(frame), "rows to", CSV_PATH)
    print("Saved source notes to", SOURCE_INFO_PATH)


def summarize_original_labels(frame):
    numeric = pd.to_numeric(frame["label"], errors="coerce")
    legitimate = int((numeric == 1).sum())
    phishing = int((numeric == 0).sum())
    other = int((~numeric.isin([0, 1])).sum())
    print("Rows in downloaded table:", len(frame))
    print("Original label 1 (legitimate, if the UCI mapping holds):", legitimate)
    print("Original label 0 (phishing, if the UCI mapping holds):", phishing)
    print("Other or missing label values:", other)
    print("Published UCI size:", PUBLISHED_ROWS, "rows;", PUBLISHED_LEGITIMATE, "legitimate;", PUBLISHED_PHISHING, "phishing.")
    return legitimate, phishing, other


def metadata_to_dict(metadata):
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        raw = metadata
    else:
        raw = dict(metadata)
    clean = {}
    for key, value in raw.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    features = None
    targets = None
    metadata = {}
    variables = None
    download_method = ""

    try:
        print("Downloading PhiUSIIL Phishing URL Dataset from UCI (id 967)...")
        features, targets, metadata, variables = download_with_ucimlrepo()
        download_method = "ucimlrepo fetch_ucirepo(id=967)"
        print("Download finished.")
    except Exception as error:
        print("Automatic download failed.")
        print(type(error).__name__ + ": " + str(error))
        manual = read_manual_csv()
        if manual is None:
            manual_download_instructions()
            sys.exit(1)
        features = manual
        targets = manual
        download_method = "manual official CSV placed in data/PhiUSIIL_Phishing_URL_Dataset.csv"
        metadata = {
            "name": "PhiUSIIL Phishing URL Dataset",
            "uci_id": UCI_ID,
            "repository_url": UCI_PAGE,
            "note": "Loaded from a manually downloaded official UCI CSV because ucimlrepo failed.",
        }

    try:
        frame = build_frame_from_parts(features, targets)
    except Exception as error:
        print("The downloaded file could not be prepared.")
        print(str(error))
        print("No fake data was created.")
        sys.exit(1)

    legitimate, phishing, other = summarize_original_labels(frame)
    metadata_dict = metadata_to_dict(metadata)
    label_text = collect_label_text(metadata_dict, variables)
    sentence_in_download = label_sentence_is_present(label_text)
    counts_match = (
        legitimate == PUBLISHED_LEGITIMATE
        and phishing == PUBLISHED_PHISHING
        and other == 0
        and len(frame) == PUBLISHED_ROWS
    )
    # The UCI dataset page states the class codes under Additional Variable Information.
    # ucimlrepo sometimes leaves the label description blank, so we also require the
    # downloaded class sizes to match the published legitimate and phishing counts.
    if not counts_match:
        print("Refusing to save the file.")
        print("The downloaded class counts do not match the published UCI sizes.")
        print("No fake data was created.")
        sys.exit(1)
    if sentence_in_download:
        confirmation_method = "The downloaded UCI metadata contains the official class-label sentence, and the class counts match the published dataset size."
    else:
        print("The ucimlrepo metadata did not include the class-label sentence.")
        print("The official UCI dataset page states:")
        print(OFFICIAL_LABEL_SENTENCE)
        print("Source page:", UCI_PAGE)
        print("Downloaded counts match that page: label 1 =", legitimate, "and label 0 =", phishing)
        confirmation_method = (
            "The official UCI dataset page states '"
            + OFFICIAL_LABEL_SENTENCE
            + "'. The downloaded file matches the published size: "
            + str(PUBLISHED_LEGITIMATE)
            + " rows with label 1 and "
            + str(PUBLISHED_PHISHING)
            + " rows with label 0."
        )
        label_text = (
            label_text
            + "\n\nVerified on the official UCI dataset page ("
            + UCI_PAGE
            + "): "
            + OFFICIAL_LABEL_SENTENCE
        ).strip()

    creators = metadata_dict.get("creators", "")
    if isinstance(creators, list):
        creators = ", ".join(str(item) for item in creators)

    source_info = {
        "dataset_name": metadata_dict.get("name", "PhiUSIIL Phishing URL Dataset"),
        "uci_id": UCI_ID,
        "official_url": UCI_PAGE,
        "doi": "https://doi.org/10.1016/j.cose.2023.103545",
        "citation": "Prasad, A. & Chandra, S. (2024). PhiUSIIL Phishing URL (Website) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.1016/j.cose.2023.103545.",
        "paper": "Prasad, A., & Chandra, S. (2023). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. Computers & Security, 103545. https://doi.org/10.1016/j.cose.2023.103545",
        "confirmation_method": confirmation_method,
        "authors": "Arvind Prasad and Shalini Chandra",
        "creators_field": creators,
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "download_method": download_method,
        "original_label_mapping": "1 = legitimate, 0 = phishing",
        "official_label_sentence": OFFICIAL_LABEL_SENTENCE,
        "label_mapping_confirmed": True,
        "label_text": label_text,
        "project_label_mapping": "0 = SAFE, 1 = PHISHING",
        "conversion": "phishing_label = 1 - original_label",
        "published_rows": PUBLISHED_ROWS,
        "published_legitimate": PUBLISHED_LEGITIMATE,
        "published_phishing": PUBLISHED_PHISHING,
        "downloaded_rows": int(len(frame)),
        "downloaded_label_1": legitimate,
        "downloaded_label_0": phishing,
        "downloaded_other_labels": other,
        "kept_columns": ["url", "label"],
        "unused_note": "Webpage-content features from the original 54-column file are not saved. The project builds features from the URL string only.",
        "metadata": metadata_dict,
    }

    # Keep the original UCI codes in urls.csv. train.py converts them later.
    save_outputs(frame, source_info)
    print("Confirmed label note:", OFFICIAL_LABEL_SENTENCE)
    print("Original labels were saved unchanged. train.py will convert them.")


if __name__ == "__main__":
    main()
