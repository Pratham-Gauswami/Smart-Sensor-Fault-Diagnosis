# 04_build_dataset_multidomain.py
import os
import pandas as pd
from tqdm import tqdm

# Base paths
train_folder = "../data/imad/BrushlessMotor/train/"
test_folder = "../data/imad/BrushlessMotor/test/"

# Train CSVs
train_csvs = {
    "source_normal": os.path.join(train_folder, "attributes_normal_source_train.csv"),
    "target_normal": os.path.join(train_folder, "attributes_normal_target_train.csv"),
}

# Test CSVs
test_csvs = {
    "source_normal": os.path.join(test_folder, "attributes_anomaly_source_test.csv"),
    "source_anomaly": os.path.join(test_folder, "attributes_anomaly_target_test.csv"),
    "target_normal": os.path.join(test_folder, "attributes_normal_source_test.csv"),
    "target_anomaly": os.path.join(test_folder, "attributes_normal_target_test.csv"),
}

# Helper: load metadata and add domain info
def load_meta(file, split, domain, label):
    df = pd.read_csv(file)
    df["split"] = split
    df["domain"] = domain      # source vs target
    df["label"] = label        # normal=0, anomaly=1
    return df

# Load all CSVs
meta_frames = []
meta_frames.append(load_meta(train_csvs["source_normal"], "train", "source", 0))
meta_frames.append(load_meta(train_csvs["target_normal"], "train", "target", 0))

meta_frames.append(load_meta(test_csvs["source_normal"], "test", "source", 0))
meta_frames.append(load_meta(test_csvs["source_anomaly"], "test", "source", 1))
meta_frames.append(load_meta(test_csvs["target_normal"], "test", "target", 0))
meta_frames.append(load_meta(test_csvs["target_anomaly"], "test", "target", 1))

meta = pd.concat(meta_frames, ignore_index=True)

print("Metadata shape:", meta.shape)
print(meta.head())

# ---- Now load parquet waveforms ----
def load_waveform(parquet_file, split):
    folder = train_folder if split == "train" else test_folder
    file_path = os.path.join(folder, parquet_file)
    try:
        df = pd.read_parquet(file_path)
        return df["MIC [Waveform]"].values  # numpy array
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

records = []
for _, row in tqdm(meta.iterrows(), total=len(meta)):
    parquet_file = row["imp23absu_mic"]  # column pointing to parquet
    waveform = load_waveform(parquet_file, row["split"])
    if waveform is not None:
        records.append({
            "segment_id": row["segment_id"],
            "split": row["split"],
            "domain": row["domain"],
            "label": row["label"],
            "waveform": waveform
        })

dataset = pd.DataFrame(records)
print("Final dataset shape:", dataset.shape)
print(dataset.head())

# Save for reuse
dataset.to_pickle("../data/processed_dataset.pkl")
