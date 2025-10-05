# 05_feature_extraction_from_pkl.py

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from tqdm import tqdm

# 1️⃣ Load processed dataset
print("Loading dataset...")
dataset = pd.read_pickle("../data/processed_dataset.pkl")
print(f"Loaded dataset with shape: {dataset.shape}")

# 2️⃣ Helper: extract statistical & frequency features
def extract_features(signal):
    # Handle NaNs or empty signals
    if signal is None or len(signal) == 0:
        return {
            "mean": np.nan, "std": np.nan, "skew": np.nan, "kurt": np.nan,
            "max": np.nan, "min": np.nan, "freq_peak": np.nan, "signal_energy": np.nan
        }

    # Normalize (z-score)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    # Statistical features
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    skew_val = skew(signal)
    kurt_val = kurtosis(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)

    # Frequency-domain features
    fft_vals = np.abs(fft(signal))
    freq_peak = np.argmax(fft_vals)
    energy = np.sum(fft_vals**2)

    return {
        "mean": mean_val, "std": std_val, "skew": skew_val, "kurt": kurt_val,
        "max": max_val, "min": min_val, "freq_peak": freq_peak, "signal_energy": energy
    }

# 3️⃣ Apply to all rows
records = []
for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
    feats = extract_features(row["waveform"])
    feats.update({
        "segment_id": row["segment_id"],
        "split": row["split"],
        "domain": row["domain"],
        "label": row["label"]
    })
    records.append(feats)

# 4️⃣ Convert to DataFrame
features_df = pd.DataFrame(records)
print("Feature DataFrame shape:", features_df.shape)
print(features_df.head())

# 5️⃣ Split and save
train_df = features_df[features_df["split"] == "train"]
test_df = features_df[features_df["split"] == "test"]

train_df.to_csv("../data/features_train.csv", index=False)
test_df.to_csv("../data/features_test.csv", index=False)

print("✅ Feature extraction completed and saved:")
print("   - ../data/features_train.csv")
print("   - ../data/features_test.csv")
