# 02_load_parquet.py
import pandas as pd
import os
import matplotlib.pyplot as plt

# Paths to your metadata files
# Paths to your metadata files
train_csv = "../data/imad/BrushlessMotor/train/attributes_normal_source_train.csv"   # adjust to your actual path
test_csv = "../data/imad/BrushlessMotor/test/attributes_normal_source_test.csv"

# Load metadata
train_meta = pd.read_csv(train_csv)
test_meta = pd.read_csv(test_csv)

print("Train shape:", train_meta.shape)
print("Test shape:", test_meta.shape)
print("\nSample from train metadata:")
print(train_meta.head())

# --- Load and visualize parquet logs ---
for col in ["imp23absu_mic", "ism330dhcx_acc", "ism330dhcx_gyro"]:
    parquet_path = train_meta.iloc[0][col]
    
    # Adjust relative path if needed
    if not os.path.isabs(parquet_path):
        parquet_path = os.path.join("../data/imad/BrushlessMotor/train", parquet_path)
    
    print(f"\nLoading {col} from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(df.head())
    print("Shape:", df.shape)
    
    # --- Quick Plot ---
    plt.figure(figsize=(10, 4))
    for c in df.columns[:3]:  # just plot first 3 columns (like x,y,z or first 3 signals)
        plt.plot(df[c].values, label=c, alpha=0.7)
    plt.title(f"Sample signals from {col}")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
