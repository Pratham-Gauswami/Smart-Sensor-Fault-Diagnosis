import pandas as pd

parquet_path = "../data/imad/BrushlessMotor/train/imp23absu_mic_20240423_10_20_21_DataLog_0.parquet"  

# load parquet
df = pd.read_parquet(parquet_path)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())