import joblib
import pandas as pd
from pathlib import Path
import shutil

# Paths
base_folder = Path(__file__).parent.parent.parent / 'imec'
log_folder = base_folder / 'loghub'
print(f"📂 Base folder: {base_folder}")
print(f"📂 Log folder: {log_folder}")
models_folder = base_folder / 'models'
best_folder = base_folder / 'best_model'

# Ensure folders exist
models_folder.mkdir(parents=True, exist_ok=True)
best_folder.mkdir(parents=True, exist_ok=True)

# Load comparison summary
summary_file = log_folder / "model_comparison_summary.csv"
if not summary_file.exists():
    raise FileNotFoundError("⚠️ Run compare_models.py first to generate model_comparison_summary.csv")

df = pd.read_csv(summary_file)
best_model_name = df.iloc[0]['ModelType']
print(f"🏆 Best model detected: {best_model_name}")

# Look for saved model
model_file = models_folder / f"{best_model_name}.pkl"
if not model_file.exists():
    raise FileNotFoundError(f"❌ Model file not found: {model_file}")

# Copy to best_model folder
best_model_path = best_folder / "best_model.pkl"
shutil.copy(model_file, best_model_path)
print(f"✅ Best model saved as: {best_model_path}")
