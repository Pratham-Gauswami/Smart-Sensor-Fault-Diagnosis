import re
import pandas as pd
from pathlib import Path

# === Define folder paths ===
log_folder = Path(__file__).parent.parent.parent / 'imec' / 'loghub'
print(f"🔍 Scanning log files in: {log_folder}")

# === Regex pattern to extract metrics ===
pattern = re.compile(
    r"Accuracy:\s*([\d.]+).*?"
    r"Precision:\s*([\d.]+).*?"
    r"Recall:\s*([\d.]+).*?"
    r"F1-Score:\s*([\d.]+)",
    re.DOTALL
)

records = []

# === Collect all matching log files ===
log_files = list(log_folder.glob("model_run_*.txt")) + \
             list(log_folder.glob("model_run_xgb_*.txt")) + \
             list(log_folder.glob("Rfmodel_run_*.txt"))

if not log_files:
    print("⚠️ No log files found in loghub folder. Check filenames and rerun training scripts.")
else:
    for log_file in log_files:
        text = log_file.read_text(errors="ignore")
        match = pattern.search(text)
        if match:
            accuracy, precision, recall, f1 = map(float, match.groups())

            # Detect model type from filename
            fname = log_file.name.lower()
            if "xgb" in fname:
                model_type = "XGBoost"
            elif "rf" in fname:
                model_type = "RandomForest"
            elif "logreg" in fname or "model_run" in fname:
                model_type = "LogisticRegression"
            else:
                model_type = "Unknown"

            records.append({
                "ModelType": model_type,
                "LogFile": log_file.name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "FullPath": str(log_file)
            })

# === Build DataFrame ===
if records:
    df = pd.DataFrame(records)
    df = df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

    print("\n=== 🧩 Model Comparison Summary ===")
    print(df.to_string(index=False))

    # Save summary
    out_path = log_folder / "model_comparison_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Summary saved to: {out_path}")

    # Print best performer
    best = df.iloc[0]
    print(f"\n🏆 Best Model: {best['ModelType']} ({best['LogFile']}) | F1-Score: {best['F1-Score']:.4f}")

else:
    print("⚠️ No valid metrics found in any log files.")
