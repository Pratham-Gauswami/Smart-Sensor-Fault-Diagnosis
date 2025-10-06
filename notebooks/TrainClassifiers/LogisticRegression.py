import pandas as pd
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
from datetime import datetime

# # 1️⃣ Load train and test datasets
# train_df = pd.read_csv('../data/features_train.csv')
# test_df = pd.read_csv('../data/features_test.csv')

train_df = pd.read_csv(Path(__file__).parent.parent.parent / 'data' / 'features_train.csv')
test_df = pd.read_csv(Path(__file__).parent.parent.parent / 'data' / 'features_test.csv')
# log_filename = (Path(__file__).parent.parent.parent / 'loghub' / 'model_run_log.txt')

# 2️⃣ Combine datasets
df = pd.concat([train_df, test_df], ignore_index=True)

# 3️⃣ Separate features and labels
X = df.drop(columns=['label'])
y = df['label']

# 4️⃣ Handle categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

if len(categorical_cols) > 0:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(X[categorical_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))
    X = pd.concat([X[numeric_cols].reset_index(drop=True),
                   encoded_df.reset_index(drop=True)], axis=1)

# 5️⃣ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6️⃣ Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 7️⃣ Train Logistic Regression
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')  # class_weight for imbalanced dataset
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# 8️⃣ Configure logging with auto-incremented filename
log_folder = Path(__file__).parent.parent.parent / 'loghub'
log_folder.mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist

base_name = 'model_run'
extension = '.txt'

# Find next available log number
i = 1
while (log_folder / f"{base_name}_{i}{extension}").exists():
    i += 1

log_filename = log_folder / f"{base_name}_{i}{extension}"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 9️⃣ Evaluate and log performance
logging.info("=== Logistic Regression Evaluation ===")
logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
logging.info(f"Precision: {precision_score(y_test, y_pred):.4f}")
logging.info(f"Recall: {recall_score(y_test, y_pred):.4f}")
logging.info(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

print(f"Evaluation logged to {log_filename}")